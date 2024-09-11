#
# Copyright 2015 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
Utility functions for data loading and training of VGSL networks.
"""
import torch
import torch.nn.functional as F
import numpy as np
import lightning.pytorch as L

import pyarrow as pa

from typing import (TYPE_CHECKING, Any, Callable, List, Literal, Optional,
                    Tuple, Union, Sequence)

from conformer_ocr.codec import ByT5Codec

from functools import partial
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import tempfile

from scipy.special import comb
from shapely.geometry import LineString

from torchvision.transforms import v2

from kraken.lib import functional_im_transforms as F_t
from kraken.lib.xml import XMLPage

if TYPE_CHECKING:
    from os import PathLike

__all__ = ['TextLineDataModule']

import logging

logger = logging.getLogger(__name__)


def _validation_worker_init_fn(worker_id):
    """ Fix random seeds so that augmentation always produces the same
        results when validating. Temporarily increase the logging level
        for lightning because otherwise it will display a message
        at info level about the seed being changed. """
    from lightning.pytorch import seed_everything
    seed_everything(42)


def collate_null(batch):
    return batch[0]


def collate_sequences(im, page_data):
    """
    Sorts and pads image data.
    """
    if isinstance(page_data[0][0], str):
        labels = [x for x, _ in page_data]
    else:
        max_label_len = max(len(x) for x, _ in page_data)
        labels = torch.stack([F.pad(x, pad=(0, max_label_len-len(x))) for x, _ in page_data]).long()
    label_lens = torch.LongTensor([len(x) for x, _ in page_data])
    curves = torch.stack([x for _, x in page_data])
    return {'image': im,
            'target': labels,
            'curves': curves,
            'target_lens': label_lens}


def optional_resize(img: 'Image.Image', max_size: int):
    """
    Resizing that return images with the longest side below `max_size`
    unchanged.

    Args:
        img: image to resize
        max_size: maximum length of any side of the image
    """
    w, h = img.size
    img_max = max(w, h)
    if img_max > max_size:
        if w > h:
            h = int(h * max_size/w)
            w = max_size
        else:
            w = int(w * max_size/h)
            h = max_size
        return img.resize((w, h))
    else:
        return img


class TextLineDataModule(L.LightningDataModule):
    def __init__(self,
                 training_data: Sequence[Union[str, 'PathLike']],
                 evaluation_data: Sequence[Union[str, 'PathLike']],
                 height: int = 0,
                 augmentation: bool = False,
                 batch_size: int = 16,
                 num_workers: int = 8,
                 reorder: Union[bool, str] = True,
                 normalize_whitespace: bool = True,
                 normalization: Optional[Literal['NFD', 'NFC', 'NFKD', 'NFKC']] = None):
        super().__init__()

        self.save_hyperparameters()

        self.tmpdir = tempfile.TemporaryDirectory(prefix='cocr', dir='/dev/shm')

        self.text_transforms: List[Callable[[str], str]] = []

        # built text transformations
        if normalization:
            self.text_transforms.append(partial(F_t.text_normalize, normalization=normalization))
        if normalize_whitespace:
            self.text_transforms.append(F_t.text_whitespace_normalize)
        if reorder:
            if reorder in ('L', 'R'):
                self.text_transforms.append(partial(F_t.text_reorder, base_dir=reorder))
            else:
                self.text_transforms.append(F_t.text_reorder)

        self.im_transforms = v2.Compose([v2.Lambda(partial(optional_resize, max_size=height)),
                                         v2.ToImage(),
                                         v2.ToDtype(torch.float32, scale=True),
                                         v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        # codec is stateless so we can just initiate it here
        self.codec = ByT5Codec()

        self.pad_id = self.codec.pad
        self.sos_id = self.codec.sos
        self.eos_id = self.codec.eos

        self.num_classes = self.codec.max_label + 1

        # pyarrow structs
        self.line_struct = pa.struct([('text', pa.list_(pa.int32())), ('curve', pa.list_(pa.float32()))])
        self.page_struct = pa.struct([('im', pa.string()), ('lines', pa.list_(self.line_struct))])

    def prepare_data(self):
        """
        Compiles the dataset(s) into pyarrow arrays and saves them to `self.tmpdir`
        """
        print('Initializing dataset.')
        self._parse_data('train.arrow', self.hparams.training_data)
        self._parse_data('val.arrow', self.hparams.evaluation_data)

    def _parse_data(self, name: Literal['train.arrow', 'val.arrow'], files: Sequence['PathLike']):
        num_lines = 0

        schema = pa.schema([('pages', self.page_struct)])

        ds = []
        for page in [XMLPage(file).to_container() for file in files]:
            im_size = Image.open(page.imagename).size
            page_data = []
            for line in page.lines:
                text = line.text
                for func in self.text_transforms:
                    text = func(text)
                if not text:
                    logger.info(f'Text line "{line.text}" is empty after transformations')
                    continue
                if not line.baseline:
                    logger.info('No baseline given for line')
                    continue
                page_data.append(pa.scalar({'text': pa.scalar(self.codec.encode(text).numpy()),
                                            'curve': self._to_curve(line.baseline, im_size)},
                                           self.line_struct))
                num_lines += 1
            else:
                logger.info(f'Empty page {page.imagename}. Skipping.')
            if len(page_data) > 1:
                ds.append(pa.scalar({'im': str(page.imagename), 'lines': page_data}, self.page_struct))

        metadata = {'num_lines': num_lines.to_bytes(4, 'little')}
        schema = schema.with_metadata(metadata)

        with pa.OSFile(self.tmpdir.name + '/' + name, 'wb') as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                ar = pa.array(ds, self.page_struct)
                writer.write(pa.RecordBatch.from_arrays([ar], schema=schema))

    @staticmethod
    def _to_curve(baseline, im_size, min_points: int = 8) -> torch.Tensor:
        """
        Converts poly(base)lines to Bezier curves and normalizes them.
        """
        baseline = np.array(baseline)
        if len(baseline) < min_points:
            ls = LineString(baseline)
            baseline = np.stack([np.array(ls.interpolate(x, normalized=True).coords)[0] for x in np.linspace(0, 1, 8)])
        # control points normalized to patch extents
        curve = np.concatenate(([baseline[0]], bezier_fit(baseline), [baseline[-1]]))/im_size
        curve = curve.flatten()
        return pa.scalar(curve, type=pa.list_(pa.float32()))

    def setup(self, stage: str):
        """
        Actually builds the datasets.
        """
        self.train_set = BinnedBaselineDataset(self.tmpdir.name + '/train.arrow',
                                               im_transforms=self.im_transforms,
                                               augmentation=self.hparams.augmentation,
                                               max_batch_size=self.hparams.batch_size)
        self.val_set = BinnedBaselineDataset(self.tmpdir.name + '/val.arrow',
                                             im_transforms=self.im_transforms,
                                             augmentation=self.hparams.augmentation,
                                             max_batch_size=self.hparams.batch_size)

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=1,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          shuffle=False,
                          collate_fn=collate_null)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          shuffle=False,
                          batch_size=1,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          collate_fn=collate_null,
                          worker_init_fn=_validation_worker_init_fn)


class BinnedBaselineDataset(Dataset):
    """
    Dataset for training a line recognition model from baseline data.

    Images are binned, so the batch_size parameter of the data loader is an
    upper limit of the number of samples returned.

    Args:
        normalization: Unicode normalization for gt
        whitespace_normalization: Normalizes unicode whitespace and strips
                                  whitespace.
        skip_empty_lines: Whether to return samples without text.
        reorder: Whether to rearrange code points in "display"/LTR order.
                 Set to L|R to change the default text direction.
        im_transforms: Function taking an PIL.Image and returning a tensor
                       suitable for forward passes.
        augmentation: Enables augmentation.
        max_batch_size: Maximum size of a batch. All samples from a batch will
                        come from a single page.
    """
    def __init__(self,
                 path: 'PathLike',
                 im_transforms: Callable[[Any], torch.Tensor] = transforms.Compose([]),
                 augmentation: bool = False,
                 max_batch_size: int = 32) -> None:
        self.path = path
        self.transforms = im_transforms
        self.aug = None
        self.max_batch_size = max_batch_size

        with pa.memory_map(path, 'rb') as source:
            self.ds_table = pa.ipc.open_file(source).read_all()
            raw_metadata = self.ds_table.schema.metadata
            if not raw_metadata or b'num_lines' not in raw_metadata:
                raise ValueError(f'{path} does not contain a valid metadata record.')
        self._len = int.from_bytes(raw_metadata[b'num_lines'], 'little')

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # just sample from a random page
        rng = np.random.default_rng()
        idx = rng.integers(0, len(self.ds_table))

        item = self.ds_table.column('pages')[idx].as_py()
        logger.debug(f'Attempting to load {item["im"]}')
        im, page_data = item['im'], item['lines']
        if not isinstance(im, Image.Image):
            im = Image.open(im).convert('RGB')
        im = self.transforms(im)

        if self.aug:
            im = im.permute((1, 2, 0)).numpy()
            o = self.aug(image=im)
            im = torch.tensor(o['image'].transpose(2, 0, 1))

        # sample up to max_batch_size lines and targets
        num_samples = min(self.max_batch_size, len(page_data))
        lines = [page_data[x] for x in rng.choice(len(page_data), num_samples, replace=False, shuffle=False)]
        lines = [(torch.tensor(x['text'], dtype=torch.int32), torch.tensor(x['curve']).view(4, 2)) for x in lines]
        return collate_sequences(im.unsqueeze(0), lines)

    def __len__(self) -> int:
        return self._len // self.max_batch_size


# magic lsq cubic bezier fit function from the internet.
def Mtk(n, t, k):
    return t**k * (1-t)**(n-k) * comb(n, k)


def BezierCoeff(ts):
    return [[Mtk(3, t, k) for k in range(4)] for t in ts]


def bezier_fit(bl):
    x = bl[:, 0]
    y = bl[:, 1]
    dy = y[1:] - y[:-1]
    dx = x[1:] - x[:-1]
    dt = (dx ** 2 + dy ** 2)**0.5
    t = dt/dt.sum()
    t = np.hstack(([0], t))
    t = t.cumsum()

    Pseudoinverse = np.linalg.pinv(BezierCoeff(t))  # (9,4) -> (4,9)

    control_points = Pseudoinverse.dot(bl)  # (4,9)*(9,2) -> (4,2)
    medi_ctp = control_points[1:-1, :]
    return medi_ctp
