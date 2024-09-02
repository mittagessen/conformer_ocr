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
import multiprocessing as mp

from typing import (TYPE_CHECKING, Any, Callable, List, Literal, Optional,
                    Tuple, Union, Sequence)

from torch.utils.data import DataLoader, Subset

from conformer_ocr.codec import TransformerCodec

from collections import Counter
from functools import partial
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image

from ctypes import c_char

from scipy.special import comb
from shapely.geometry import LineString

from kraken.containers import Segmentation
from kraken.lib import functional_im_transforms as F_t
from kraken.lib.xml import XMLPage
from kraken.lib.util import is_bitonal
from kraken.lib.dataset import ImageInputTransforms
from kraken.lib.dataset.recognition import DefaultAugmenter

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


class TextLineDataModule(L.LightningDataModule):
    def __init__(self,
                 training_data: Sequence[Union[str, 'PathLike']],
                 evaluation_data: Optional[Sequence[Union[str, 'PathLike']]] = None,
                 height: int = 0,
                 pad: int = 0,
                 augmentation: bool = False,
                 batch_size: int = 16,
                 num_workers: int = 8,
                 partition: Optional[float] = 0.95,
                 codec: Optional[TransformerCodec] = None,
                 format_type: Literal['alto', 'page', 'xml'] = 'xml',
                 reorder: Union[bool, str] = True,
                 normalize_whitespace: bool = True,
                 normalization: Optional[Literal['NFD', 'NFC', 'NFKD', 'NFKC']] = None):
        super().__init__()

        self.save_hyperparameters()

        if format_type in ['xml', 'page', 'alto']:
            DatasetClass = BinnedBaselineDataset
            logger.info(f'Parsing {len(training_data)} XML files for training data')
            training_data = [{'page': XMLPage(file, format_type).to_container()} for file in training_data]
            if evaluation_data:
                logger.info(f'Parsing {len(evaluation_data)} XML files for validation data')
                evaluation_data = [{'page': XMLPage(file, format_type).to_container()} for file in evaluation_data]
        else:
            raise ValueError(f'format_type {format_type} not in [alto, page, xml, binary].')

        self.transforms = ImageInputTransforms(1, height, 0, 3, (pad, 0), valid_norm=False)

        if evaluation_data:
            train_set = self._build_dataset(DatasetClass, training_data)
            self.train_set = Subset(train_set, range(len(train_set)))
            val_set = self._build_dataset(DatasetClass, evaluation_data)
            self.val_set = Subset(val_set, range(len(val_set)))
        else:
            raise ValueError('Cannot use random splits with binned dataset class')

        if len(self.train_set) == 0:
            raise ValueError('No valid training data provided. Please add some.')

        if len(self.val_set) == 0:
            raise ValueError('No valid validation data provided. Please add some.')

        self.train_set.dataset.encode(codec)
        self.codec = self.train_set.dataset.codec
        self.pad_id = self.codec.pad
        self.sos_id = self.codec.sos
        self.eos_id = self.codec.eos

        val_diff = set(self.val_set.dataset.alphabet).difference(
            set(self.train_set.dataset.codec.c2l.keys())
        )
        logger.info(f'Adding {len(val_diff)} dummy labels to validation set codec.')

        val_codec = self.codec.add_labels(val_diff)
        self.val_set.dataset.encode(val_codec)
        self.val_codec = val_codec

        self.num_classes = self.train_set.dataset.codec.max_label + 1

        self.save_hyperparameters()

    def _build_dataset(self, DatasetClass, training_data, **kwargs):

        dataset = DatasetClass(normalization=self.hparams.normalization,
                               whitespace_normalization=self.hparams.normalize_whitespace,
                               reorder=self.hparams.reorder,
                               im_transforms=self.transforms,
                               augmentation=self.hparams.augmentation,
                               max_batch_size=self.hparams.batch_size,
                               **kwargs)

        for sample in training_data:
            try:
                dataset.add(**sample)
            except Exception as e:
                logger.warning(str(e))
        if self.hparams.format_type == 'binary' and self.hparams.normalization:
            logger.debug('Rebuilding dataset using unicode normalization')
            dataset.rebuild_alphabet()

        return dataset

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

    def state_dict(self):
        # track whatever you want here
        return {"codec": self.codec.c2l}

    def load_state_dict(self, state_dict):
        # restore the state based on what you tracked in (def state_dict)
        self.codec = TransformerCodec(state_dict['codec'])


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
                 normalization: Optional[str] = None,
                 whitespace_normalization: bool = True,
                 skip_empty_lines: bool = True,
                 reorder: Union[bool, Literal['L', 'R']] = True,
                 im_transforms: Callable[[Any], torch.Tensor] = transforms.Compose([]),
                 augmentation: bool = False,
                 max_batch_size: int = 32) -> None:
        self.training_set: List = []
        self.alphabet: Counter = Counter()
        self.text_transforms: List[Callable[[str], str]] = []
        self.transforms = im_transforms
        self.aug = None
        self.skip_empty_lines = skip_empty_lines
        self.failed_samples = set()
        self.max_batch_size = max_batch_size

        self.seg_type = 'baselines'
        # built text transformations
        if normalization:
            self.text_transforms.append(partial(F_t.text_normalize, normalization=normalization))
        if whitespace_normalization:
            self.text_transforms.append(F_t.text_whitespace_normalize)
        if reorder:
            if reorder in ('L', 'R'):
                self.text_transforms.append(partial(F_t.text_reorder, base_dir=reorder))
            else:
                self.text_transforms.append(F_t.text_reorder)
        if augmentation:
            self.aug = DefaultAugmenter()

        self._im_mode = mp.Value(c_char, b'1')
        self._len = 0

    def add(self, page: Segmentation):
        """
        Adds all lines on a page to the dataset.

        Invalid lines will be skipped and a warning will be printed.

        Args:

            page: Segmentation container object for a page.
        """
        if page.type != 'baselines':
            raise ValueError(f'Invalid segmentation of type {page.type} (expected "baselines")')
        im_size = Image.open(page.imagename).size
        page_data = []
        for line in page.lines:
            text = line.text
            for func in self.text_transforms:
                text = func(text)
            if not text and self.skip_empty_lines:
                logger.warning(f'Text line "{line.text}" is empty after transformations')
                continue
            if not line.baseline:
                logger.warning('No baseline given for line')
                continue
            # to normalized Bézier curve
            curve = self._to_curve(line.baseline, im_size)
            page_data.append((text, curve))
            self.alphabet.update(text)
        if len(page_data):
            self.training_set.append((page.imagename, page_data))
            self._len += len(page_data)
        else:
            logger.warning(f'Empty page {page.imagename}. Skipping.')

    def encode(self, codec: Optional[TransformerCodec] = None) -> None:
        """
        Adds a codec to the dataset and encodes all text lines.

        Has to be run before sampling from the dataset.
        """
        if codec:
            self.codec = codec
        else:
            self.codec = TransformerCodec(''.join(self.alphabet.keys()))
        tmp_pages = []
        for im, page in self.training_set:
            tmp_page = []
            for text, curve in page:
                tmp_page.append((self.codec.encode(text), curve))
            tmp_pages.append((im, tmp_page))
        self.training_set = tmp_pages

    def no_encode(self) -> None:
        """
        Creates an unencoded dataset.
        """
        pass

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # just sample from a random page
        rng = np.random.default_rng()
        idx = rng.integers(0, len(self.training_set))

        item = self.training_set[idx]
        logger.debug(f'Attempting to load {item[0]}')
        im, page_data = item
        if not isinstance(im, Image.Image):
            im = Image.open(im)
        im = self.transforms(im)
        if im.shape[0] == 3:
            im_mode = b'R'
        elif im.shape[0] == 1:
            im_mode = b'L'
        if is_bitonal(im):
            im_mode = b'1'

        with self._im_mode.get_lock():
            if im_mode > self._im_mode.value:
                logger.info(f'Upgrading "im_mode" from {self._im_mode.value} to {im_mode}')
                self._im_mode.value = im_mode
        if self.aug:
            im = im.permute((1, 2, 0)).numpy()
            o = self.aug(image=im)
            im = torch.tensor(o['image'].transpose(2, 0, 1))

        # sample up to max_batch_size lines and targets
        num_samples = min(self.max_batch_size, len(page_data))
        lines = [page_data[x] for x in rng.choice(len(page_data), num_samples, replace=False, shuffle=False)]
        return collate_sequences(im.unsqueeze(0), lines)

    def __len__(self) -> int:
        return self._len // self.max_batch_size

    @staticmethod
    def _to_curve(baseline, im_size, min_points: int = 8) -> torch.Tensor:
        baseline = np.array(baseline)
        if len(baseline) < min_points:
            ls = LineString(baseline)
            baseline = np.stack([np.array(ls.interpolate(x, normalized=True).coords)[0] for x in np.linspace(0, 1, 8)])
        # control points normalized to patch extents
        curve = np.concatenate(([baseline[0]], bezier_fit(baseline), [baseline[-1]]))/im_size
        return torch.from_numpy(curve)

    @property
    def im_mode(self):
        return {b'1': '1',
                b'L': 'L',
                b'R': 'RGB'}[self._im_mode.value]


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
