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
import traceback
import dataclasses
import lightning.pytorch as L
import multiprocessing as mp

from typing import (TYPE_CHECKING, Any, Callable, List, Literal, Optional,
                    Tuple, Union, Sequence)

from torch.utils.data import DataLoader, Subset, random_split

from conformer_ocr.codec import TransformerCodec

from collections import Counter
from functools import partial
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image

from ctypes import c_char

from scipy.special import comb
from shapely.geometry import LineString, Polygon

from shapely.ops import clip_by_rect

from kraken.containers import Segmentation, BaselineLine
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


def collate_sequences(batch):
    """
    Sorts and pads sequences.
    """
    sorted_batch = sorted(batch, key=lambda x: x['image'].shape[2], reverse=True)
    seqs = [x['image'] for x in sorted_batch]
    seq_lens = torch.LongTensor([seq.shape[2] for seq in seqs])
    max_len = seqs[0].shape[2]
    seqs = torch.stack([F.pad(seq, pad=(0, max_len-seq.shape[2])) for seq in seqs])
    if isinstance(sorted_batch[0]['target'], str):
        labels = [x['target'] for x in sorted_batch]
    else:
        max_label_len = max(len(x['target']) for x in sorted_batch)
        labels = torch.stack([F.pad(x['target'], pad=(0, max_label_len-len(x['target']))) for x in sorted_batch]).long()
    label_lens = torch.LongTensor([len(x['target']) for x in sorted_batch])
    return {'image': seqs, 'target': labels, 'seq_lens': seq_lens, 'target_lens': label_lens}


class TextLineDataModule(L.LightningDataModule):
    def __init__(self,
                 training_data: Sequence[Union[str, 'PathLike']],
                 evaluation_data: Optional[Sequence[Union[str, 'PathLike']]] = None,
                 height: int = 96,
                 pad: int = 16,
                 augmentation: bool = False,
                 batch_size: int = 16,
                 num_workers: int = 8,
                 partition: Optional[float] = 0.95,
                 codec: Optional[TransformerCodec] = None,
                 format_type: Literal['alto', 'page', 'xml'] = 'xml',
                 binary_dataset_split: bool = False,
                 reorder: Union[bool, str] = True,
                 normalize_whitespace: bool = True,
                 normalization: Optional[Literal['NFD', 'NFC', 'NFKD', 'NFKC']] = None):
        super().__init__()

        self.save_hyperparameters()

        if format_type in ['xml', 'page', 'alto']:
            DatasetClass = PolygonGTDataset
            logger.info(f'Parsing {len(training_data)} XML files for training data')
            training_data = [{'page': XMLPage(file, format_type).to_container()} for file in training_data]
            if evaluation_data:
                logger.info(f'Parsing {len(evaluation_data)} XML files for validation data')
                evaluation_data = [{'page': XMLPage(file, format_type).to_container()} for file in evaluation_data]
            if binary_dataset_split:
                logger.warning('Internal binary dataset splits are enabled but using non-binary dataset files. Will be ignored.')
                binary_dataset_split = False
        else:
            raise ValueError(f'format_type {format_type} not in [alto, page, xml, binary].')

        self.transforms = ImageInputTransforms(1, height, 0, 3, (pad, 0), valid_norm=False)

        if evaluation_data:
            train_set = self._build_dataset(DatasetClass, training_data)
            self.train_set = Subset(train_set, range(len(train_set)))
            val_set = self._build_dataset(DatasetClass, evaluation_data)
            self.val_set = Subset(val_set, range(len(val_set)))
        elif binary_dataset_split:
            train_set = self._build_dataset(DatasetClass, training_data, split_filter='train')
            self.train_set = Subset(train_set, range(len(train_set)))
            val_set = self._build_dataset(DatasetClass, training_data, split_filter='validation')
            self.val_set = Subset(val_set, range(len(val_set)))
            logger.info(f'Found {len(self.train_set)} (train) / {len(self.val_set)} (val) samples in pre-encoded dataset')
        else:
            train_set = self._build_dataset(DatasetClass, training_data)
            train_len = int(len(train_set)*partition)
            val_len = len(train_set) - train_len
            logger.info(f'No explicit validation data provided. Splitting off '
                        f'{val_len} (of {len(train_set)}) samples to validation '
                        'set. (Will disable alphabet mismatch detection.)')
            self.train_set, self.val_set = random_split(train_set, (train_len, val_len))

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
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          shuffle=True,
                          collate_fn=collate_sequences)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          shuffle=False,
                          batch_size=1,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          collate_fn=collate_sequences,
                          worker_init_fn=_validation_worker_init_fn)

    def state_dict(self):
        # track whatever you want here
        return {"codec": self.codec.c2l}

    def load_state_dict(self, state_dict):
        # restore the state based on what you tracked in (def state_dict)
        self.codec = TransformerCodec(state_dict['codec'])


class PolygonGTDataset(Dataset):
    """
    Dataset for training a line recognition model from polygonal/baseline data.

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
    """
    def __init__(self,
                 normalization: Optional[str] = None,
                 whitespace_normalization: bool = True,
                 skip_empty_lines: bool = True,
                 reorder: Union[bool, Literal['L', 'R']] = True,
                 im_transforms: Callable[[Any], torch.Tensor] = transforms.Compose([]),
                 augmentation: bool = False) -> None:
        self._images: Union[List[Image.Image], List[torch.Tensor]] = []
        self._gt: List[str] = []
        self.alphabet: Counter = Counter()
        self.text_transforms: List[Callable[[str], str]] = []
        self.transforms = im_transforms
        self.aug = None
        self.skip_empty_lines = skip_empty_lines
        self.failed_samples = set()

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

    def add(self,
            line: Optional[BaselineLine] = None,
            page: Optional[Segmentation] = None):
        """
        Adds an individual line or all lines on a page to the dataset.

        Args:
            line: BaselineLine container object of a line.
            page: Segmentation container object for a page.
        """
        if line:
            self.add_line(line)
        if page:
            self.add_page(page)
        if not (line or page):
            raise ValueError('Neither line nor page data provided in dataset builder')

    def add_page(self, page: Segmentation):
        """
        Adds all lines on a page to the dataset.

        Invalid lines will be skipped and a warning will be printed.

        Args:

            page: Segmentation container object for a page.
        """
        if page.type != 'baselines':
            raise ValueError(f'Invalid segmentation of type {page.type} (expected "baselines")')
        for line in page.lines:
            try:
                self.add_line(dataclasses.replace(line, imagename=page.imagename))
            except ValueError as e:
                logger.warning(e)

    def add_line(self, line: BaselineLine):
        """
        Adds a line to the dataset.

        Args:
            line: BaselineLine container object for a line.

        Raises:
            ValueError if the transcription of the line is empty after
            transformation or either baseline or bounding polygon are missing.
        """
        if line.type != 'baselines':
            raise ValueError(f'Invalid line of type {line.type} (expected "baselines")')

        text = line.text
        for func in self.text_transforms:
            text = func(text)
        if not text and self.skip_empty_lines:
            raise ValueError(f'Text line "{line.text}" is empty after transformations')
        if not line.baseline:
            raise ValueError('No baseline given for line')
        if not line.boundary:
            raise ValueError('No boundary given for line')

        self._images.append((line.imagename, line.baseline, line.boundary))
        self._gt.append(text)
        self.alphabet.update(text)

    def encode(self, codec: Optional[TransformerCodec] = None) -> None:
        """
        Adds a codec to the dataset and encodes all text lines.

        Has to be run before sampling from the dataset.
        """
        if codec:
            self.codec = codec
        else:
            self.codec = TransformerCodec(''.join(self.alphabet.keys()))
        self.training_set: List[Tuple[Union[Image.Image, torch.Tensor], torch.Tensor]] = []
        for im, gt in zip(self._images, self._gt):
            self.training_set.append((im, self.codec.encode(gt)))

    def no_encode(self) -> None:
        """
        Creates an unencoded dataset.
        """
        self.training_set: List[Tuple[Union[Image.Image, torch.Tensor], str]] = []
        for im, gt in zip(self._images, self._gt):
            self.training_set.append((im, gt))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.training_set[index]
        try:
            logger.debug(f'Attempting to load {item[0]}')
            im = item[0][0]
            if not isinstance(im, Image.Image):
                im = Image.open(im)
            im, curve = convert_line(im, baseline=item[0][1], boundary=item[0][2])
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
            return {'image': im, 'curve': torch.from_numpy(curve), 'target': item[1]}
        except Exception:
            self.failed_samples.add(index)
            idx = np.random.randint(0, len(self.training_set))
            logger.debug(traceback.format_exc())
            logger.info(f'Failed. Replacing with sample {idx}')
            return self[idx]

    def __len__(self) -> int:
        return len(self._images)

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

def convert_line(image: Image.Image, baseline, boundary, min_points: int = 8):
    """
    Converts a baseline to a Bezier representation and crops the input image
    roughly around the line.
    """
    baseline = np.array(baseline)
    if len(baseline) < min_points:
        ls = LineString(baseline)
        baseline = np.stack([np.array(ls.interpolate(x, normalized=True).coords)[0] for x in np.linspace(0, 1, 8)])
    # get a rough environment from the bounding polygon
    pol = Polygon(boundary).envelope
    buff = min(np.abs(pol.bounds[0] - pol.bounds[2]), np.abs(pol.bounds[1] - pol.bounds[3]))
    patch = clip_by_rect(pol.buffer(buff).envelope, 0, 0, image.width, image.height).bounds
    # control points normalized to patch extents
    curve = ((np.concatenate(([baseline[0]], bezier_fit(baseline),
                              [baseline[-1]])) - (patch[0],
                                                  patch[1]))/(patch[2]-patch[0],
                                                              patch[3]-patch[1])).flatten().tolist()
    line_im = image.crop(patch)
    return line_im, curve
