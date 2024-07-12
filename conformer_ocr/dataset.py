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
from typing import (TYPE_CHECKING, Literal, Optional, Sequence, Union)

import io
import json
import torch
import torch.nn.functional as F
import numpy as np
import pyarrow as pa
import traceback
import dataclasses
import lightning.pytorch as L

from collections import Counter
from functools import partial
from typing import (TYPE_CHECKING, Any, Callable, List, Literal, Optional,
                    Sequence, Tuple, Union)

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split

from kraken.lib import functional_im_transforms as F_t
from kraken.lib.xml import XMLPage
from kraken.lib.codec import PytorchCodec
from kraken.lib.dataset import PolygonGTDataset, ImageInputTransforms
from kraken.lib.exceptions import KrakenInputException

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
    stc = True
    for x in sorted_batch:
        if not x['semantic_token']:
            stc = False
            break
    if stc:
        semantic_tokens = torch.stack([x['semantic_token'] for x in sorted_batch])
    seq_lens = torch.LongTensor([seq.shape[2] for seq in seqs])
    max_len = seqs[0].shape[2]
    seqs = torch.stack([F.pad(seq, pad=(0, max_len-seq.shape[2])) for seq in seqs])
    if isinstance(sorted_batch[0]['target'], str):
        labels = [x['target'] for x in sorted_batch]
    else:
        labels = torch.cat([x['target'] for x in sorted_batch]).long()
    label_lens = torch.LongTensor([len(x['target']) for x in sorted_batch])
    ret = {'image': seqs,
           'target': labels,
           'seq_lens': seq_lens,
           'target_lens': label_lens}
    if stc:
        ret['semantic_token'] = semantic_tokens
    return ret


class DefaultAugmenter():
    def __init__(self):
        import cv2
        cv2.setNumThreads(0)
        from albumentations import (Blur, Compose, ElasticTransform,
                                    MedianBlur, MotionBlur, OneOf,
                                    OpticalDistortion, PixelDropout,
                                    ShiftScaleRotate, ToFloat)

        self._transforms = Compose([
                                    ToFloat(),
                                    PixelDropout(p=0.2),
                                    OneOf([
                                        MotionBlur(p=0.2),
                                        MedianBlur(blur_limit=3, p=0.1),
                                        Blur(blur_limit=3, p=0.1),
                                    ], p=0.2),
                                    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=1, p=0.2),
                                    OneOf([
                                        OpticalDistortion(p=0.3),
                                        ElasticTransform(alpha=64, sigma=25, alpha_affine=0.25, p=0.1),
                                    ], p=0.2),
                                   ], p=0.5)

    def __call__(self, image):
        return self._transforms(image=image)


class ArrowIPCRecognitionDataset(Dataset):
    """
    Dataset for training a recognition model from a precompiled dataset in
    Arrow IPC format.
    """
    def __init__(self,
                 normalization: Optional[str] = None,
                 whitespace_normalization: bool = True,
                 skip_empty_lines: bool = True,
                 reorder: Union[bool, Literal['L', 'R']] = True,
                 im_transforms: Callable[[Any], torch.Tensor] = transforms.Compose([]),
                 augmentation: bool = False,
                 split_filter: Optional[str] = None,
                 semantic_token_fields: Optional[List[str]] = None) -> None:
        """
        Creates a dataset for a polygonal (baseline) transcription model.

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
            split_filter: Enables filtering of the dataset according to mask
                          values in the set split. If set to `None` all rows
                          are sampled, if set to `train`, `validation`, or
                          `test` only rows with the appropriate flag set in the
                          file will be considered.
            semantic_token_fields: List of semantic tokens to stack onto the
                                   image tensor in the height dimension.
        """
        self.alphabet: Counter = Counter()
        self.text_transforms: List[Callable[[str], str]] = []
        self.failed_samples = set()
        self.transforms = im_transforms
        self.aug = None
        self._split_filter = split_filter
        self._num_lines = 0
        self.arrow_table = None
        self.codec = None
        self.skip_empty_lines = skip_empty_lines
        self.legacy_polygons_status = None
        self.semantic_token_fields = semantic_token_fields

        self.seg_type = None
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

        self.im_mode = self.transforms.mode

    def add(self, file: Union[str, 'PathLike']) -> None:
        """
        Adds an Arrow IPC file to the dataset.

        Args:
            file: Location of the precompiled dataset file.
        """
        # extract metadata and update alphabet
        with pa.memory_map(file, 'rb') as source:
            ds_table = pa.ipc.open_file(source).read_all()
            raw_metadata = ds_table.schema.metadata
            if not raw_metadata or b'lines' not in raw_metadata:
                raise ValueError(f'{file} does not contain a valid metadata record.')
            metadata = json.loads(raw_metadata[b'lines'])
        if metadata['type'] == 'kraken_recognition_baseline':
            if not self.seg_type:
                self.seg_type = 'baselines'
            if self.seg_type != 'baselines':
                raise ValueError(f'File {file} has incompatible type {metadata["type"]} for dataset with type {self.seg_type}.')
        elif metadata['type'] == 'kraken_recognition_bbox':
            if not self.seg_type:
                self.seg_type = 'bbox'
            if self.seg_type != 'bbox':
                raise ValueError(f'File {file} has incompatible type {metadata["type"]} for dataset with type {self.seg_type}.')
        else:
            raise ValueError(f'Unknown type {metadata["type"]} of dataset.')
        if self._split_filter and metadata['counts'][self._split_filter] == 0:
            logger.warning(f'No explicit split for "{self._split_filter}" in dataset {file} (with splits {metadata["counts"].items()}).')
            return
        if metadata['im_mode'] > self.im_mode and self.transforms.mode >= metadata['im_mode']:
            logger.info(f'Upgrading "im_mode" from {self.im_mode} to {metadata["im_mode"]}.')
            self.im_mode = metadata['im_mode']
        # centerline normalize raw bbox dataset
        if self.seg_type == 'bbox' and metadata['image_type'] == 'raw':
            self.transforms.valid_norm = True

        legacy_polygons = metadata.get('legacy_polygons', True)
        if self.legacy_polygons_status is None:
            self.legacy_polygons_status = legacy_polygons
        elif self.legacy_polygons_status != legacy_polygons:
            self.legacy_polygons_status = "mixed"

        self.alphabet.update(metadata['alphabet'])
        num_lines = metadata['counts'][self._split_filter] if self._split_filter else metadata['counts']['all']
        # check that all semantic token fields exist as table columns
        if self.semantic_token_fields:
            for stf in self.semantic_token_fields:
                if stf not in ds_table.column_names:
                    logger.warning(f'Requested semantic token {stf} not available as a column name in {ds_table.column_names}.')
        if self._split_filter:
            ds_table = ds_table.filter(ds_table.column(self._split_filter))
        if self.skip_empty_lines:
            logger.debug('Getting indices of empty lines after text transformation.')
            self.skip_empty_lines = False
            mask = np.ones(len(ds_table), dtype=bool)
            for index in range(len(ds_table)):
                try:
                    self._apply_text_transform(ds_table.column('lines')[index].as_py(),)
                except KrakenInputException:
                    mask[index] = False
                    continue
            num_lines = np.count_nonzero(mask)
            logger.debug(f'Filtering out {np.count_nonzero(~mask)} empty lines')
            if np.any(~mask):
                ds_table = ds_table.filter(pa.array(mask))
            self.skip_empty_lines = True
        if not self.arrow_table:
            self.arrow_table = ds_table
        else:
            self.arrow_table = pa.concat_tables([self.arrow_table, ds_table])
        self._num_lines += num_lines

    def rebuild_alphabet(self):
        """
        Recomputes the alphabet depending on the given text transformation.
        """
        self.alphabet = Counter()
        for index in range(len(self)):
            try:
                text = self._apply_text_transform(self.arrow_table.column('lines')[index].as_py(),)
                self.alphabet.update(text)
            except KrakenInputException:
                continue

    def _apply_text_transform(self, sample) -> str:
        """
        Applies text transform to a sample.
        """
        text = sample['text']
        for func in self.text_transforms:
            text = func(text)
        if not text:
            logger.debug(f'Text line "{sample["text"]}" is empty after transformations')
            if not self.skip_empty_lines:
                raise KrakenInputException('empty text line')
        return text

    def encode(self, codec: Optional[PytorchCodec] = None) -> None:
        """
        Adds a codec to the dataset.
        """
        if codec:
            self.codec = codec
            logger.info(f'Trying to encode dataset with codec {codec}')
            for index in range(self._num_lines):
                try:
                    text = self._apply_text_transform(
                        self.arrow_table.column('lines')[index].as_py(),
                    )
                    self.codec.encode(text)
                except KrakenEncodeException as e:
                    raise e
                except KrakenInputException:
                    pass
        else:
            self.codec = PytorchCodec(''.join(self.alphabet.keys()))

    def no_encode(self) -> None:
        """
        Creates an unencoded dataset.
        """
        pass

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        semantic_token = None
        try:
            sample = self.arrow_table.column('lines')[index].as_py()
            logger.debug(f'Loading sample {index}')
            im = Image.open(io.BytesIO(sample['im']))
            im = self.transforms(im)
            if self.aug:
                im = im.permute((1, 2, 0)).numpy()
                o = self.aug(image=im)
                im = torch.tensor(o['image'].transpose(2, 0, 1))
            if self.semantic_token_fields:
                semantic_token = [self.arrow_table.column(x)[index].as_py() if x in self.arrow_table else False for x in self.semantic_token_fields]
                semantic_token = torch.tensor(semantic_token, dtype=im.dtype)
            text = self._apply_text_transform(sample)
        except Exception:
            self.failed_samples.add(index)
            idx = np.random.randint(0, len(self))
            logger.debug(traceback.format_exc())
            logger.info(f'Failed. Replacing with sample {idx}')
            return self[idx]

        return {'image': im,
                'semantic_token': semantic_token,
                'target': self.codec.encode(text) if self.codec is not None else text}

    def __len__(self) -> int:
        return self._num_lines


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
                 codec: Optional[PytorchCodec] = None,
                 format_type: Literal['alto', 'page', 'xml', 'binary'] = 'xml',
                 binary_dataset_split: bool = False,
                 reorder: Union[bool, str] = True,
                 normalize_whitespace: bool = True,
                 normalization: Optional[Literal['NFD', 'NFC', 'NFKD', 'NFKC']] = None,
                 semantic_token_fields: Optional[List[str]] = None):
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
        elif format_type == 'binary':
            DatasetClass = ArrowIPCRecognitionDataset
            logger.info(f'Got {len(training_data)} binary dataset files for training data')
            training_data = [{'file': file} for file in training_data]
            if evaluation_data:
                logger.info(f'Got {len(evaluation_data)} binary dataset files for validation data')
                evaluation_data = [{'file': file} for file in evaluation_data]
        else:
            raise ValueError(f'format_type {format_type} not in [alto, page, xml, binary].')

        self.transforms = ImageInputTransforms(1, height, 0, 1, (pad, 0), valid_norm=False)

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
                               semantic_token_fields=self.hparams.semantic_token_fields,
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
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          collate_fn=collate_sequences,
                          worker_init_fn=_validation_worker_init_fn)

    def state_dict(self):
        # track whatever you want here
        return {"codec": self.codec.c2l}

    def load_state_dict(self, state_dict):
        # restore the state based on what you tracked in (def state_dict)
        self.codec = PytorchCodec(state_dict['codec'])

