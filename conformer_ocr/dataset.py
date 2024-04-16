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
from typing import (TYPE_CHECKING, Any, Callable, List, Literal, Optional, Sequence,
                    Tuple, Union)

import torch
import pytorch_lightning as pl

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset, random_split

from kraken.containers import BaselineLine, Segmentation
from kraken.lib.codec import PytorchCodec
from kraken.lib.dataset import (ArrowIPCRecognitionDataset, PolygonGTDataset,
                                ImageInputTransforms)

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
    from pytorch_lightning import seed_everything
    level = logging.getLogger("lightning_fabric.utilities.seed").level
    logging.getLogger("lightning_fabric.utilities.seed").setLevel(logging.WARN)
    seed_everything(42)
    logging.getLogger("lightning_fabric.utilities.seed").setLevel(level)


def collate_sequences(batch):
    """
    Sorts and pads sequences.
    """
    seqs = pad_sequence([x['image'].transpose(0, 2) for x in batch], batch_first=True).transpose(1, 3)
    labels = pad_sequence([x['target'] for x in batch], batch_first=True)
    seq_lens = torch.LongTensor([seq.shape[2] for seq in seqs])
    label_lens = torch.LongTensor([len(x['target']) for x in batch])
    return {'image': seqs, 'target': labels, 'seq_lens': seq_lens, 'target_lens': label_lens}


class TextLineDataModule(pl.LightningDataModule):
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
                 binary_dataset_split: bool = False,
                 reorder: Union[bool, str] = True,
                 normalize_whitespace: bool = True,
                 normalization: Optional[Literal['NFD', 'NFC', 'NFKD', 'NFKC']] = None):
        super().__init__()

        self.save_hyperparameters()

        logger.info(f'Got {len(training_data)} binary dataset files for training data')
        training_data = [{'file': file} for file in training_data]
        if evaluation_data:
            logger.info(f'Got {len(evaluation_data)} binary dataset files for validation data')
            evaluation_data = [{'file': file} for file in evaluation_data]

        self.transforms = ImageInputTransforms(1, height, 0, 1, (pad, 0), valid_norm=False)

        if evaluation_data:
            train_set = self._build_dataset(ArrowIPCRecognitionDataset, training_data)
            self.train_set = Subset(train_set, range(len(train_set)))
            val_set = self._build_dataset(ArrowIPCRecognitionDataset, evaluation_data)
            self.val_set = Subset(val_set, range(len(val_set)))
        elif binary_dataset_split:
            train_set = self._build_dataset(ArrowIPCRecognitionDataset, training_data, split_filter='train')
            self.train_set = Subset(train_set, range(len(train_set)))
            val_set = self._build_dataset(ArrowIPCRecognitionDataset, training_data, split_filter='validation')
            self.val_set = Subset(val_set, range(len(val_set)))
            logger.info(f'Found {len(self.train_set)} (train) / {len(self.val_set)} (val) samples in pre-encoded dataset')
        else:
            train_set = self._build_dataset(ArrowIPCRecognitionDataset, training_data)
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
                               **kwargs)

        for sample in training_data:
            try:
                dataset.add(**sample)
            except Exception as e:
                logger.warning(str(e))
        if self.hparams.normalization:
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
