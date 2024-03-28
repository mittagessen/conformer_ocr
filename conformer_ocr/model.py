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
Training loop interception helpers
"""
import logging
import re
import warnings
from typing import (TYPE_CHECKING, Any, Callable, Dict, Literal, Optional,
                    Sequence, Union)

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from pytorch_lightning.callbacks import Callback, EarlyStopping
from torch.optim import lr_scheduler
from torchmetrics.text import CharErrorRate, WordErrorRate

from conformer_ocr.conformer.encoder import ConformerEncoder

from conformer_ocr import default_specs

from kraken.lib.ctc_decoder import greedy_decoder

if TYPE_CHECKING:
    from os import PathLike

logger = logging.getLogger(__name__)


class RecognitionModel(pl.LightningModule):
    def __init__(self,
                 num_classes: int,
                 hyper_params: Dict = None,
                 batches_per_epoch: int = 0):
        """
        A LightningModule encapsulating the training setup for a text
        recognition model.

        Setup parameters (load, training_data, evaluation_data, ....) are
        named, model hyperparameters (everything in
        `kraken.lib.default_specs.SEGMENTATION_HYPER_PARAMS`) are in in the
        `hyper_params` argument.

        Args:
            hyper_params (dict): Hyperparameter dictionary containing all fields
                                 from
                                 kraken.lib.default_specs.SEGMENTATION_HYPER_PARAMS
            **kwargs: Setup parameters, i.e. CLI parameters of the segtrain() command.
        """
        super().__init__()

        self.best_epoch = -1
        self.best_metric = 0.0
        self.best_model = None

        hyper_params_ = default_specs.RECOGNITION_HYPER_PARAMS.copy()

        if hyper_params:
            hyper_params_.update(hyper_params)

        self.save_hyperparameters()

        # set multiprocessing tensor sharing strategy
        if 'file_system' in torch.multiprocessing.get_all_sharing_strategies():
            logger.debug('Setting multiprocessing tensor sharing strategy to file_system')
            torch.multiprocessing.set_sharing_strategy('file_system')

        logger.info(f'Creating conformer model with {num_classes} outputs')
        self.encoder = ConformerEncoder(input_dim=hyper_params['height'],
                                                 encoder_dim=hyper_params['encoder_dim'],
                                                 num_layers=hyper_params['num_encoder_layers'],
                                                 num_attention_heads=hyper_params['num_attention_heads'],
                                                 feed_forward_expansion_factor=hyper_params['feed_forward_expansion_factor'],
                                                 conv_expansion_factor=hyper_params['conv_expansion_factor'],
                                                 input_dropout_p=hyper_params['input_dropout_p'],
                                                 feed_forward_dropout_p=hyper_params['feed_forward_dropout_p'],
                                                 attention_dropout_p=hyper_params['attention_dropout_p'],
                                                 conv_dropout_p=hyper_params['conv_dropout_p'],
                                                 conv_kernel_size=hyper_params['conv_kernel_size'],
                                                 half_step_residual=hyper_params['half_step_residual'])
        self.fc = nn.Linear(hyper_params['encoder_dim'], num_classes, bias=False)
        self.nn = nn.Sequential(self.encoder, self.fc)

        # loss
        self.criterion = nn.CTCLoss(reduction='sum', zero_infinity=True)

        self.val_cer = CharErrorRate()
        self.val_wer = WordErrorRate()

    def forward(self, x, seq_lens=None):
        encoder_outputs, encoder_lens = self.encoder(x, seq_lens)
        return self.fc(encoder_outputs), encoder_lens

    def training_step(self, batch, batch_idx):
        input, target = batch['image'], batch['target']
        input = input.squeeze(1).transpose(1, 2)
        seq_lens, label_lens = batch['seq_lens'], batch['target_lens']
        encoder_outputs, encoder_lens = self.encoder(input, seq_lens)
        decoder_outputs = self.fc(encoder_outputs)
        logits = nn.functional.log_softmax(decoder_outputs, dim=-1)

        # NCW -> WNC
        loss = self.criterion(logits.transpose(0, 1),  # type: ignore
                              target,
                              encoder_lens,
                              label_lens)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input = batch['image'].squeeze(1).transpose(1, 2)
        o, olens = self.encoder.forward(input, batch['seq_lens'])
        o = self.fc.forward(o).cpu().float().numpy()

        dec_strs = []
        pred = []
        for seq, seq_len in zip(o, olens):
            locs = greedy_decoder(seq[:, :seq_len])
            pred.append(''.join(x[0] for x in self.trainer.datamodule.val_codec.decode(locs)))
        idx = 0
        decoded_targets = []
        for offset in batch['target_lens']:
            decoded_targets.append(''.join([x[0] for x in self.trainer.datamodule.val_codec.decode([(x, 0, 0, 0) for x in batch['target'][idx:idx+offset]])]))
            idx += offset
        self.val_cer.update(pred, decoded_targets)
        self.val_wer.update(pred, decoded_targets)

    def on_validation_epoch_end(self):
        accuracy = 1.0 - self.val_cer.compute()
        word_accuracy = 1.0 - self.val_wer.compute()

        if accuracy > self.best_metric:
            logger.debug(f'Updating best metric from {self.best_metric} ({self.best_epoch}) to {accuracy} ({self.current_epoch})')
            self.best_epoch = self.current_epoch
            self.best_metric = accuracy
        logger.info(f'validation run: total chars {self.val_cer.total} errors {self.val_cer.errors} accuracy {accuracy}')
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_word_accuracy', word_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_metric', accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.val_cer.reset()
        self.val_wer.reset()

    def save_checkpoint(self, filename):
        self.trainer.save_checkpoint(filename)

    def configure_callbacks(self):
        callbacks = []
        if self.hparams.hyper_params['quit'] == 'early':
            callbacks.append(EarlyStopping(monitor='val_accuracy',
                                           mode='max',
                                           patience=self.hparams.hyper_params['lag'],
                                           stopping_threshold=1.0))

        return callbacks

    # configuration of optimizers and learning rate schedulers
    # --------------------------------------------------------
    #
    # All schedulers are created internally with a frequency of step to enable
    # batch-wise learning rate warmup. In lr_scheduler_step() calls to the
    # scheduler are then only performed at the end of the epoch.
    def configure_optimizers(self):
        return _configure_optimizer_and_lr_scheduler(self.hparams.hyper_params,
                                                     self.nn.parameters(),
                                                     loss_tracking_mode='max')

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # linear warmup between 0 and the initial learning rate `lrate` in `warmup`
        # steps.
        if self.hparams.hyper_params['warmup'] and self.trainer.global_step < self.hparams.hyper_params['warmup']:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.hparams.hyper_params['warmup'])
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.hyper_params['lrate']

    def lr_scheduler_step(self, scheduler, metric):
        if not self.hparams.hyper_params['warmup'] or self.trainer.global_step >= self.hparams.hyper_params['warmup']:
            # step OneCycleLR each batch if not in warmup phase
            if isinstance(scheduler, lr_scheduler.OneCycleLR):
                scheduler.step()
            # step every other scheduler epoch-wise
            elif self.trainer.is_last_batch:
                if metric is None:
                    scheduler.step()
                else:
                    scheduler.step(metric)


def _configure_optimizer_and_lr_scheduler(hparams, params, loss_tracking_mode='max'):
    optimizer = hparams.get("optimizer")
    lrate = hparams.get("lrate")
    momentum = hparams.get("momentum")
    weight_decay = hparams.get("weight_decay")
    schedule = hparams.get("schedule")
    gamma = hparams.get("gamma")
    step_size = hparams.get("step_size")
    rop_factor = hparams.get("rop_factor")
    rop_patience = hparams.get("rop_patience")
    epochs = hparams.get("epochs")
    completed_epochs = hparams.get("completed_epochs")

    # XXX: Warmup is not configured here because it needs to be manually done in optimizer_step()
    logger.debug(f'Constructing {optimizer} optimizer (lr: {lrate}, momentum: {momentum})')
    if optimizer in ['Adam', 'AdamW']:
        optim = getattr(torch.optim, optimizer)(params, lr=lrate, weight_decay=weight_decay)
    else:
        optim = getattr(torch.optim, optimizer)(params,
                                                lr=lrate,
                                                momentum=momentum,
                                                weight_decay=weight_decay)
    lr_sched = {}
    if schedule == 'exponential':
        lr_sched = {'scheduler': lr_scheduler.ExponentialLR(optim, gamma, last_epoch=completed_epochs-1),
                    'interval': 'step'}
    elif schedule == 'cosine':
        lr_sched = {'scheduler': lr_scheduler.CosineAnnealingLR(optim, gamma, last_epoch=completed_epochs-1),
                    'interval': 'step'}
    elif schedule == 'step':
        lr_sched = {'scheduler': lr_scheduler.StepLR(optim, step_size, gamma, last_epoch=completed_epochs-1),
                    'interval': 'step'}
    elif schedule == 'reduceonplateau':
        lr_sched = {'scheduler': lr_scheduler.ReduceLROnPlateau(optim,
                                                                mode=loss_tracking_mode,
                                                                factor=rop_factor,
                                                                patience=rop_patience),
                    'interval': 'step'}
    elif schedule != 'constant':
        raise ValueError(f'Unsupported learning rate scheduler {schedule}.')

    ret = {'optimizer': optim}
    if lr_sched:
        ret['lr_scheduler'] = lr_sched

    if schedule == 'reduceonplateau':
        lr_sched['monitor'] = 'val_mean_iu'
        lr_sched['strict'] = False
        lr_sched['reduce_on_plateau'] = True

    return ret
