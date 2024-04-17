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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import lr_scheduler
from torchaudio.transforms import RNNTLoss
from torchaudio.models import emformer_rnnt_base, RNNTBeamSearch
from torchmetrics.text import CharErrorRate, WordErrorRate

from conformer_ocr import default_specs

from importlib_resources import files
from kraken.lib.ctc_decoder import greedy_decoder

if TYPE_CHECKING:
    from os import PathLike

logger = logging.getLogger(__name__)


class TransducerRecognitionModel(pl.LightningModule):
    def __init__(self,
                 num_classes: int,
                 batches_per_epoch: int = 0,
                 pad=16,
                 batch_size=32,
                 quit='fixed',
                 lag=10,
                 optimizer='AdamW',
                 lr=1e-3,
                 momentum=0.9,
                 weight_decay=1e-3,
                 schedule='cosine',
                 step_size=10,
                 gamma=0.1,
                 rop_factor=0.1,
                 rop_patience=5,
                 cos_t_max=30,
                 cos_min_lr=1e-4,
                 warmup=15000,
                 height=96,
                 encoder_input_dim=256,
                 encoder_ffn_dim=1024,
                 num_encoder_layers=18,
                 num_attention_heads=8,
                 feed_forward_expansion_factor=4,
                 conv_expansion_factor=2,
                 input_dropout_p=0.1,
                 feed_forward_dropout_p=0.1,
                 attention_dropout_p=0.1,
                 conv_dropout_p=0.1,
                 conv_kernel_size=9,
                 half_step_residual=True,
                 subsampling_conv_channels=256,
                 subsampling_factor=4,
                 decoder_hidden_state_dim=640,
                 decoder_output_dim=640,
                 decoder_dropout_p=0.1,
                 **kwargs):
        """
        A LightningModule encapsulating the training setup for a text
        recognition model.

        Setup parameters (load, training_data, evaluation_data, ....) are
        named, model hyperparameters (everything in
        `kraken.lib.default_specs.SEGMENTATION_HYPER_PARAMS`) are in in the
        `hyper_params` argument.

        Args:
        """
        super().__init__()

        self.best_epoch = -1
        self.best_metric = 0.0
        self.best_model = None

        self.save_hyperparameters()

        # set multiprocessing tensor sharing strategy
        if 'file_system' in torch.multiprocessing.get_all_sharing_strategies():
            logger.debug('Setting multiprocessing tensor sharing strategy to file_system')
            torch.multiprocessing.set_sharing_strategy('file_system')

        logger.info(f'Creating conformer model with {num_classes} outputs')
        self.nn = emformer_rnnt_base(num_symbols=num_classes+1)
        self.nn.transcriber.load_state_dict(torch.load(files('conformer_ocr').joinpath('emformer_encoder.pkl')))

        #self.nn = conformer_rnnt_model(input_dim=height,
        #                               encoding_dim=decoder_output_dim,
        #                               time_reduction_stride=subsampling_factor,
        #                               conformer_input_dim=encoder_input_dim,
        #                               conformer_ffn_dim=encoder_ffn_dim,
        #                               conformer_num_layers=num_encoder_layers,
        #                               conformer_num_heads=num_attention_heads,
        #                               conformer_depthwise_conv_kernel_size=conv_kernel_size,
        #                               conformer_dropout=conv_dropout_p,
        #                               num_symbols=num_classes,
        #                               symbol_embedding_dim=decoder_hidden_state_dim,
        #                               num_lstm_layers=2,
        #                               lstm_hidden_dim=decoder_hidden_state_dim,
        #                               lstm_layer_norm=True,
        #                               lstm_layer_norm_epsilon=1e-5,
        #                               lstm_dropout=decoder_dropout_p,
        #                               joiner_activation='tanh')

        # loss
        self.criterion = RNNTLoss(reduction="sum", clamp=1.0)

        self.val_cer = CharErrorRate()
        self.val_wer = WordErrorRate()

        self.blank_idx = num_classes

    def forward(self, x, seq_lens=None):
        return self.nn(x, seq_lens)

    def training_step(self, batch, batch_idx):
        # prepend blank token to 
        targets = batch['target']
        prepended_targets = targets.new_empty([targets.size(0), targets.size(1) + 1])
        prepended_targets[:, 1:] = targets
        prepended_targets[:, 0] = self.blank_idx
        prepended_target_lens = batch['target_lens'] + 1

        logits, encoder_lens, _, _ = self.nn(batch['image'].squeeze(1).transpose(1, 2),
                                             batch['seq_lens'],
                                             prepended_targets,
                                             prepended_target_lens)

        loss = self.criterion(logits=logits,
                              targets=batch['target'],
                              logit_lengths=encoder_lens.int(),
                              target_lengths=batch['target_lens'].int())

        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        targets = batch['target']
        prepended_targets = targets.new_empty([targets.size(0), targets.size(1) + 1])
        prepended_targets[:, 1:] = targets
        prepended_targets[:, 0] = 0
        prepended_target_lens = batch['target_lens'] + 1

        logits, encoder_lens, _, _ = self.nn(batch['image'].squeeze(1).transpose(1, 2),
                                             batch['seq_lens'],
                                             prepended_targets,
                                             prepended_target_lens)

        loss = self.criterion(logits=logits,
                              targets=batch['target'],
                              logit_lengths=encoder_lens.int(),
                              target_lengths=batch['target_lens'].int())

        self.log('val_metric', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

        #o, encoder_lens = self.nn.transcribe(batch['image'].squeeze(1).transpose(1, 2),
        #                                     batch['seq_lens'])
        #o = o.transpose(1, 2).cpu().float().numpy()

        #pred = []
        #for seq, seq_len in zip(o, encoder_lens):
        #    locs = greedy_decoder(seq[:, :seq_len])
        #    pred.append(''.join(x[0] for x in self.trainer.datamodule.val_codec.decode(locs)))
        #decoded_targets = []
        #for target, tlen in zip(batch['target'], batch['target_lens']):
        #    decoded_targets.append(''.join([x[0] for x in self.trainer.datamodule.val_codec.decode([(x, 0, 0, 0) for x in target[:tlen]])]))
        #self.val_cer.update(pred, decoded_targets)
        #self.val_wer.update(pred, decoded_targets)

#    def on_validation_epoch_end(self):
#        accuracy = 1.0 - self.val_cer.compute()
#        word_accuracy = 1.0 - self.val_wer.compute()
#
#        if accuracy > self.best_metric:
#            logger.debug(f'Updating best metric from {self.best_metric} ({self.best_epoch}) to {accuracy} ({self.current_epoch})')
#            self.best_epoch = self.current_epoch
#            self.best_metric = accuracy
#        logger.info(f'validation run: total chars {self.val_cer.total} errors {self.val_cer.errors} accuracy {accuracy}')
#        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#        self.log('val_word_accuracy', word_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#        self.log('val_metric', accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)
#        self.val_cer.reset()
#        self.val_wer.reset()

    def save_checkpoint(self, filename):
        self.trainer.save_checkpoint(filename)

    def configure_callbacks(self):
        callbacks = []
        if self.hparams.quit == 'early':
            callbacks.append(EarlyStopping(monitor='val_accuracy',
                                           mode='max',
                                           patience=self.hparams.lag,
                                           stopping_threshold=1.0))

        return callbacks

    # configuration of optimizers and learning rate schedulers
    # --------------------------------------------------------
    #
    # All schedulers are created internally with a frequency of step to enable
    # batch-wise learning rate warmup. In lr_scheduler_step() calls to the
    # scheduler are then only performed at the end of the epoch.
    def configure_optimizers(self):
        return _configure_optimizer_and_lr_scheduler(self.hparams,
                                                     self.nn.parameters(),
                                                     loss_tracking_mode='max')

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # linear warmup between 0 and the initial learning rate `lr` in `warmup`
        # steps.
        if self.hparams.warmup and self.trainer.global_step < self.hparams.warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.hparams.warmup)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr

    def lr_scheduler_step(self, scheduler, metric):
        if not self.hparams.warmup or self.trainer.global_step >= self.hparams.warmup:
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
    lr = hparams.get("lr")
    momentum = hparams.get("momentum")
    weight_decay = hparams.get("weight_decay")
    schedule = hparams.get("schedule")
    gamma = hparams.get("gamma")
    cos_t_max = hparams.get("cos_t_max")
    cos_min_lr = hparams.get("cos_min_lr")
    step_size = hparams.get("step_size")
    rop_factor = hparams.get("rop_factor")
    rop_patience = hparams.get("rop_patience")
    epochs = hparams.get("epochs")
    completed_epochs = hparams.get("completed_epochs")

    # XXX: Warmup is not configured here because it needs to be manually done in optimizer_step()
    logger.debug(f'Constructing {optimizer} optimizer (lr: {lr}, momentum: {momentum})')
    if optimizer in ['Adam', 'AdamW']:
        optim = getattr(torch.optim, optimizer)(params, lr=lr, weight_decay=weight_decay)
    else:
        optim = getattr(torch.optim, optimizer)(params,
                                                lr=lr,
                                                momentum=momentum,
                                                weight_decay=weight_decay)
    lr_sched = {}
    if schedule == 'exponential':
        lr_sched = {'scheduler': lr_scheduler.ExponentialLR(optim, gamma, last_epoch=completed_epochs-1),
                    'interval': 'step'}
    elif schedule == 'cosine':
        lr_sched = {'scheduler': lr_scheduler.CosineAnnealingLR(optim,
                                                                cos_t_max,
                                                                cos_min_lr,
                                                                last_epoch=completed_epochs-1),
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
