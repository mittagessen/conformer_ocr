#
# Copyright 2024 Benjamin Kiessling
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
conformer_ocr.pred
~~~~~~~~~~~~~

Recognition inference
"""
import PIL
import json
import uuid
import torch
import logging
import numpy as np
import tarfile
import tempfile
import lightning.pytorch as L
import safetensors.torch

import torch.nn.functional as F

from torch import nn
from typing import (Any, Callable, Dict, Literal, TYPE_CHECKING, Union, Tuple,
                    Optional, List)

from conformer_ocr.conformer.encoder import ConformerEncoder
from conformer_ocr.codec import TransformerCodec

from kraken.containers import Segmentation, BaselineLine
from kraken.lib.ctc_decoder import greedy_decoder

if TYPE_CHECKING:
    from os import PathLike

logger = logging.getLogger(__name__)


class PytorchRecognitionModel(nn.Module):
    def __init__(self,
                 num_classes: int,
                 height: int,
                 encoder_dim: int,
                 num_encoder_layers: int,
                 num_attention_heads: int,
                 feed_forward_expansion_factor: int,
                 conv_expansion_factor: int,
                 input_dropout_p: float,
                 feed_forward_dropout_p: float,
                 attention_dropout_p: float,
                 conv_dropout_p: float,
                 conv_kernel_size: int,
                 half_step_residual: bool,
                 subsampling_conv_channels: int,
                 subsampling_factor: int,
                 codec: TransformerCodec,
                 ctc_decoder=greedy_decoder,
                 **kwargs):
        """
        A nn.Module version of a conformer_ocr.model.RecognitionModel for
        inference.
        """
        super().__init__()
        encoder = ConformerEncoder(in_channels=1,
                                   input_dim=height,
                                   encoder_dim=encoder_dim,
                                   num_layers=num_encoder_layers,
                                   num_attention_heads=num_attention_heads,
                                   feed_forward_expansion_factor=feed_forward_expansion_factor,
                                   conv_expansion_factor=conv_expansion_factor,
                                   input_dropout_p=input_dropout_p,
                                   feed_forward_dropout_p=feed_forward_dropout_p,
                                   attention_dropout_p=attention_dropout_p,
                                   conv_dropout_p=conv_dropout_p,
                                   conv_kernel_size=conv_kernel_size,
                                   half_step_residual=half_step_residual,
                                   subsampling_conv_channels=subsampling_conv_channels,
                                   subsampling_factor=subsampling_factor)
        decoder = nn.Linear(encoder_dim, num_classes, bias=True)
        self.nn = nn.ModuleDict({'encoder': encoder,
                                 'decoder': decoder})

        self.codec = codec
        self.ctc_decoder = ctc_decoder
        self.height = height
        self.channels = 1
        self.width = 0


    def forward(self, line: torch.Tensor, lens: torch.Tensor = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Performs a forward pass on a torch tensor of one or more lines with
        shape (N, C, H, W) and returns a numpy array (N, W, C).

        Args:
            line: NCHW line(s) tensor
            lens: Optional tensor containing sequence lengths if N > 1

        Returns:
            Tuple with (N, W, C) shaped numpy array and final output sequence
            lengths.

        Raises:
            KrakenInputException: Is raised if the channel dimension isn't of
                                  size 1 in the network output.
        """
        with torch.no_grad():
            line = line.squeeze(1).transpose(1, 2)
            encoder_outputs, encoder_lens = self.nn.encoder(line, lens)
            probits = self.nn.decoder(encoder_outputs)
        return probits, encoder_lens

    def predict(self, line: torch.Tensor, lens: Optional[torch.Tensor] = None) -> List[List[Tuple[str, int, int, float]]]:
        """
        Performs a forward pass on a torch tensor of a line with shape (N, C, H, W)
        and returns the decoding as a list of tuples (string, start, end,
        confidence).

        Args:
            line: NCHW line tensor
            lens: Optional tensor containing sequence lengths if N > 1

        Returns:
            List of decoded sequences.
        """
        o, olens = self.forward(line, lens)
        o = o.transpose(1, 2).cpu().float().numpy()

        dec_seqs = []
        pred = []
        for seq, seq_len in zip(o, olens):
            locs = self.ctc_decoder(seq[:, :seq_len])
            dec_seqs.append(''.join(x[0] for x in self.codec(locs)))
        return dec_seqs


    def predict_string(self, line: torch.Tensor, lens: Optional[torch.Tensor] = None) -> List[str]:
        """
        Performs a forward pass on a torch tensor of a line with shape (N, C, H, W)
        and returns a string of the results.

        Args:
            line: NCHW line tensor
            lens: Optional tensor containing the sequence lengths of the input batch.
        """
        o, olens = self.forward(line, lens)
        o = o.transpose(1, 2).cpu().float().numpy()

        dec_strs = []
        for seq, seq_len in zip(o, olens):
            locs = self.ctc_decoder(seq[:, :seq_len])
            dec_strs.append(''.join(x[0] for x in self.codec.decode(locs)))
        return dec_strs

    def predict_labels(self, line: torch.tensor, lens: torch.Tensor = None) -> List[List[Tuple[int, int, int, float]]]:
        """
        Performs a forward pass on a torch tensor of a line with shape (N, C, H, W)
        and returns a list of tuples (class, start, end, max). Max is the
        maximum value of the softmax layer in the region.
        """
        o, olens = self.forward(line, lens)
        o = o.transpose(1, 2).cpu().float().numpy()

        oseqs = []
        for seq, seq_len in zip(o, olens):
            oseqs.append(self.ctc_decoder(seq[:, :seq_len]))
        return oseqs

    @classmethod
    def load_safetensors(cls, path: 'PathLike'):
        """
        Loads a safetensors archive
        """
        with tarfile.open(path, 'r') as tf:
            metadata = json.load(tf.extractfile('metadata.json'))
            if not 'codec' in metadata:
                raise ValueError('No codec in metadata record')
            codec = TransformerCodec(metadata['codec'])
            if not 'hyper_params' in metadata:
                raise ValueError('No hyperparameters in metadata record')
            net = cls(**metadata['hyper_params'], codec=codec)
            weights = safetensors.torch.load(tf.extractfile('model.safetensors').read())
        net.nn.load_state_dict(weights)
        return net.eval()

    @classmethod
    def load_checkpoint(cls, path: 'PathLike'):
        """
        Loads a lightning checkpoint
        """
        state_dict = torch.load(path, map_location='cpu')
        if not 'TextLineDataModule' in state_dict:
            raise ValueError('Checkpoint does not contain data module state.')
        codec = TransformerCodec(state_dict['TextLineDataModule']['codec'])
        if not 'hyper_parameters' in state_dict:
            raise ValueError('No hyperparameters in state_dict')
        net = cls(**state_dict['hyper_parameters'], codec=codec)
        net.load_state_dict(state_dict['state_dict'], strict=False)
        return net.eval()


def checkpoint_to_safetensors(model: 'nn.Module' = None,
                              data_module: 'nn.Module' = None,
                              checkpoint_path: 'PathLike' = None,
                              output_path: 'PathLike' = None,
                              metadata: 'PathLike' = None):
    """
    Converts a pytorch lightning checkpoint of a RecognitionModel and
    TextLineDataModule into a safetensors output file containing the necessary
    metadata for inference.

    Args:
    """
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if not 'TextLineDataModule' in state_dict:
        raise ValueError('Checkpoint does not contain data module state.')
    codec = state_dict['TextLineDataModule']['codec']
    net = RecognitionModel.load_from_checkpoint(checkpoint_path, map_location='cpu')
    kraken_metadata = {'codec': codec,
                       'seg_type': 'baselines',
                       'one_channel_mode': 'L',
                       'model_type': 'recognition',
                       'legacy_polygons': False,
                       'hyper_params': dict(net.hparams),
                       'kraken_min_version': 5.3,
                      }

    if metadata:
        with open(metadata, 'r') as fp:
            metadata = json.load(fp)
    else:
        metadata = {}

    with tempfile.TemporaryDirectory() as tmp_output_dir:

        safetensors.torch.save_file(net.nn.state_dict(),
                                    filename=tmp_output_dir + '/model.safetensors')
        with open(tmp_output_dir + '/_kraken_metadata.json', 'w') as fp:
            json.dump(metadata, fp)
        with tarfile.open(output_path, 'w') as tar_p:
            tar_p.add(tmp_output_dir + '/model.safetensors', arcname='model.safetensors')
            tar_p.add(tmp_output_dir + '/_kraken_metadata.json', arcname='_kraken_metadata.json')
            tar_p.add(tmp_output_dir + '/metadata.json', arcname='metadata.json')
