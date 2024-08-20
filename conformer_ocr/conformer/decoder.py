# Copyright (c) 2021, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from conformer_ocr.conformer.embedding import PositionalEncoding


class TransformerDecoder(nn.Module):
    """
    Transformer decoder for Conformer models.
    """
    def __init__(self,
                 num_classes: int,
                 decoder_dim: int = 512,
                 num_decoder_heads: int = 4,
                 decoder_d_ffn: int = 1024,
                 activation: str = 'gelu',
                 num_decoder_layers: int = 4,
                 sos_id: int = -1,
                 eos_id: int = -1):
        super(TransformerDecoder, self).__init__()
        layer = nn.TransformerDecoderLayer(d_model=decoder_dim,
                                           nhead=num_decoder_heads,
                                           dim_feedforward=decoder_d_ffn,
                                           activation=activation)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_decoder_layers)
        self.embedding = nn.Embedding(num_embeddings=num_classes,
                                      embedding_dim=decoder_dim)
        self.fc = nn.Linear(decoder_dim, num_classes)
        self.positional_encoding = PositionalEncoding(decoder_dim)
        self.sos_id = sos_id
        self.eos_id = eos_id

    def forward(self, tgt: Tensor, memory: Tensor, memory_key_padding_mask: Tensor) -> Tensor:
        """
        Forward propagate a `inputs` for decoder training.

        Args:
            tgt (torch.FloatTensor): Teacher forcing target length
            memory (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            (Tensor, Tensor)

            * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * output_lengths (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        tgt_embed = self.positional_encoding(self.embedding(tgt).permute(1, 0, 2))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_embed.size(0),
                                                                  tgt.device)
        decoder_out = self.decoder(target=tgt,
                                   memory=memory,
                                   tgt_mask=tgt_mask,
                                   memory_key_padding_mask=memory_key_padding_mask)
        logits = self.fc(decoder_out)  # WNC
        return logits.permute(1, 0, 2)  # NWC

    def predict(self,
                memory: Tensor,
                memory_key_padding_mask: Tensor,
                prefix: Optional[Tensor] = None,
                max_len: int = 1024):
        """
        Inference.
        """
        output_tokens = []
        if not prefix:
            prefix = torch.LongTensor([self.sos_id]).unsqueeze(1)

        while len(output_tokens) < max_len:

            prefix_embedding = self.positional_encoding(self.embedding(prefix))

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(prefix.size(0),
                                                                      prefix.device)
            decoder_out = self.decoder(target=prefix_embedding,
                                       memory=memory,
                                       tgt_mask=tgt_mask,
                                       memory_key_padding_mask=memory_key_padding_mask)

            logits = self.fc(decoder_out[-1, :, :])  # 1, vocab_size
            token = logits.argmax(1).item()
            if token == self.eos_id:  # end of generation
                break
            output_tokens.append(token)
            prefix_embedding = torch.cat([prefix,
                                          torch.LongTensor([token]).unsqueeze(1)], dim=0)

        return output_tokens
