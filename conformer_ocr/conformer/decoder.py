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
from typing import Optional

from conformer_ocr.conformer.embedding import PositionalEncoding


class TransformerDecoder(nn.Module):
    """
    Transformer decoder for Conformer models.
    """
    def __init__(self,
                 num_classes: int,
                 encoder_dim: int = 512,
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

        if encoder_dim != decoder_dim:
            self.emb_adapter = nn.Linear(encoder_dim, decoder_dim)
        else:
            self.emb_adapter = nn.Identity()

        self.sos_id = sos_id
        self.eos_id = eos_id

    def forward(self,
                tgt: torch.LongTensor,
                memory: torch.FloatTensor,
                memory_key_padding_mask: torch.BoolTensor) -> torch.FloatTensor:
        """
        Forward propagate a `inputs` for decoder training.

        Args:
            tgt (`torch.LongTensor: A sequence of decoder labels with shape (N, S)
            memory: The encoder embeddings with shape (N, W, E)
            memory_key_padding_mask: Encoder padding mask (N, W)

        Returns:
            A Tensor of size (N, W, O)
        """
        tgt_embed = self.positional_encoding(self.embedding(tgt).permute(1, 0, 2))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_embed.size(0),
                                                                  tgt.device)
        memory = self.emb_adapter(memory)

        decoder_out = self.decoder(tgt=tgt_embed,
                                   memory=memory.permute(1, 0, 2),  # WNC
                                   tgt_mask=tgt_mask,
                                   tgt_is_causal=True,
                                   memory_key_padding_mask=memory_key_padding_mask)
        logits = self.fc(decoder_out)  # WNC
        return logits.permute(1, 0, 2)  # NWC

    @torch.no_grad()
    def generate(self,
                 memory: torch.FloatTensor,
                 memory_key_padding_mask: torch.BoolTensor,
                 prompt: Optional[torch.LongTensor] = None,
                 max_len: int = 1024):
        """
        Autoregressive text generation for inference.

        Args:
            memory: (N, W, E)
            memory_key_padding_mask: (N, W)
            prompt: Tensor of size (S) containing the decoded prefix. If None
                    the decoder initializes with the SOS token (optional)
            max_len: maximum length of decoded output sequence
        """
        output_tokens = []
        if not prompt:
            prompt = torch.tensor([[self.sos_id]], dtype=torch.long, device=memory.device)  # NW

        while len(output_tokens) < max_len:
            prompt_embedding = self.positional_encoding(self.embedding(prompt).permute(1, 0, 2))

            decoder_out = self.decoder(tgt=prompt_embedding,
                                       memory=memory.permute(1, 0, 2),
                                       memory_key_padding_mask=memory_key_padding_mask)

            logits = self.fc(decoder_out[-1, :, :].clone().float())  # 1, vocab_size
            new_token = logits.argmax(-1).item()
            if new_token == self.eos_id:  # end of generation
                break
            output_tokens.append(new_token)
            prompt_embedding = torch.cat([prompt,
                                          torch.tensor([[new_token]], dtype=torch.long, device=prompt.device)], dim=0)

        return output_tokens
