# Copyright 2022 The OpenAI Authors and The HuggingFace Inc. team.
# Copyright 2024 Benjamin Kiessling
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
import torch.nn.functional as F

from torch import Tensor, nn
from typing import Iterable, Optional

from conformer_ocr.conformer.cache import DecoderCache, Cache
from conformer_ocr.conformer.embedding import SinusoidalPositionalEmbedding
from conformer_ocr.conformer.prompt_encoder import PromptEncoder


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x,
                        self.weight.to(x.dtype),
                        None if self.bias is None else self.bias.to(x.dtype))


class Conv1d(nn.Conv1d):
    def _conv_forward(self,
                      x: Tensor,
                      weight: Tensor,
                      bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(x,
                                     weight.to(x.dtype),
                                     None if bias is None else bias.to(x.dtype))


class MultiHeadAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 is_causal: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.scaling = self.head_dim**-0.5
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self,
                hidden_states: torch.Tensor,
                key_value_states: Optional[torch.Tensor] = None,
                past_key_value: Optional[DecoderCache] = None) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self._shape(self.q_proj(hidden_states), tgt_len, bsz)

        if past_key_value is not None:
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        # use key_value_states if cross attention
        current_states = key_value_states if key_value_states is not None else hidden_states
        if is_cross_attention and past_key_value and self in past_key_value.key_cache:
            # reuse k,v, cross_attentions
            key_states = past_key_value.key_cache[self]
            value_states = past_key_value.value_cache[self]
        else:
            key_states = self._shape(self.k_proj(current_states), -1, bsz)
            value_states = self._shape(self.v_proj(current_states), -1, bsz)
            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                key_states, value_states = past_key_value.update(key_states,
                                                                 value_states,
                                                                 self)

        attn_output = torch.nn.functional.scaled_dot_product_attention(query_states,
                                                                       key_states,
                                                                       value_states,
                                                                       attn_mask=None,
                                                                       dropout_p=0.0,
                                                                       is_causal=self.is_causal)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output


class DecoderLayer(nn.Module):
    def __init__(self,
                 decoder_dim: int,
                 num_decoder_heads: int):
        super().__init__()

        self.attn = MultiHeadAttention(decoder_dim, num_decoder_heads, is_causal=True)
        self.attn_ln = LayerNorm(decoder_dim)

        self.cross_attn = MultiHeadAttention(decoder_dim, num_decoder_heads)
        self.cross_attn_ln = LayerNorm(decoder_dim)

        n_mlp = decoder_dim * 4
        self.mlp = nn.Sequential(Linear(decoder_dim, n_mlp),
                                 nn.GELU(),
                                 Linear(n_mlp, decoder_dim))
        self.mlp_ln = LayerNorm(decoder_dim)

    def forward(self,
                x: Tensor,
                xa: Optional[Tensor] = None,
                past_key_value: Optional[DecoderCache] = None):
        x = x + self.attn(self.attn_ln(x), past_key_value=past_key_value)
        x = x + self.cross_attn(self.cross_attn_ln(x), xa, past_key_value=past_key_value)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self,
                 num_classes: int,
                 encoder_dim: int = 512,
                 decoder_dim: int = 512,
                 num_decoder_heads: int = 4,
                 num_decoder_layers: int = 4,
                 sos_id: int = -1,
                 eos_id: int = -1,
                 max_output_len: int = 1024):
        super().__init__()

        if encoder_dim != decoder_dim:
            self.emb_adapter = nn.Linear(encoder_dim, decoder_dim)
        else:
            self.emb_adapter = nn.Identity()

        self.token_embedding = nn.Embedding(num_embeddings=num_classes,
                                            embedding_dim=decoder_dim)

        self.pos_embedding = SinusoidalPositionalEmbedding(5000, decoder_dim)
        self.curve_embedding = PromptEncoder(decoder_dim)

        #self.blocks: Iterable[DecoderLayer] = nn.ModuleList(
        #    [
        #        DecoderLayer(decoder_dim, num_decoder_heads) for _ in range(num_decoder_layers)
        #    ]
        #)
        #self.ln = LayerNorm(decoder_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=decoder_dim, nhead=num_decoder_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc = nn.Linear(decoder_dim, num_classes)

        self.sos_id = sos_id
        self.eos_id = eos_id

        self.max_output_len = max_output_len

    def forward(self,
                tgt: torch.LongTensor,
                memory: torch.FloatTensor,
                curves: Optional[torch.FloatTensor] = None,
                past_key_value: Optional[DecoderCache] = None):
        """
        Args:
            tgt: A sequence of decoder labels with shape (N, S)
            memory: The encoder embeddings with shape (1, W, E). The first
                    dimension automatically gets repeated N times.
            curves: Normalized curve control points with shape (N, 4, 2)
            past_key_value: Optional decoder cache.
        """
        x = self.token_embedding(tgt)
        x = x + self.pos_embedding(tgt.size(), past_key_value_length).to(x.device)

        x = x.to(memory.dtype).transpose(0, 1)

        memory = self.emb_adapter(memory)
        # repeat first dimension N times
        memory = memory.repeat(tgt.size(0), 1, 1)
        # add curve positional embeddings
        memory = memory + self.curve_embedding(curves).unsqueeze(1).expand(-1, memory.size(1), -1)
        memory = memory.transpose(0, 1)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(0),
                                                                  tgt.device)

        x = self.decoder(tgt=x,
                         memory=memory,
                         tgt_mask=tgt_mask,
                         tgt_is_causal=True)

        return self.fc(x)

    @torch.no_grad()
    def generate(self,
                 memory: torch.FloatTensor,
                 curves: torch.FloatTensor,
                 prompt: Optional[torch.LongTensor] = None,
                 use_cache: bool = True):
        """
        Autoregressive text generation for inference. Only works with
        batch_size == 1 for now.

        Args:
            memory: (N, W, E)
            curves: curve control points for the baseline (N, 4, 2)
            prompt: Tensor of size (S) containing the decoded prefix. If None
                    the decoder initializes with the SOS token (optional)
            use_cache: Enables/disables caching
        """
        if use_cache:
            past_key_value = DecoderCache(Cache(), Cache())
        else:
            past_key_value = None

        output_tokens = []
        if prompt is None:
            prompt = torch.tensor([[self.sos_id]], dtype=torch.long, device=memory.device)  # NW

        while len(output_tokens) < self.max_output_len:
            logits = self.forward(tgt=prompt,
                                  memory=memory,
                                  curves=curves,
                                  past_key_value=past_key_value)
            logits = logits[:, -1, :].clone().float()  # 1, vocab_size
            new_token = logits.argmax(-1).item()
            if new_token == self.eos_id:  # end of generation
                break
            output_tokens.append(new_token)
            if use_cache:
                # only run last token after first forward pass
                prompt = torch.tensor([[new_token]], dtype=torch.long, device=memory.device)  # NW
            else:
                prompt = torch.cat([prompt, torch.tensor([[new_token]], dtype=torch.long, device=memory.device)], dim=1)
        return output_tokens
