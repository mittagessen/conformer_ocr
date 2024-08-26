# MIT License
#
# Copyright (c) 2022 OpenAI
#               2024 Benjamin Kiessling
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor, nn
from typing import Dict, Iterable, Optional


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


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
    def __init__(self, decoder_dim: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(decoder_dim, decoder_dim)
        self.key = Linear(decoder_dim, decoder_dim, bias=False)
        self.value = Linear(decoder_dim, decoder_dim)
        self.out = Linear(decoder_dim, decoder_dim)

    def forward(self,
                x: Tensor,
                xa: Optional[Tensor] = None,
                mask: Optional[Tensor] = None,
                kv_cache: Optional[dict] = None,):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self,
                      q: Tensor,
                      k: Tensor,
                      v: Tensor,
                      mask: Optional[Tensor] = None):

        n_batch, n_ctx, decoder_dim = q.shape
        scale = (decoder_dim // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self,
                 decoder_dim: int,
                 n_head: int,
                 cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(decoder_dim, n_head)
        self.attn_ln = LayerNorm(decoder_dim)

        self.cross_attn = (
            MultiHeadAttention(decoder_dim, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(decoder_dim) if cross_attention else None

        n_mlp = decoder_dim * 4
        self.mlp = nn.Sequential(
            Linear(decoder_dim, n_mlp), nn.GELU(), Linear(n_mlp, decoder_dim)
        )
        self.mlp_ln = LayerNorm(decoder_dim)

    def forward(self,
                x: Tensor,
                xa: Optional[Tensor] = None,
                mask: Optional[Tensor] = None,
                kv_cache: Optional[dict] = None):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
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
        self.register_buffer('positional_embedding', sinusoids(1024, decoder_dim))
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(decoder_dim, num_decoder_heads, cross_attention=True)
                for _ in range(num_decoder_layers)
            ]
        )
        self.ln = LayerNorm(decoder_dim)
        self.fc = nn.Linear(decoder_dim, num_classes)

        self.sos_id = sos_id
        self.eos_id = eos_id

        self.max_output_len = max_output_len

        self.kv_cache = {}
        self.hooks = []

    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                kv_cache: Optional[dict] = None):
        """
        tgt (`torch.LongTensor: A sequence of decoder labels with shape (N, S)
        memory: The encoder embeddings with shape (N, W, E)
        kv_cache: Dict
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1),
                                                                  tgt.device)

        x = self.token_embedding(tgt) + self.positional_embedding[offset:offset+tgt.shape[-1]]
        x = x.to(memory.dtype)

        memory = self.emb_adapter(memory)

        for block in self.blocks:
            x = block(x, memory, mask=tgt_mask, kv_cache=kv_cache)

        return self.fc(x)

    @torch.no_grad()
    def generate(self,
                 memory: torch.FloatTensor,
                 prompt: Optional[torch.LongTensor] = None):
        """
        Autoregressive text generation for inference. Only works with
        batch_size == 1 for now.

        Args:
            memory: (N, W, E)
            prompt: Tensor of size (S) containing the decoded prefix. If None
                    the decoder initializes with the SOS token (optional)
            max_len: maximum length of decoded output sequence
        """
        if not self.kv_cache:
            self.kv_cache, self.hooks = self.install_kv_cache_hooks()

        output_tokens = []
        if prompt is None:
            prompt = torch.tensor([[self.sos_id]], dtype=torch.long, device=memory.device)  # NW

        while len(output_tokens) < self.max_output_len:
            logits = self.forward(tgt=prompt,
                                  memory=memory,
                                  kv_cache=self.kv_cache)

            logits = logits[-1, :, :].clone().float()  # 1, vocab_size
            new_token = logits.argmax(-1).item()
            if new_token == self.eos_id:  # end of generation
                break
            output_tokens.append(new_token)
            # only run last token after first forward pass
            prompt = torch.tensor([[new_token]], dtype=torch.long, device=memory.device)  # NW

        self.cleanup_caches()
        return output_tokens

    def cleanup_caches(self):
        for hook in self.hooks:
            hook.remove()

        self.kv_cache = {}
        self.hooks = []

    def install_kv_cache_hooks(self, cache: Optional[Dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Args:
            cache: Optional dict to populate the cache.

        Returns:
            cache: A dictionary object mapping the key/value projection modules to its cache
            hooks: List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.apply(install_hooks)
        return cache, hooks
