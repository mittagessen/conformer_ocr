import torch
from torch import nn

from typing import Tuple, Dict, List
from dataclasses import dataclass


class Cache(nn.Module):
    """
    A dynamically growing cache.
    """
    def __init__(self) -> None:
        super().__init__()
        self.key_cache: Dict[nn.Module, torch.Tensor] = {}
        self.value_cache: List[nn.Module, torch.Tensor] = {}

    def update(self,
               key_states: torch.Tensor,
               value_states: torch.Tensor,
               layer: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer (`nn.Module`):
                Key of the layer to cache the states for.

        Return:
            A tuple containing the updated key and value states.
        """
        if layer not in self.key_cache:
            self.key_cache[layer] = key_states
            self.value_cache[layer] = value_states
        else:
            self.key_cache[layer] = torch.cat([self.key_cache[layer], key_states], dim=-2)
            self.value_cache[layer] = torch.cat([self.value_cache[layer], value_states], dim=-2)

        return self.key_cache[layer], self.value_cache[layer]

    def get_seq_length(self) -> int:
        if len(self.key_cache):
            return next(iter(self.key_cache.values())).shape[-2]
        else:
            return 0


@dataclass
class DecoderCache:
    self_attention_cache: Cache
    cross_attention_cache: Cache
