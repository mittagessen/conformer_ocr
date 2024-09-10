#
# Copyright 2017 Benjamin Kiessling
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
Pytorch compatible codec with many-to-many mapping between labels and
graphemes.
"""
import torch
import logging
from collections import Counter
from typing import Dict, List, Sequence, Set, Union, Tuple

import numpy as np
from torch import IntTensor

from transformers import ByT5Tokenizer

from kraken.lib.exceptions import KrakenCodecException, KrakenEncodeException

__all__ = ['ByT5Codec']

logger = logging.getLogger(__name__)


class ByT5Codec(object):
    """
    Builds a codec converting between graphemes/code points and integer
    label sequences.

    charset may either be a string, a list or a dict. In the first case
    each code point will be assigned a label, in the second case each
    string in the list will be assigned a label, and in the final case each
    key string will be mapped to the value sequence of integers. In the
    first two cases labels will be assigned automatically. When a mapping
    is manually provided the label codes need to be a prefix-free code.

    Indices 0, 1, and 2 are reserved for padding, SOS, and EOS respectively.
    Output labels and input dictionaries are/should be 3-indexed.

    Encoded sequences have SOS/EOS labels pre-/appended automatically. The
    decoder requires them to be stripped beforehand.

    Args:
        charset: Input character set.
        strict: Flag indicating if encoding/decoding errors should be ignored
                or cause an exception.

    Raises:
        KrakenCodecException: If the character set contains duplicate
                              entries or the mapping is non-singular or
                              non-prefix-free.
    """
    tokenizer = ByT5Tokenizer(eos_token='')
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    # for T5 sos is pad token id
    sos = tokenizer.pad_token_id

    def __init__(self, *args, **kwargs):
        pass

    def __len__(self) -> int:
        """
        Total number of input labels the codec can decode.
        """
        return len(self.tokenizer)

    @property
    def is_valid(self) -> bool:
        """
        Returns True if the codec is prefix-free (in label space) and
        non-singular (in both directions).
        """
        return True

    @property
    def max_label(self) -> int:
        """
        Returns the maximum label value.
        """
        return max(x for x in self.tokenizer.get_vocab().values())

    def encode(self, s: str) -> IntTensor:
        """
        Encodes a string into a sequence of labels.

        If the code is non-singular we greedily encode the longest sequence first.

        Args:
            s: Input unicode string

        Returns:
            Ecoded label sequence
        """
        return torch.tensor(self.tokenizer(s).input_ids, dtype=torch.int)

    def decode(self, labels: IntTensor) -> str:
        """
        Decodes a labelling.

        Given a labelling with cuts and  confidences returns a string with the
        cuts and confidences aggregated across label-code point
        correspondences. When decoding multilabels to code points the resulting
        cuts are min/max, confidences are averaged.

        Args:
            labels: Input containing tuples (label, start, end,
                           confidence).

        Returns:
            A list of tuples (code point, start, end, confidence)
        """
        return self.tokenizer.decode(labels)

    def __repr__(self):
        return 'ByT5Codec()'
