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
import logging
from collections import Counter
from typing import Dict, List, Sequence, Set, Tuple, Union, Optional, Iterator, TYPE_CHECKING

import io
import numpy as np
import sentencepiece as spm
from torch import IntTensor

from kraken.lib.exceptions import KrakenCodecException, KrakenEncodeException

if TYPE_CHECKING:
    from os import PathLike

__all__ = ['SentencePieceCodec']

logger = logging.getLogger(__name__)


class SentencePieceCodec(object):
    """
    Builds a codec converting between code point and integer label sequences
    using the SentencePiece algorithm.

    The `model` and `sentences` argument are mutually exclusive.

    Args:
        model: path to sentencepiece model to load
        sentences: Iterator of strings to use for training a sentencepiece
                   model
        strict: Flag indicating if encoding/decoding errors should be ignored
                or cause an exception.
        vocab_size: Size of the vocabulary when training a SentencePiece model.
    """
    def __init__(self,
                 model: Optional[Union['PathLike', str]] = None,
                 sentences: Optional[Iterator[str]] = None,
                 strict: bool = False,
                 vocab_size: int = 1024):
        super().__init__()
        if model and sentences:
            raise ValueError('`model` and `sentences` arguments are mutually exclusive')

        if model:
            self.spp = spm.SentencePieceProcessor(model_file=model)
        if sentences:
            _model = io.BytesIO()
            spm.SentencePieceTrainer.train(sentence_iterator=sentences,
                                           model_writer=_model,
                                           normalization_rule_name='identity',
                                           remove_extra_whitespaces=False,
                                           split_by_whitespace=False,
                                           character_coverage=1.0,
                                           vocab_size=vocab_size)
            self.spp = spm.SentencePieceProcessor(model_proto=_model.getvalue())

        self.strict = strict

    def __len__(self) -> int:
        """
        Total number of input labels the codec can decode.
        """
        return self.spp.vocab_size()

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
        return self.spp.vocab_size() - 1

    def encode(self, s: str) -> IntTensor:
        """
        Encodes a string into a sequence of labels.

        Args:
            s: Input unicode string

        Returns:
            Encoded label sequence

        Raises:
            KrakenEncodeException: if the a subsequence is not encodable and the
                                   codec is set to strict mode.
        """
        labels = self.spp.encode(s)
        if 0 in labels:
            if self.strict:
                raise KrakenEncodeException(f'Non-encodable sequence {s}. encountered.')
            logger.warning(f'Non-encodable sequence {s} encountered.')
        return IntTensor(labels)

    def decode(self, labels: Sequence[Tuple[int, int, int, float]]) -> List[Tuple[str, int, int, float]]:
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
        if not labels:
            return labels
        proto = self.spp.decode_ids_as_immutable_proto([int(x[0]) for x in labels])
        return [(piece.surface,) + label[1:] for piece, label in zip(proto.pieces, labels)]

    def merge(self, codec) -> None:
        """
        Not supported for Sentencepiece codecs
        """
        raise ValueError('Merging of sentencepiece codecs is not supported.')

    def add_labels(self, charset) -> None:
        """
        Not supported for Sentencepiece codecs
        """
        raise ValueError('Adding labels to sentencepiece codecs is not supported.')
