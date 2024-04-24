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
import uuid
import torch
import logging
import numpy as np
import torch.nn.functional as F

from typing import Any, Callable, Dict, Literal, TYPE_CHECKING
from torchvision.transforms import v2

from kraken.containers import Segmentation, BaselineLine

if TYPE_CHECKING:
    from os import PathLike
    from torch import nn

logger = logging.getLogger(__name__)


def load_model_checkpoint(filename: 'PathLike', device: torch.device) -> 'nn.Module':
    """
    Instantiates a pure torch nn.Module from a lightning checkpoint and returns
    the class mapping.
    """
    pass

def ocr(*args, **kwargs):
    pass
