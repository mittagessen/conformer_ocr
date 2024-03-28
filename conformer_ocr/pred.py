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
import shapely.geometry as geom
import torch.nn.functional as F

from typing import Any, Callable, Dict, Literal, TYPE_CHECKING
from torchvision.transforms import v2

from transformers import SegformerConfig, SegformerForSemanticSegmentation

from segfblla.tiles import ImageSlicer

from kraken.blla import vec_regions, vec_lines
from kraken.containers import Segmentation, BaselineLine
from kraken.lib.segmentation import polygonal_reading_order, is_in_region

if TYPE_CHECKING:
    from os import PathLike
    from torch import nn

logger = logging.getLogger(__name__)


def load_model_checkpoint(filename: 'PathLike', device: torch.device) -> 'nn.Module':
    """
    Instantiates a pure torch nn.Module from a lightning checkpoint and returns
    the class mapping.
    """
    lm = torch.load(filename, map_location=device)
    model_weights = lm['state_dict']
    config = SegformerConfig.from_dict(lm['model_config'])
    net = SegformerForSemanticSegmentation(config)
    for key in list(model_weights):
        model_weights[key.replace("net.", "")] = model_weights.pop(key)
    net.load_state_dict(model_weights)
    net.class_mapping = lm['BaselineDataModule']['class_mapping']
    net.topline = lm['BaselineDataModule'].get('topline', False)
    net.patch_size = lm['BaselineDataModule'].get('patch_size', (512, 512))
    return net

def ocr(*args, **kwargs):
    pass
