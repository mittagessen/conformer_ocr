"""
Implementation of tile-based inference allowing to predict huge images that
does not fit into GPU memory entirely in a sliding-window fashion and merging
prediction mask back to full-resolution.

Based on pytorch-toolbelt's slicer but works with tensors in NCHW format
instead of numpy arrays in HWC.
"""
import dataclasses
import math
from itertools import count
from typing import List, Iterable, Tuple, Union, Sequence

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["ImageSlicer"]


class ImageSlicer(object):
    tile_size: Tuple[int, int]
    tile_step: Tuple[int, int]
    batch_size: int
    """
    Helper class to slice image into tiles and merge them back
    """

    def __init__(self,
                 image_shape: Tuple[int, int],
                 tile_size: Union[int, Tuple[int, int]],
                 overlap: int = (32, 32),
                 batch_size: int = 1):
        """

        :param image_shape: Shape of the source image (H, W)
        :param tile_size: Tile size (Scalar or tuple (H, W)
        :param overlap: Step in pixels between tiles (Scalar or tuple (H, W))
        :param batch_size: Number of tiles returned by split() with each iteration.
        """
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.batch_size = batch_size

        if isinstance(tile_size, Sequence):
            if len(tile_size) != 2:
                raise ValueError(f"Tile size must have exactly 2 elements. Got: tile_size={tile_size}")
            self.tile_size = int(tile_size[0]), int(tile_size[1])
        else:
            self.tile_size = int(tile_size), int(tile_size)

        if isinstance(overlap, (np.ndarray, Sequence)):
            if len(overlap) != 2:
                raise ValueError(f"Overlap must have exactly 2 elements. Got: overlap={overlap}")
            self.tile_step = int(self.tile_size[0] - overlap[0]), int(self.tile_size[1] - overlap[1])
        else:
            self.tile_step = int(self.tile_size[0] - overlap), int(self.tile_size[1] - overlap)
            overlap = int(overlap), int(overlap)
        if self.tile_step <= (1, 1):
            raise ValueError(f'overlap={overlap} is larger in at least one dimension than tile_size={self.tile_size}')

        self.weight = torch.ones(1, *self.tile_size)

        if self.tile_step[0] < 1 or self.tile_step[0] > self.tile_size[0]:
            raise ValueError()
        if self.tile_step[1] < 1 or self.tile_step[1] > self.tile_size[1]:
            raise ValueError()

        self.margin_left = 0
        self.margin_right = 0
        self.margin_top = 0
        self.margin_bottom = 0

        nw = max(1, math.ceil((self.image_width - overlap[1]) / self.tile_step[1]))
        nh = max(1, math.ceil((self.image_height - overlap[0]) / self.tile_step[0]))

        extra_w = self.tile_step[1] * nw - (self.image_width - overlap[1])
        extra_h = self.tile_step[0] * nh - (self.image_height - overlap[0])

        self.margin_left = extra_w // 2
        self.margin_right = extra_w - self.margin_left
        self.margin_top = extra_h // 2
        self.margin_bottom = extra_h - self.margin_top

        crops = []
        bbox_crops = []

        for y in range(
            0, self.image_height + self.margin_top + self.margin_bottom - self.tile_size[0] + 1, self.tile_step[0]
        ):
            for x in range(
                0, self.image_width + self.margin_left + self.margin_right - self.tile_size[1] + 1, self.tile_step[1]
            ):
                crops.append((x, y, self.tile_size[1], self.tile_size[0]))
                bbox_crops.append((x - self.margin_left, y - self.margin_top, self.tile_size[1], self.tile_size[0]))

        self.crops = np.array(crops)
        self.bbox_crops = np.array(bbox_crops)
        self.num_batches = int(np.ceil(len(self.bbox_crops)/self.batch_size))

    def split(self, image: torch.Tensor) -> Iterable[Tuple[torch.Tensor, Tuple[int, int, int, int]]]:
        if (image.shape[1] != self.image_height) or (image.shape[2] != self.image_width):
            raise ValueError()

        tiles = []
        for idx, coords, crop_coords in zip(count(), self.crops, self.bbox_crops):
            x, y, tile_width, tile_height = crop_coords
            x1 = max(x, 0)
            y1 = max(y, 0)
            x2 = min(image.shape[2], x + tile_width)
            y2 = min(image.shape[1], y + tile_height)

            tile = image[:, y1:y2, x1:x2]
            if x < 0 or y < 0 or (x + tile_width) > image.shape[2] or (y + tile_height) > image.shape[1]:
                tile = F.pad(tile, (max(0, -x),
                                    max(0, x + tile_width - image.shape[2]),
                                    max(0, -y),
                                    max(0, y + tile_height - image.shape[1])))
            tiles.append(tile)
            if len(tiles) == self.batch_size or idx == len(self.crops) - 1:
                yield torch.stack(tiles)
                tiles = []

    @property
    def target_shape(self):
        target_shape = (
            self.image_height + self.margin_bottom + self.margin_top,
            self.image_width + self.margin_right + self.margin_left,
        )
        return target_shape

    def merge(self, tiles: torch.Tensor):
        """
        Merges a tensor of shape (N, C, H, W) into (C, H*T, W*T) where T =
        sqrt(N).
        """
        if tiles.shape[0] != len(self.crops):
            raise ValueError

        target_shape = (
            tiles.shape[1],
            self.image_height + self.margin_bottom + self.margin_top,
            self.image_width + self.margin_right + self.margin_left,
        )

        image = torch.zeros(target_shape, device=tiles.device, dtype=tiles.dtype)
        norm_mask = torch.zeros(target_shape, dtype=tiles.dtype)

        w = self.weight.expand(tiles.shape[1], *self.tile_size)

        for tile, (x, y, tile_width, tile_height) in zip(tiles, self.crops):
            image[:, y : y + tile_height, x : x + tile_width] += tile
            norm_mask[:, y : y + tile_height, x : x + tile_width] += w

        norm_mask = norm_mask.clamp(torch.finfo(norm_mask.dtype).eps)
        normalized = image.div(norm_mask)
        crop = self.crop_to_orignal_size(normalized)
        return crop

    def crop_to_orignal_size(self, image):
        assert image.shape[1] == self.target_shape[0]
        assert image.shape[2] == self.target_shape[1]
        crop = image[:,
            self.margin_top : self.image_height + self.margin_top,
            self.margin_left : self.image_width + self.margin_left,
        ]
        assert crop.shape[1] == self.image_height
        assert crop.shape[2] == self.image_width
        return crop
