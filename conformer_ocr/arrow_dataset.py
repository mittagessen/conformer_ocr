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
Arrow IPC format dataset builder for polygon-free training
"""
__all__ = ['build_binary_dataset']

import io
import json
import tempfile
from collections import Counter
from functools import partial
from multiprocessing import Pool
from typing import TYPE_CHECKING, Literal, Callable, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
from PIL import Image, UnidentifiedImageError


from scipy.special import comb
from shapely.geometry import LineString, Polygon

from shapely.ops import clip_by_rect

from kraken.containers import Segmentation, BaselineLine
from kraken.lib.exceptions import KrakenInputException
from kraken.lib.util import is_bitonal, make_printable
from kraken.lib.xml import XMLPage

if TYPE_CHECKING:
    from os import PathLike

import logging

logger = logging.getLogger(__name__)


# magic lsq cubic bezier fit function from the internet.
def Mtk(n, t, k):
    return t**k * (1-t)**(n-k) * comb(n,k)


def BezierCoeff(ts):
    return [[Mtk(3,t,k) for k in range(4)] for t in ts]


def bezier_fit(bl):
    x = bl[:, 0]
    y = bl[:, 1]
    dy = y[1:] - y[:-1]
    dx = x[1:] - x[:-1]
    dt = (dx ** 2 + dy ** 2)**0.5
    t = dt/dt.sum()
    t = np.hstack(([0], t))
    t = t.cumsum()

    Pseudoinverse = np.linalg.pinv(BezierCoeff(t))  # (9,4) -> (4,9)

    control_points = Pseudoinverse.dot(bl)  # (4,9)*(9,2) -> (4,2)
    medi_ctp = control_points[1:-1,:]
    return medi_ctp


def convert_line(image: Image.Image, rec: BaselineLine, min_points: int = 8):
    """
    Converts a baseline to a Bezier representation and crops the input image
    roughly around the line.
    """
    baseline = np.array(rec.baseline)
    if len(baseline) < min_points:
        ls = LineString(rec.baseline)
        baseline = np.stack([np.array(ls.interpolate(x, normalized=True).coords)[0] for x in np.linspace(0, 1, 8)])
    # get a rough environment from the bounding polygon
    pol = Polygon(rec.boundary).envelope
    buff = min(np.abs(pol.bounds[0] - pol.bounds[2]), np.abs(pol.bounds[1] - pol.bounds[3]))
    patch = clip_by_rect(pol.buffer(buff).envelope, 0, 0, image.width, image.height).bounds
    # control points normalized to patch extents
    curve = ((np.concatenate(([baseline[0]], bezier_fit(baseline),
                              [baseline[-1]])) - (patch[0],
                                                  patch[1]))/(patch[2]-patch[0],
                                                              patch[3]-patch[1])).flatten().tolist()
    line_im = image.crop(patch)
    return line_im, curve, rec.text


def _extract_line(xml_record, skip_empty_lines: bool = True):
    lines = []
    try:
        im = Image.open(xml_record.imagename)
    except (FileNotFoundError, UnidentifiedImageError):
        return lines, None, None
    if is_bitonal(im):
        im = im.convert('1')
    for idx, rec in enumerate(xml_record.lines):
        try:
            line_im, curve, text = convert_line(im, rec)
        except Exception as e:
            logger.warning(f'Unexpected exception {e} from line {idx} in {xml_record.imagename}')
            continue
        if not text and skip_empty_lines:
            continue
        fp = io.BytesIO()
        line_im.save(fp, format='png')
        lines.append({'text': text, 'curve': curve, 'im': fp.getvalue()})
    return lines, im.mode


def build_binary_dataset(files: Optional[List[Union[str, 'PathLike', 'Segmentation']]] = None,
                         output_file: Union[str, 'PathLike'] = None,
                         format_type: Literal['xml', 'alto', 'page'] = 'xml',
                         num_workers: int = 0,
                         ignore_splits: bool = False,
                         random_split: Optional[Tuple[float, float, float]] = None,
                         recordbatch_size: int = 100,
                         skip_empty_lines: bool = True,
                         callback: Callable[[int, int], None] = lambda chunk, lines: None) -> None:
    """
    Parses XML files and dumps the baseline-style line images and text into a
    binary dataset.

    Args:
        files: List of XML input files or Segmentation container objects.
        output_file: Path to the output file.
        format_type: One of `xml`, `alto`, `page`.
        num_workers: Number of workers for parallelized extraction of line
                     images. Set to `0` to disable parallelism.
        ignore_splits: Switch to disable serialization of the explicit
                       train/validation/test splits contained in the source
                       files.
        random_split: Serializes a random split into the dataset with the
                       proportions (train, val, test).
        recordbatch_size: Minimum number of records per RecordBatch written to
                          the output file. Larger batches require more
                          transient memory but slightly improve reading
                          performance.
        skip_empty_lines: Do not compile empty text lines into the dataset.
        callback: Function called every time a new recordbatch is flushed into
                  the Arrow IPC file.
    """

    logger.info('Parsing XML files')
    extract_fn = partial(_extract_line, skip_empty_lines=skip_empty_lines)
    parse_fn = None
    if format_type in ['xml', 'alto', 'page']:
        parse_fn = XMLPage
    else:
        raise ValueError(f'invalid format {format_type} for parse_(xml,alto,page)')

    docs = []
    if parse_fn:
        for doc in files:
            try:
                data = parse_fn(doc).to_container()
            except (FileNotFoundError, KrakenInputException, ValueError):
                logger.warning(f'Invalid input file {doc}')
                continue
            try:
                imagename = data.imagename
                with open(imagename, 'rb') as fp:
                    Image.open(fp)
            except (FileNotFoundError, UnidentifiedImageError) as e:
                logger.warning(f'Could not open file {e.filename} in {doc}')
                continue
            docs.append(data)
        logger.info(f'Parsed {len(docs)} files.')
    else:
        docs = files.copy()
        logger.info(f'Got {len(docs)} preparsed files.')

    logger.info('Assembling dataset alphabet.')
    alphabet = Counter()
    num_lines = 0
    for doc in docs:
        for line in doc.lines:
            num_lines += 1
            alphabet.update(line.text)

    callback(0, num_lines)

    for k, v in sorted(alphabet.items(), key=lambda x: x[1], reverse=True):
        char = make_printable(k)
        if char == k:
            char = '\t' + char
        logger.info(f'{char}\t{v}')

    ds_type = 'cocr_recognition_nopo'

    metadata = {'lines': {'type': ds_type,
                          'alphabet': alphabet,
                          'text_type': 'raw',
                          'image_type': 'raw',
                          'splits': ['train', 'eval', 'test'],
                          'im_mode': '1',
                          'counts': Counter({'all': 0,
                                             'train': 0,
                                             'validation': 0,
                                             'test': 0
                                             }
                                            ),
                          }
                }

    ty = pa.struct([('text', pa.string()), ('im', pa.binary())])
    schema = pa.schema([('lines', ty), ('train', pa.bool_()), ('validation', pa.bool_()), ('test', pa.bool_())])

    def _make_record_batch(line_cache):
        ar = pa.array(line_cache, type=ty)
        if random_split:
            indices = np.random.choice(4, len(line_cache), p=(0.0,) + random_split)
        else:
            indices = np.zeros(len(line_cache))
        tr_ind = np.zeros(len(line_cache), dtype=bool)
        tr_ind[indices == 1] = True
        val_ind = np.zeros(len(line_cache), dtype=bool)
        val_ind[indices == 2] = True
        test_ind = np.zeros(len(line_cache), dtype=bool)
        test_ind[indices == 3] = True

        train_mask = pa.array(tr_ind)
        val_mask = pa.array(val_ind)
        test_mask = pa.array(test_ind)
        rbatch = pa.RecordBatch.from_arrays([ar, train_mask, val_mask, test_mask], schema=schema)
        return rbatch, (len(line_cache), int(sum(indices == 1)), int(sum(indices == 2)), int(sum(indices == 3)))

    line_cache = []
    logger.info('Writing lines to temporary file.')
    with tempfile.TemporaryDirectory() as tmp_output_dir:
        tmp_file = tmp_output_dir + '/dataset.arrow'
        with pa.OSFile(tmp_file, 'wb') as sink:
            with pa.ipc.new_file(sink, schema) as writer:

                if num_workers and num_workers > 1:
                    logger.info(f'Spinning up processing pool with {num_workers} workers.')
                    with Pool(num_workers) as pool:
                        for page_lines, im_mode in pool.imap_unordered(extract_fn, docs):
                            if page_lines:
                                line_cache.extend(page_lines)
                                # comparison RGB(A) > L > 1
                                if im_mode > metadata['lines']['im_mode']:
                                    metadata['lines']['im_mode'] = im_mode

                            if len(line_cache) >= recordbatch_size:
                                logger.info(f'Flushing {len(line_cache)} lines into {tmp_file}.')
                                rbatch, counts = _make_record_batch(line_cache)
                                metadata['lines']['counts'].update({'all': counts[0],
                                                                    'train': counts[1],
                                                                    'validation': counts[2],
                                                                    'test': counts[3]})
                                writer.write(rbatch)
                                callback(len(line_cache), num_lines)
                                line_cache = []
                else:
                    for page_lines, im_mode in map(extract_fn, docs):
                        if page_lines:
                            line_cache.extend(page_lines)
                            # comparison RGB(A) > L > 1
                            if im_mode > metadata['lines']['im_mode']:
                                metadata['lines']['im_mode'] = im_mode

                        if len(line_cache) >= recordbatch_size:
                            logger.info(f'Flushing {len(line_cache)} lines into {tmp_file}.')
                            rbatch, counts = _make_record_batch(line_cache)
                            metadata['lines']['counts'].update({'all': counts[0],
                                                                'train': counts[1],
                                                                'validation': counts[2],
                                                                'test': counts[3]})
                            writer.write(rbatch)
                            callback(len(line_cache), num_lines)
                            line_cache = []

                if line_cache:
                    logger.info(f'Flushing last {len(line_cache)} lines into {tmp_file}.')
                    rbatch, counts = _make_record_batch(line_cache)
                    metadata['lines']['counts'].update({'all': counts[0],
                                                        'train': counts[1],
                                                        'validation': counts[2],
                                                        'test': counts[3]})
                    writer.write(rbatch)
                    callback(len(line_cache), num_lines)

        logger.info('Dataset metadata')
        logger.info(f"type: {metadata['lines']['type']}\n"
                    f"text_type: {metadata['lines']['text_type']}\n"
                    f"image_type: {metadata['lines']['image_type']}\n"
                    f"splits: {metadata['lines']['splits']}\n"
                    f"im_mode: {metadata['lines']['im_mode']}\n"
                    f"lines: {metadata['lines']['counts']}\n")

        with pa.memory_map(tmp_file, 'rb') as source:
            logger.info(f'Rewriting output ({output_file}) to update metadata.')
            ds = pa.ipc.open_file(source).read_all()
            metadata['lines']['counts'] = dict(metadata['lines']['counts'])
            metadata['lines'] = json.dumps(metadata['lines'])
            schema = schema.with_metadata(metadata)
            with pa.OSFile(output_file, 'wb') as sink:
                with pa.ipc.new_file(sink, schema) as writer:
                    writer.write(ds)
