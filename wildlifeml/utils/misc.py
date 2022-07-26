"""Miscellaneous utilities."""
import math
import os
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)
from urllib import request

import numpy as np
from PIL import Image, ImageDraw

from wildlifeml.data import map_bbox_to_img, map_img_to_bboxes
from wildlifeml.utils.io import load_json


def download_file(url: str, target_path: str) -> None:
    """Download the content of an url to the target location."""
    if os.path.exists(target_path):
        print('File "{}" already exists. Download is skipped.'.format(target_path))

    target_root = os.path.split(target_path)[0]
    os.makedirs(target_root, exist_ok=True)
    request.urlretrieve(url, target_path)


def list_files(
    directory: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.gif', '.png')
) -> List[str]:
    """List all files in a directory that match the extension."""
    file_paths = []
    for file_name in os.listdir(directory):
        if file_name.lower().endswith(extensions):
            file_paths.append(file_name)
    return file_paths


def truncate_float(x: float, precision: int = 3) -> float:
    """
    Truncate a float scalar to the defined precision.

    For example: truncate_float(0.0003214884) --> 0.000321
    This function is primarily used to achieve a certain float representation
    before exporting to JSON.
    Code is from 'https://github.com/microsoft/CameraTraps'.

    :param x: Scalar to truncate
    :param precision: The number of significant digits to preserve, should be
                      greater or equal 1
    """
    if precision <= 0:
        raise ValueError('Precision needs to be > 0.')

    if np.isclose(x, 0):
        return 0
    else:
        # Determine the factor, which shifts the decimal point of x
        # just behind the last significant digit
        factor = math.pow(10, precision - 1 - math.floor(math.log10(abs(x))))
        # Shift decimal point by multiplication with factor, flooring, and
        # division by factor
        return math.floor(x * factor) / factor


def render_bbox(
    img: Image,
    x_coords: List[Tuple[int, int]],
    y_coords: List[Tuple[int, int]],
    outline: str = 'red',
    border_width: int = 10,
) -> Image:
    """Render a bounding box into a PIL Image."""
    img_draw = ImageDraw.Draw(img)
    for x, y in zip(x_coords, y_coords):
        img_draw.rectangle(
            xy=((x[0], y[0]), (x[1], y[1])),
            outline=outline,
            width=border_width,
        )
    return img


def separate_empties(
    detector_file_path: str,
    conf_threshold: Optional[float] = None,
) -> Tuple[List[str], List[str]]:
    """Separate images into empty and non-empty instances according to Megadetector."""
    detector_dict = load_json(detector_file_path)
    keys_empty = [
        k for k in detector_dict.keys() if detector_dict[k].get('category') == -1
    ]
    if conf_threshold is not None:
        keys_empty.append(
            [
                k
                for k in detector_dict.keys()
                if detector_dict[k].get('conf') < conf_threshold
            ]
        )
    keys_nonempty = list(set(detector_dict.keys()) - set(keys_empty))
    return keys_empty, keys_nonempty


def map_preds_to_img(
    preds_bboxes: Dict[str, float],
    detector_dict: Dict,
) -> Dict[str, int]:
    """Map predictions on bbox level back to img level."""
    keys_imgs = list(set([map_bbox_to_img(k) for k in preds_bboxes.keys()]))
    preds_imgs = {}

    for key in keys_imgs:
        # Find all bbox predictions for img
        keys_k = map_img_to_bboxes(key, detector_dict)
        preds_k = [preds_bboxes[k] for k in keys_k]
        weights_k = [detector_dict[k].get('conf') for k in keys_k]
        weights_k_normalized = [i / sum(weights_k) for i in weights_k]
        pred = sum([i * j for i, j in zip(weights_k_normalized, preds_k)])
        preds_imgs.update({key: pred})

    return preds_imgs
