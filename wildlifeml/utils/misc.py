"""Miscellaneous utilities."""
import math
import os
from typing import (
    List,
    Optional,
    Tuple,
)
from urllib import request

import numpy as np
from PIL import Image, ImageDraw

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
    x_coords: Tuple[int, int],
    y_coords: Tuple[int, int],
    outline: str = 'red',
    border_width: int = 10,
) -> Image:
    """Render a bounding box into a PIL Image."""
    img_draw = ImageDraw.Draw(img)
    img_draw.rectangle(
        xy=((x_coords[0], y_coords[0]), (x_coords[1], y_coords[1])),
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
