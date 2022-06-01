"""Miscellaneous utilities."""
import glob
import math
import os
from typing import List, Tuple
from urllib import request

import numpy as np


def download_file(url: str, target_path: str) -> None:
    """Download the content of an url to the target location."""
    if os.path.exists(target_path):
        print('File "{}" already exists. Download is skipped.'.format(target_path))

    target_root = os.path.split(target_path)[0]
    os.makedirs(target_root, exist_ok=True)
    request.urlretrieve(url, target_path)


def list_image_paths(
    directory: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.gif', '.png')
) -> List[str]:
    """Traverse directory recursively and return absolute paths matching extension."""
    file_paths = []
    for filename in glob.iglob(directory + '/**/*', recursive=True):
        if filename.lower().endswith(extensions):
            file_paths.append(os.path.abspath(filename))
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
