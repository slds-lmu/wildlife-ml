"""Classes and functions for cropping images with bounding boxes."""
from typing import Tuple

import numpy as np


class Cropper:
    """Cropping module for extracting the content of bounding boxes."""

    def __init__(self, rectify: bool = True, fill: bool = True) -> None:
        """Initialize Cropper object."""
        self.rectify = rectify
        self.fill = fill

    def crop(
        self, img: np.ndarray, bbox: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """Crop an image array to a bounding box."""
        height, width, _ = img.shape

        # Convert the relative coords from megadetector output to absolute indices.
        x_coords, y_coords = Cropper.get_absolute_coords(bbox, dims=(height, width))

        if self.rectify:
            # Correct bounding box for rectangular network input
            x_coords, y_coords = Cropper.rectify_bbox(
                x_coords, y_coords, dims=(height, width)
            )

        if self.fill:
            # Ensures a square as output, but could contain black borders
            x_length = x_coords[1] - x_coords[0]
            y_length = y_coords[1] - y_coords[0]
            edge_length = max(x_length, y_length)
            cropped_img = np.pad(
                img[y_coords[0]: y_coords[1], x_coords[0]: x_coords[1]],
                [(0, edge_length - y_length), (0, edge_length - x_length), (0, 0)],
            )
            return cropped_img

        return img[y_coords[0]: y_coords[1] + 1, x_coords[0]: x_coords[1] + 1]

    @staticmethod
    def get_absolute_coords(
        bbox: Tuple[float, float, float, float], dims: Tuple[int, int]
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get array indices from relative bbox positions."""
        height, width = dims

        x_start = int(bbox[0] * width) - 1
        x_end = int((bbox[0] + bbox[2]) * width) - 1
        y_start = int(bbox[1] * height) - 1
        y_end = int((bbox[1] + bbox[3]) * height) - 1

        return (x_start, x_end), (y_start, y_end)

    @staticmethod
    def rectify_bbox(
        x_coords: Tuple[int, int], y_coords: Tuple[int, int], dims: Tuple[int, int]
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Rectify bounding box.

        For maintaining the aspect ratio in the subsequent resizing process,
        the bounding box is rescaled. The longest edge serves as base.
        Note: In very weird aspect ratios and large bounding boxes, this could result
        in non-quadratic bbox outputs!
        """
        height, width = dims

        x_length = (x_coords[1] - x_coords[0])
        y_length = (y_coords[1] - y_coords[0])
        diff_xy = x_length - y_length
        half_diff = 0.5 * abs(diff_xy)

        if diff_xy > 0:
            y_start = max(y_coords[0] - np.ceil(half_diff), 0)
            y_end = min(y_coords[1] + np.floor(half_diff), height - 1)
            y_coords = (int(y_start), int(y_end))
        elif diff_xy < 0:
            x_start = max(x_coords[0] - np.ceil(half_diff), 0)
            x_end = min(x_coords[1] + np.floor(half_diff), width - 1)
            x_coords = (int(x_start), int(x_end))

        return x_coords, y_coords
