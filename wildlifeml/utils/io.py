"""Classes and functions for I/O Operations."""
import json
from typing import Any

from PIL import Image


def load_image(file_path: str, mode: str = 'RGB') -> Image:
    """Load an image from a path."""
    return Image.open(file_path).convert(mode)


def save_as_json(dictionary: Any, target: str) -> None:
    """Save a python object as JSON file."""
    with open(target, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=4)
