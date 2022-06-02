"""Classes and functions for I/O Operations."""
import csv
import json
from typing import Any, List

from PIL import Image


def load_image(file_path: str, mode: str = 'RGB') -> Image:
    """Load an image from a path."""
    return Image.open(file_path).convert(mode)


def load_csv(file_path: str, ignore_header: bool = False) -> List[List[str]]:
    """Load a csv with base python."""
    with open(file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        if ignore_header:
            next(reader)
        return [r for r in reader]


def save_as_json(dictionary: Any, target: str) -> None:
    """Save a python object as JSON file."""
    with open(target, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=4)
