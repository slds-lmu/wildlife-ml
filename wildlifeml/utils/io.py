"""Classes and functions for I/O Operations."""
import csv
import json
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from PIL import Image


def load_image(file_path: str, mode: str = 'RGB') -> Image:
    """Load an image from a path."""
    return Image.open(file_path).convert(mode)


def load_json(file_path: str) -> Dict:
    """Load a json file as dictionary."""
    with open(file_path, 'r') as f:
        return json.load(f)


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


def save_as_csv(rows: List, target: str, header: Optional[List] = None) -> None:
    """Save a list of rows to a csv file."""
    with open(target, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        if header is not None:
            writer.writerow(header)

        writer.writerows(rows)
