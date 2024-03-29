"""Classes and functions for I/O Operations."""
import csv
import json
import os
import pickle
from typing import (
    Any,
    Collection,
    Dict,
    List,
    Optional,
)

from PIL import Image


def load_image(file_path: str, mode: str = 'RGB') -> Image:
    """Load an image from a path."""
    img = Image.open(file_path)
    img.load()
    return img.convert(mode)


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


def load_csv_dict(file_path: str) -> List[Dict[str, str]]:
    """Read a comma separated file with header information."""
    with open(file_path, 'r') as f:
        csv_reader = csv.DictReader(f, delimiter=',')
        return [r for r in csv_reader]


def load_pickle(filepath: str) -> Any:
    """Load a binary file to a python object."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


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


def save_as_csv_dict(
    rows: List[Dict[str, str]], target: str, header: Collection
) -> None:
    """Save a list of rows to a csv file."""
    with open(target, 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def save_as_pickle(file: Any, filepath: str) -> None:
    """Save a generic object to a binary file."""
    file_dir, _ = os.path.split(filepath)
    os.makedirs(file_dir, exist_ok=True)

    with open(filepath, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
