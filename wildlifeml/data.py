"""Classes for accessing data."""
import os
import random
import shutil
from collections import Counter
from copy import deepcopy
from math import ceil
from typing import (
    Dict,
    Final,
    List,
    Optional,
    Tuple,
)

import albumentations as A
import numpy as np
from tensorflow.keras.utils import Sequence
from tqdm import tqdm

from wildlifeml.preprocessing.cropping import Cropper
from wildlifeml.utils.datasets import map_bbox_to_img
from wildlifeml.utils.io import (
    load_csv,
    load_image,
    load_json,
    save_as_csv,
    save_as_json,
)
from wildlifeml.utils.misc import list_files

BBOX_SUFFIX_LEN: Final[int] = 4


class WildlifeDataset(Sequence):
    """Dataset object for handling wildlife datasets with bounding boxes."""

    def __init__(
        self,
        keys: List[str],
        image_dir: str,
        detector_file_path: str,
        batch_size: int,
        label_file_path: Optional[str] = None,
        shuffle: bool = True,
        resolution: int = 224,
        augmentation: Optional[A.Compose] = None,
        do_cropping: bool = True,
        rescale_bbox: bool = True,
        pad: bool = True,
    ) -> None:
        """Initialize a WildlifeDataset object."""
        self.keys = keys
        self.img_dir = image_dir

        if label_file_path is not None:
            self.is_supervised = True
            self.label_dict = {
                key: float(val) for key, val in load_csv(label_file_path)
            }
        else:
            self.is_supervised = False
            self.label_dict = {}

        self.detector_dict = load_json(detector_file_path)

        self.batch_size = batch_size
        self.shuffle = shuffle
        if self.shuffle:
            self._exec_shuffle()

        self.target_resolution = resolution
        self.augmentation = augmentation
        self.do_cropping = do_cropping
        self.cropper = Cropper(rescale_bbox=rescale_bbox, pad=pad)

    def set_keys(self, keys: List[str]) -> None:
        """Change keys in dataset after instantiation. HANDLE WITH CARE."""
        self.keys = keys

    def _exec_shuffle(self) -> None:
        """Shuffle the dataset."""
        random.shuffle(self.keys)

    def on_epoch_end(self) -> None:
        """Execute after every epoch in the keras `.fit()` method."""
        if self.shuffle:
            self._exec_shuffle()

    def __len__(self) -> int:
        """Return the number of batches in the dataset."""
        return ceil(len(self.keys) / self.batch_size)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return a batch with training data and labels on bounding-box level."""
        # Extract keys that correspond to batch.
        start_idx = idx * self.batch_size
        end_idx = min(len(self.keys), start_idx + self.batch_size)
        batch_keys = self.keys[start_idx:end_idx]

        imgs = []
        for key in batch_keys:
            entry = self.detector_dict[key]
            img = np.asarray(load_image(entry['file']))

            # Crop according to bounding box if applicable
            if self.do_cropping and entry.get('bbox') is not None:
                img = self.cropper.crop(img, bbox=entry['bbox'])

            # Resize to target resolution for network
            img = A.resize(
                img, height=self.target_resolution, width=self.target_resolution
            )

            # Apply data augmentations to image
            if self.augmentation is not None:
                img = self.augmentation(image=img)['image']

            imgs.append(img)

        # Extract labels
        if self.is_supervised:
            batch_keys_stem = [map_bbox_to_img(key) for key in batch_keys]
            labels = np.asarray([self.label_dict[key] for key in batch_keys_stem])
        else:
            # We need to add a dummy for unsupervised case because TF.
            labels = np.empty(shape=self.batch_size, dtype=float)

        return np.stack(imgs).astype(float), labels


def modify_dataset(
    dataset: WildlifeDataset,
    extend: bool = False,
    keys: Optional[List[str]] = None,
    new_label_dict: Optional[Dict] = None,
) -> WildlifeDataset:
    """Clone a WildlifeDataset object and extend or subset."""
    if keys is None:
        if new_label_dict is None:
            raise IOError('You need to provide either keys or a label dictionary.')
        keys = list(new_label_dict.keys())

    new_dataset = deepcopy(dataset)
    new_keys = keys + new_dataset.keys if extend else keys
    new_dataset.set_keys(new_keys)

    if extend:
        if not all(x in dataset.detector_dict.keys() for x in keys):
            raise ValueError('No Megadetector results found for provided keys.')
        if dataset.is_supervised:
            if new_label_dict is None:
                raise ValueError(
                    'You are attempting to extend a supervised dataset. '
                    'Please provide labels for keys to be appended.'
                )
            new_dataset.label_dict.update(new_label_dict)

    else:
        if not dataset.is_supervised and new_label_dict is not None:
            new_dataset.label_dict.update(new_label_dict)
            new_dataset.is_supervised = True

    return new_dataset


# --------------------------------------------------------------------------------------


class BBoxMapper:
    """Object for mapping between images and bboxes (et vice versa)."""

    def __init__(self, detector_file_path: str, cache_file_path: str):
        """Initialize BBoxMapper."""
        self.detector_dict = load_json(detector_file_path)
        self.cache_file_path = cache_file_path
        self.keys_img_sorted: List = []

    def map_img_to_bboxes(self) -> None:
        """Create mapping from img to bbox keys and cache."""
        keys_bbox_sorted = sorted(list(self.detector_dict.keys()))
        keys_img = [map_bbox_to_img(k) for k in self.detector_dict.keys()]
        keys_img_sorted = sorted(keys_img)
        cnts = list(Counter(keys_img_sorted).values())

        key_map = {}
        start, end = 0, cnts[0]
        for i in range(len(keys_img_sorted)):
            key_map.update({keys_img_sorted[i]: keys_bbox_sorted[start : end - 1]})
            start = end
            end = cnts[i + 1]
        save_as_json(key_map, self.cache_file_path)


# --------------------------------------------------------------------------------------


class DatasetConverter:
    """
    Object for converting a Dataset to a flattened folder structure.

    This module allows to convert a dataset from the structure:
    ```
    └── images
        ├── class01
            ├── img01
            ├── ...
            └── imgxx
        ├── class02
        ├── class03
        ├── ...
        └── classxx
    ```
    to:
    ```
    ├── labels.csv
    ├── label_map.json
    └── images
        ├── class01_img01
        ├── ...
        └── class01_imgx
    ```
    The images are parsed to a flat hierarchy. The labels obtained from class folders
    are converted to numeric indices and listed in the `labels.csv` file.
    The `label_map.json` contains the mapping table for identifying, which string label
    corresponds to a binary label.
    """

    def __init__(
        self,
        root_dir: str,
        target_dir: str,
        label_file: str = 'labels.csv',
        label_map_file: str = 'label_map.json',
    ) -> None:
        """Initialize a DatasetConverter object."""
        self.root_dir = root_dir
        self.target_dir = target_dir
        self.label_file = label_file
        self.label_map_file = label_map_file

    def convert(self) -> None:
        """Convert the target directory to a flattened structure."""
        os.makedirs(self.target_dir, exist_ok=True)

        class_dirs = os.listdir(self.root_dir)
        label_map = {class_dirs[i]: i for i in range(len(class_dirs))}
        label_list = []

        for cls in (pbar := tqdm(class_dirs)):
            pbar.set_description(f'Processing {cls}')

            orig_root = os.path.join(self.root_dir, cls)
            label = label_map[cls]
            cls_files = list_files(orig_root)

            for file in tqdm(cls_files, leave=False):
                # Add old class directory to new filename for unique identifier
                new_file = cls + '_' + file
                new_path = os.path.join(self.target_dir, new_file)
                old_path = os.path.join(self.root_dir, cls, file)

                # Copy file to new location
                shutil.copyfile(old_path, new_path)

                # Add label to index
                label_list.append((new_file, label))

        # Save labels and label map
        target_root = os.path.abspath(os.path.dirname(self.target_dir))
        save_as_json(label_map, os.path.join(target_root, self.label_map_file))
        save_as_csv(label_list, os.path.join(target_root, self.label_file))
