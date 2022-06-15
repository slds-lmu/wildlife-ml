"""Classes for accessing data."""
import os
import random
import re
import shutil
from math import ceil
from typing import (
    List,
    Optional,
    Tuple,
)

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.image import resize
from tensorflow.keras import Sequential
from tensorflow.keras.utils import Sequence
from tqdm import tqdm

from wildlifeml.preprocessing.cropping import Cropper
from wildlifeml.utils.io import (
    load_csv,
    load_image,
    load_json,
    save_as_csv,
    save_as_json,
)
from wildlifeml.utils.misc import list_files


class WildlifeDataset(Sequence):
    """Dataset object for handling wildlife datasets with bounding boxes."""

    def __init__(
        self,
        keys: List[str],
        label_file_path: str,
        detector_file_path: str,
        batch_size: int,
        shuffle: bool = True,
        resolution: int = 224,
        augmentation: Optional[Sequential] = None,
        do_cropping: bool = True,
        rescale_bbox: bool = True,
        pad: bool = True,
    ) -> None:
        """Initialize a WildlifeDataset object."""
        self.keys = keys
        self.label_dict = {key: val for key, val in load_csv(label_file_path)}

        self.detector_dict = load_json(detector_file_path)

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.target_resolution = resolution
        self.augmentation = augmentation
        self.do_cropping = do_cropping
        self.cropper = Cropper(rescale_bbox=rescale_bbox, pad=pad)

    def on_epoch_end(self) -> None:
        """Execute after every epoch in the keras `.fit()` method."""
        if self.shuffle:
            random.shuffle(self.keys)

    def __len__(self) -> int:
        """Return the number of batches in the dataset."""
        return ceil(len(self.keys) / self.batch_size)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return a batch with training data and labels."""
        # Extract keys that correspond with batch.
        start_idx = idx * self.batch_size
        end_idx = min(len(self.keys), start_idx + self.batch_size)
        batch_keys = self.keys[start_idx:end_idx]

        imgs = []
        for key in batch_keys:
            entry = self.detector_dict[key]
            img = np.asarray(load_image(entry['file']))

            # Crop according to bounding box if applicable
            if self.do_cropping and len(entry['detections']) > 0:
                img = self.cropper.crop(img, bbox=entry['detections'][0]['bbox'])

            # Resize to target resolution for network
            img = resize(img, (self.target_resolution, self.target_resolution))

            # Apply data augmentations to image
            if self.augmentation is not None:
                img = self.augmentation(img)

            imgs.append(img)

        # Extract labels
        labels = np.asarray([self.label_dict[key] for key in batch_keys])

        return np.stack(imgs), labels


# --------------------------------------------------------------------------------------


def do_train_split(
    label_file_path: str,
    splits: Tuple[float, float, float],
    strategy: str = 'random',
    meta_file_path: Optional[str] = None,
    stratifier: Optional[str] = None,
    random_state: Optional[int] = None,
    detector_file_path: Optional[str] = None,
    min_threshold: float = 0.0,
) -> Tuple[List[str], List[str], List[str]]:
    """Split a csv with labels in train & test data and filter with detector results."""
    label_dict = {key: value for key, value in load_csv(label_file_path)}

    # Filter detector results for relevant detections

    if detector_file_path is not None:
        detector_dict = load_json(detector_file_path)
        print(
            'Filtering images with no detected object '
            'and not satisfying minimum threshold.'
        )
        new_keys = [
            key
            for key, val in detector_dict.items()
            if len(val['detections']) > 0 and val['max_detection_conf'] >= min_threshold
        ]
        print(
            'Filtered out {} elements. Current dataset size is {}.'.format(
                len(label_dict) - len(new_keys), len(new_keys)
            )
        )
        label_dict = {
            key: label_dict[key] for key in label_dict.keys() if key in new_keys
        }

    # Define stratification variable (none, class labels or class labels + custom)

    if strategy == 'random':
        stratify = None

    elif strategy == 'class':
        stratify = [val for val in label_dict.values()]

    elif strategy == 'class_plus_custom':
        if meta_file_path is None or stratifier is None:
            raise ValueError(
                f'Strategy "{strategy}" requires metadata and variable specification'
            )
        meta_dict = {
            key: value
            for key, value in load_csv(meta_file_path)
            if key in label_dict.keys()
        }
        stratify = np.dstack(
            ([val for val in label_dict.values()], [val for val in meta_dict.values()])
        ).squeeze(0)

    else:
        raise ValueError('"{}" is not a valid splitting strategy.'.format(strategy))

    # Make stratified split and throw error if stratification variable lacks support

    keys_train, keys_val, keys_test = [], [], []

    stratification_warning = (
        'Stratified sampling is only supported for stratifying '
        'variables with sufficient data support in each category. '
        'Try grouping infrequent categories into larger ones.'
    )
    try:
        keys_train, keys_test = train_test_split(
            [k for k in label_dict.keys()],
            train_size=splits[0] + splits[1],
            test_size=splits[2],
            random_state=random_state,
            stratify=stratify,
        )

    except ValueError as e:
        if bool(re.search('least populated class', str(e))):
            raise ValueError(stratification_warning)

    # Reiterate process to split keys_train in train and val if required

    if splits[1] > 0:

        label_dict = {key: val for key, val in label_dict.items() if key in keys_train}

        if strategy == 'class':
            stratify = [val for val in label_dict.values()]

        elif strategy == 'class_plus_custom':
            meta_dict = {
                key: val for key, val in meta_dict.items() if key in keys_train
            }
            stratify = np.dstack(
                (
                    [val for val in label_dict.values()],
                    [val for val in meta_dict.values()],
                )
            ).squeeze(0)

        try:
            keys_train, keys_val = train_test_split(
                keys_train,
                train_size=splits[0],
                test_size=splits[1],
                random_state=random_state,
                stratify=stratify,
            )
        except ValueError as e:
            if bool(re.search('least populated class', str(e))):
                raise ValueError(stratification_warning)

    return keys_train, keys_val, keys_test


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
            pbar.set_description('Processing {}'.format(cls))

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
