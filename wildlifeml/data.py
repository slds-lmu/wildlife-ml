"""Classes for accessing data."""
import os
import random
import shutil
from copy import deepcopy
from math import ceil
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
)

import albumentations as A
import numpy as np
from sklearn.model_selection import train_test_split
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
        image_dir: str,
        detector_file_path: str,
        batch_size: int,
        label_file_path: Optional[str] = None,
        shuffle: bool = True,
        resolution: int = 224,
        augmentation: Optional[A.Compose] = None,
        do_cropping: bool = True,
        rectify: bool = True,
        fill: bool = True,
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
        self.cropper = Cropper(rectify=rectify, fill=fill)

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
        """Return a batch with training data and labels."""
        # Extract keys that correspond with batch.
        start_idx = idx * self.batch_size
        end_idx = min(len(self.keys), start_idx + self.batch_size)
        batch_keys = self.keys[start_idx:end_idx]

        imgs = []
        for key in batch_keys:
            entry = self.detector_dict[key]
            img_path = os.path.join(self.img_dir, key)
            img = np.asarray(load_image(img_path))

            # Crop according to bounding box if applicable
            if self.do_cropping and len(entry['detections']) > 0:
                img = self.cropper.crop(img, bbox=entry['detections'][0]['bbox'])

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
            labels = np.asarray([self.label_dict[key] for key in batch_keys])
        else:
            # We need to add a dummy for unsupervised case because TF.
            labels = np.empty(shape=self.batch_size, dtype=float)

        return np.stack(imgs).astype(float), labels


def clone_dataset(keys: List[str], dataset: WildlifeDataset) -> WildlifeDataset:
    """Clone a WildlifeDataset object with a new set of keys."""
    new_dataset = deepcopy(dataset)
    new_dataset.set_keys(keys)
    return new_dataset


# --------------------------------------------------------------------------------------


def do_train_split(
    label_file_path: str,
    splits: Tuple[float, float, float],
    strategy: Literal['random', 'class', 'class_plus_custom'],
    stratifier_file_path: Optional[str] = None,
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
    stratify = get_stratifier(
        strategy=strategy,
        label_dict=label_dict,
        stratifier_file_path=stratifier_file_path,
    )

    # Make stratified split and throw error if stratification variable lacks support
    # keys_train, keys_val, keys_test = [], [], []
    keys_train, keys_test = try_split(
        list(label_dict.keys()),
        train_size=splits[0] + splits[1],
        test_size=splits[2],
        random_state=random_state,
        stratify=stratify,
    )
    keys_val = ['']

    # Reiterate process to split keys_train in train and val if required
    if splits[1] > 0:
        label_dict = {key: val for key, val in label_dict.items() if key in keys_train}
        stratify = get_stratifier(
            strategy=strategy,
            label_dict=label_dict,
            stratifier_file_path=stratifier_file_path,
        )
        keys_train, keys_val = try_split(
            keys_train,
            train_size=splits[0] + splits[1],
            test_size=splits[2],
            random_state=random_state,
            stratify=stratify,
        )

    return keys_train, keys_val, keys_test


def get_stratifier(
    strategy: str,
    label_dict: Dict,
    stratifier_file_path: Optional[str] = None,
) -> Any:
    """Create stratifying variable for dataset splitting."""
    if strategy == 'random':
        return None

    elif strategy == 'class':
        return np.array(list(label_dict.values()))

    elif strategy == 'class_plus_custom':
        if stratifier_file_path is None:
            raise ValueError(
                f'Strategy "{strategy}" requires file with key-stratifier pairs'
            )
        stratifier_dict = {
            key: value
            for key, value in load_csv(stratifier_file_path)
            if key in label_dict.keys()
        }
        return np.dstack(
            (list(label_dict.values()), list(stratifier_dict.values()))
        ).squeeze(0)

    else:
        raise ValueError(f'"{strategy}" is not a valid splitting strategy')


def try_split(
    keys: List[str],
    train_size: float,
    test_size: float,
    random_state=Optional[int],
    stratify=Any,
) -> Tuple[List[str], List[str]]:
    """Attempt stratified split with sklearn."""
    stratification_warning = (
        'Stratified sampling is only supported for stratifying '
        'variables with sufficient data support in each category. '
        'Try grouping infrequent categories into larger ones.'
    )
    try:
        return train_test_split(
            keys,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    except ValueError as e:
        if 'least populated class' in str(e):
            e.args = (stratification_warning,)
        raise e


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
