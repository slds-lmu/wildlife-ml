"""Classes for accessing data."""
import os
import random
import shutil
from copy import deepcopy
from math import ceil
from typing import (
    Any,
    Dict,
    Final,
    List,
    Optional,
    Tuple,
)

import albumentations as A
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
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


def map_img_to_bboxes(img_key: str, detector_dict: Dict) -> List[str]:
    """Find all bbox-level keys for img key."""
    return [k for k in detector_dict.keys() if img_key in k]


def map_bbox_to_img(bbox_key: str) -> str:
    """Find img key for bbox-level key."""
    return bbox_key[: len(bbox_key) - BBOX_SUFFIX_LEN]


# --------------------------------------------------------------------------------------


def do_stratified_splitting(
    detector_dict: Dict,
    img_keys: List[str],
    splits: Tuple[float, float, float],
    meta_dict: Optional[Dict] = None,
    random_state: Optional[int] = None,
) -> Tuple[List[Any], ...]:
    """Perform stratified holdout splitting."""
    keys_array = np.array(img_keys)
    if meta_dict is not None:
        strat_dict = get_strat_dict(meta_dict)
        strat_var_array = np.array([strat_dict[k] for k in img_keys])
    else:
        strat_dict = {}
        strat_var_array = np.empty(len(keys_array))

    # Split intro train and test keys
    sss_tt = StratifiedShuffleSplit(
        n_splits=1,
        train_size=splits[0] + splits[1],
        test_size=splits[2],
        random_state=random_state,
    )
    idx_train, idx_test = next(iter(sss_tt.split(keys_array, strat_var_array)))
    keys_train = img_keys[idx_train]
    keys_test = img_keys[idx_test]
    keys_val = []

    if splits[1] > 0:
        # Split train again into train and val
        keys_array = np.array(keys_train)
        if len(strat_dict) > 0:
            strat_dict_train = {k: v for k, v in strat_dict.items() if k in keys_train}
            strat_var_array = np.array([strat_dict_train[k] for k in keys_train])
        else:
            strat_var_array = np.empty(len(keys_array))
        sss_tv = StratifiedShuffleSplit(
            n_splits=1,
            train_size=splits[0],
            test_size=splits[1],
            random_state=random_state,
        )
        idx_train, idx_val = next(iter(sss_tv.split(keys_array, strat_var_array)))
        keys_train = keys_train[idx_train]
        keys_val = keys_train[idx_val]

    # Get keys on bbox level
    keys_train = [map_img_to_bboxes(k, detector_dict) for k in keys_train]
    keys_val = [map_img_to_bboxes(k, detector_dict) for k in keys_val]
    keys_test = [map_img_to_bboxes(k, detector_dict) for k in keys_test]

    return keys_train, keys_val, keys_test


def do_stratified_cv(
    detector_dict: Dict,
    img_keys: List[str],
    folds: Optional[int],
    meta_dict: Optional[Dict],
    random_state: Optional[int] = None,
) -> Tuple[List[Any], ...]:
    """Perform stratified cross-validation."""
    keys_array = np.array(img_keys)
    if meta_dict is not None:
        strat_dict = get_strat_dict(meta_dict)
        strat_var_array = np.array([strat_dict[k] for k in img_keys])
    else:
        strat_var_array = np.empty(len(keys_array))

    if folds is None:
        raise ValueError('Please provide number of folds in cross-validation.')

    # Get k train-test splits
    skf = StratifiedKFold(n_splits=folds, random_state=random_state)
    idx_train = [list(i) for i, _ in skf.split(keys_array, strat_var_array)]
    idx_test = [list(j) for _, j in skf.split(keys_array, strat_var_array)]
    keys_train, keys_test = [], []
    for i, _ in enumerate(idx_train):
        slice_keys = img_keys[np.array(idx_train[i])]
        keys_train.append([map_img_to_bboxes(k, detector_dict) for k in slice_keys])
    for i, _ in enumerate(idx_test):
        slice_keys = img_keys[np.array(idx_test[i])]
        keys_test.append([map_img_to_bboxes(k, detector_dict) for k in slice_keys])

    return keys_train, keys_test


def get_strat_dict(meta_dict: Dict[str, Dict]) -> Dict[str, str]:
    """Create stratifying variable for dataset splitting."""
    if len(meta_dict) == 0:
        return {}

    else:
        lengths = [len(v) for v in meta_dict.values()]
        if len(set(lengths)) > 1:
            raise ValueError(
                'All variables provided for stratification must have the same number '
                'of elements.'
            )
        strat_dict = {
            k: '_'.join([str(v) for v in meta_dict[k].values()])
            for k in meta_dict.keys()
        }
        return strat_dict


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
