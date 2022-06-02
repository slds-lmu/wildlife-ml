"""Classes for accessing data."""
import os
import shutil
from typing import Tuple

import numpy as np
from tensorflow.keras.utils import Sequence
from tqdm import tqdm

from wildlifeml.utils.io import save_as_csv, save_as_json
from wildlifeml.utils.misc import list_files


class WildlifeDataset(Sequence):
    """Dataset object for handling wildlife datasets with bounding boxes."""

    def __init__(self, batch_size: int, shuffle: bool = True) -> None:
        """Initialize a WildlifeDataset object."""
        self.batch_size = batch_size
        self.shuffle = shuffle

    def on_epoch_end(self) -> None:
        """Execute after every epoch in the keras `.fit()` method."""
        if self.shuffle:
            pass

    def __len__(self) -> int:
        """Return the number of batches in the dataset."""
        pass

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return a batch with training data and labels."""
        pass


class DatasetConverter:
    """
    Object for converting a Dataset to a flattened folder structure.

    This module allows to convert a dataset from the structure:
    ```
    └── images
        ├── class01
            ├── img1
            ├── ...
            └── imgx
        ├── class01
        ├── class02
        ├── ...
        └── classxx
    ```
    to:
    ```
    ├── labels.csv
    ├── label_map.json
    └── images
        ├── class01_img1
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
