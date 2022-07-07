"""Create example data to test AL loop on."""

import os
import shutil
from typing import Final

from wildlifeml.data import DatasetConverter

ROOT_DIR: Final[str] = '../wildlife_ml/Images/original_images/'
TARGET_DIR: Final[str] = 'example_data/'

converter = DatasetConverter(root_dir=ROOT_DIR, target_dir=TARGET_DIR)
converter.convert()

img_dir = os.path.join(TARGET_DIR, 'images')
os.makedirs(img_dir, exist_ok=True)
files = os.listdir(TARGET_DIR)

for f in files:
    if f.endswith('.JPG'):
        shutil.move(os.path.join(TARGET_DIR, f), img_dir)
