"""Create example data to test AL loop on."""

import os
import shutil

from wildlifeml.data import DatasetConverter

target_dir = 'example_data/'
converter = DatasetConverter(
    root_dir='../wildlife_ml/Images/original_images/', target_dir=target_dir
)
converter.convert()

img_dir = os.path.join(target_dir, 'images')
os.makedirs(img_dir, exist_ok=True)
files = os.listdir(target_dir)

for f in files:
    if f.endswith('.JPG'):
        shutil.move(os.path.join(target_dir, f), img_dir)
