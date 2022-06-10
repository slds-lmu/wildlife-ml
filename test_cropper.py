import numpy as np
import os
from PIL import Image
from wildlifeml.preprocessing.cropping import Cropper
from wildlifeml.preprocessing.megadetector import MegaDetector
from wildlifeml.data import WildlifeDataset

megadetector = MegaDetector(batch_size=1, confidence_threshold=0.8)
cropper = Cropper(rectify=False, fill=False)

if not os.path.exists('example_data_megadetector.json'):
    md_results = megadetector.predict_directory('example_data')
dataset = WildlifeDataset(
    keys=['example.JPG', 'example2.JPG'],
    label_file_path='example_data/labels.csv',
    detector_file_path='example_data_megadetector.json',
    batch_size=1,
)
example = dataset[1][0][0]
Image.fromarray(example.astype(np.uint8)).show()
