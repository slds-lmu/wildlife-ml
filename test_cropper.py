import numpy as np
from PIL import Image
from wildlifeml.preprocessing.megadetector import MegaDetector
from wildlifeml.data import WildlifeDataset

detect = False

megadetector = MegaDetector(batch_size=1, confidence_threshold=0.8)

# if not os.path.exists('example_data_megadetector.json'):
#     md_results = megadetector.predict_directory('example_data')
if detect:
    md_results = megadetector.predict_directory('example_data')
dataset = WildlifeDataset(
    keys=['example4.JPG'],
    label_file_path='example_data/labels.csv',
    detector_file_path='example_data_megadetector.json',
    batch_size=1,
    rectify=True,
    fill=False,
)

example = dataset[0][0][0]
Image.fromarray(example.astype(np.uint8)).show()
