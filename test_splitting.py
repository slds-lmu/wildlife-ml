import os

from wildlifeml.data import do_train_split
from wildlifeml.preprocessing.megadetector import MegaDetector

detect = True
megadetector = MegaDetector(batch_size=1, confidence_threshold=0.8)

root = '../../../../common/bothmannl/'
img_dir = os.path.join(root, 'wildlife_images/usecase2/original_images/')
megadetector.predict_directory(
    img_dir, output_file=os.path.join(root, 'metadata/uc2_md_new.json')
)

train_keys, val_keys, test_keys = do_train_split(
    label_file_path=os.path.join(root, 'metadata/uc2_labels.csv'),
    detector_file_path=os.path.join(root, 'metadata/uc2_md_new.json'),
    min_threshold=0.9,
    splits=(0.6, 0.1, 0.3),
    strategy='random',
)
breakpoint()
