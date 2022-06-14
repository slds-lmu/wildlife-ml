import os

from wildlifeml.data import do_train_split
from wildlifeml.preprocessing.megadetector import MegaDetector
from wildlifeml.utils.io import load_csv_long, save_as_csv

detect = False
megadetector = MegaDetector(batch_size=1, confidence_threshold=0.0)

root = '/common/bothmannl/'
img_dir = os.path.join(root, 'wildlife_images/usecase2/original_images/')
if detect:
    megadetector.predict_directory(img_dir, output_file='uc2_md_new.json')

label_file = load_csv_long(os.path.join(root, 'metadata/uc2_labels.csv'))
label_dict = [{x['orig_name']: x['true_class'] for x in label_file}]
save_as_csv(label_dict, 'uc2_labels_new.csv', header=['id', 'class'])

meta_file = label_file
meta_dict = [
    {
        x['orig_name']: {'class': x['true_class'], 'stratifier': x['station']}
        for x in label_file
    }
]
breakpoint()
save_as_csv(meta_dict, 'uc2_meta.csv', header=['id', 'class', 'stratifier'])

train_keys, val_keys, test_keys = do_train_split(
    label_file_path='uc2_labels_new.csv',
    detector_file_path='uc2_md_new.json',
    meta_file_path='uc2_meta.csv',
    min_threshold=0.9,
    splits=(0.6, 0.1, 0.3),
    strategy='random',
)
breakpoint()
