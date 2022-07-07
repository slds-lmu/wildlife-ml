"""Run active learning without an actual human in the loop."""

import os
import random
from typing import (
    Dict,
    Final,
    Optional,
    Tuple,
)

import albumentations as A
import click
from tensorflow import keras

from wildlifeml import ActiveLearner, MegaDetector
from wildlifeml.data import (
    WildlifeDataset,
    do_train_split,
    filter_detector_keys,
    subset_dataset,
)
from wildlifeml.training.trainer import WildlifeTrainer
from wildlifeml.utils.io import (
    load_csv,
    load_pickle,
    save_as_csv,
)

# Dictionary containing desired params
CFG: Final[Dict] = {
    'batch_size': 32,
    'num_classes': 6,
    'transfer_epochs': 0,
    'finetune_epochs': 1,
    'finetune_layers': 1,
    'model_backbone': 'resnet50',
    'num_workers': 32,
    'eval_metrics': ['accuracy'],
    'finetune_callbacks': [keras.callbacks.EarlyStopping(patience=2)],
    'target_resolution': 224,
    'cropping': True,
    'detector_batch_size': 1,
    'detector_confidence_threshold': 0.1,
    'split_strategy': 'class',
    'splits': (0.7, 0.1, 0.2),
    'al_pool_size': 300,
    'al_iterations': 2,
    'al_batch_size': 8,
    'al_acquisitor': 'entropy',
}


def carve_out_pool_data(
    label_file: str, pool_size: int, dir_img: str
) -> Tuple[str, str]:
    """Mimicking actual AL loop: randomly split data into labeled and pool subsets."""
    label_dict = {key: value for key, value in load_csv(label_file)}
    try:
        pool_keys = random.sample(list(label_dict.keys()), pool_size)
    except ValueError as e:
        if 'Sample larger than population' in str(e):
            e.args = ('Pool size must be smaller than total number of instances.',)
        raise e
    # Create new dicts and csv files for labeled and pool data, respectively, to keep
    # original label file unaltered
    label_dict_pool = {k: v for k, v in label_dict.items() if k in pool_keys}
    label_dict_labeled = {
        k: v for k, v in label_dict.items() if k not in label_dict_pool
    }
    label_files = [
        os.path.join(dir_img, f'label_file_{n}.csv') for n in ['pool', 'labeled']
    ]
    for n, d in zip(label_files, [label_dict_pool, label_dict_labeled]):
        save_as_csv(rows=[(k, v) for k, v in d.items()], target=n)

    return label_files[0], label_files[1]


@click.command()
@click.option('--dir_img', '-di', help='Directory with images.', required=True)
@click.option('--label_file', '-lf', help='Path to a csv with labels.', required=True)
@click.option('--detector_file', '-df', help='Path to Megadetector .json')
@click.option(
    '--run_detector',
    help='Indicate whether to run the Megadetector before training',
    default=False,
)
@click.option('--dir_act', '-da', help='Directory for labeling process.', required=True)
def main(
    dir_img: str,
    label_file: str,
    detector_file: Optional[str],
    dir_act: str,
    run_detector: bool = False,
) -> None:
    """Train a supervised classifier on wildlife data."""
    if detector_file is None and not run_detector:
        raise ValueError(
            'You did not pass a detector file and chose not to run the detector '
            'for the data. Please pass a file path or set "run_detector" to "True".'
        )

    if run_detector:
        # ------------------------------------------------------------------------------
        # DETECT BOUNDING BOXES
        # ------------------------------------------------------------------------------
        md = MegaDetector(
            batch_size=CFG['detector_batch_size'],
            confidence_threshold=CFG['detector_confidence_threshold'],
        )
        md.predict_directory(directory=dir_img, output_file=detector_file)

    if detector_file is None:
        detector_file = dir_img + '_megadetector.json'

    # ----------------------------------------------------------------------------------
    # CREATE DATASETS
    # ----------------------------------------------------------------------------------

    label_file_pool, label_file_labeled = carve_out_pool_data(
        label_file=label_file,
        pool_size=CFG['al_pool_size'],
        dir_img=dir_img,
    )
    label_dict_pool = {k: v for k, v in load_csv(label_file_pool)}
    label_dict_labeled = {k: v for k, v in load_csv(label_file_labeled)}

    # Get keys of pool data above detection threshold
    pool_keys = filter_detector_keys(
        keys=list(label_dict_pool.keys()),
        detector_file_path=detector_file,
        min_threshold=CFG['detector_confidence_threshold'],
    )
    # Split active training dataset keys in train, val and test, keeping only keys
    # above detection threshold
    train_keys, val_keys, test_keys = do_train_split(
        label_file_path=label_file_labeled,
        detector_file_path=detector_file,
        min_threshold=CFG['detector_confidence_threshold'],
        splits=CFG['splits'],
        strategy=CFG['split_strategy'],
    )

    # Declare training augmentation
    augmentation = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ]
    )

    # Initialize wildlife datasets

    labeled_dataset = WildlifeDataset(
        keys=list(label_dict_labeled.keys()),
        image_dir=dir_img,
        label_file_path=label_file_labeled,
        detector_file_path=detector_file,
        batch_size=CFG['batch_size'],
        resolution=CFG['target_resolution'],
        do_cropping=CFG['cropping'],
        augmentation=augmentation,
        shuffle=True,
    )
    train_dataset = subset_dataset(labeled_dataset, train_keys)
    val_dataset = subset_dataset(labeled_dataset, val_keys)
    test_dataset = subset_dataset(labeled_dataset, test_keys)
    for d in [val_dataset, test_dataset]:
        d.shuffle = False
        d.augmentation = None

    pool_dataset = WildlifeDataset(
        keys=pool_keys,
        image_dir=dir_img,
        detector_file_path=detector_file,
        batch_size=CFG['batch_size'],
        resolution=CFG['target_resolution'],
        do_cropping=CFG['cropping'],
        shuffle=False,
    )

    # ----------------------------------------------------------------------------------
    # CONDUCT ACTIVE LEARNING
    # ----------------------------------------------------------------------------------

    trainer = WildlifeTrainer(
        batch_size=CFG['batch_size'],
        loss_func=keras.losses.SparseCategoricalCrossentropy(),
        num_classes=CFG['num_classes'],
        transfer_epochs=CFG['transfer_epochs'],
        finetune_epochs=CFG['finetune_epochs'],
        transfer_optimizer=keras.optimizers.Adam(),
        finetune_optimizer=keras.optimizers.Adam(),
        finetune_layers=CFG['finetune_layers'],
        model_backbone=CFG['model_backbone'],
        transfer_callbacks=None,
        finetune_callbacks=None,
        num_workers=CFG['num_workers'],
        eval_metrics=CFG['eval_metrics'],
    )

    active_learner = ActiveLearner(
        trainer=trainer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        pool_dataset=pool_dataset,
        label_file_path=label_file_labeled,
        al_batch_size=CFG['al_batch_size'],
        active_directory=dir_act,
        acquisitor_name=CFG['al_acquisitor'],
        start_fresh=True,
        test_dataset=test_dataset,
        test_logfile_path=os.path.join(dir_act, 'test_logfile.pkl'),
        state_cache='.activecache.json',
        random_state=123,
    )

    print('---> Running initial AL iteration')
    active_learner.run()
    active_learner.do_fresh_start = False

    for i in range(CFG['al_iterations']):
        print(f'---> Starting AL iteration {i + 1}')
        imgs_to_label = os.listdir(os.path.join(dir_act, 'images'))
        # Mimic annotation process by reading from existing label list
        labels_supplied = [
            (key, value)
            for key, value in load_csv(label_file_pool)
            if key in imgs_to_label
        ]
        save_as_csv(
            rows=labels_supplied, target=os.path.join(dir_act, 'active_labels.csv')
        )
        print('---> Supplied fresh labeled data')
        active_learner.run()

    results = load_pickle(active_learner.test_logfile_path)
    print(results)


if __name__ == '__main__':
    main()
