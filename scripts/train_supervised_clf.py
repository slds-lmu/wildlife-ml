"""This script serves as an example to train a supervised classifier with wildlifeml."""
from typing import (
    Dict,
    Final,
    Optional,
)

import albumentations as A
import click
from tensorflow import keras

from wildlifeml import MegaDetector, WildlifeTrainer
from wildlifeml.data import WildlifeDataset, do_train_split

# Dictionary containing desired params
CFG: Final[Dict] = {
    'batch_size': 32,
    'num_classes': 6,
    'transfer_epochs': 10,
    'finetune_epochs': 10,
    'finetune_layers': 10,
    'model_backbone': 'resnet50',
    'num_workers': 32,
    'target_resolution': 224,
    'cropping': True,
    'detector_batch_size': 1,
    'detector_confidence_threshold': 0.1,
    'split_strategy': 'random',
    'splits': (0.8, 0.2),
}


@click.command()
@click.option('--directory', '-d', help='Directory with images.', required=True)
@click.option('--label_file', '-lf', help='Path to a csv with labels.', required=True)
@click.option('--detector_file', '-df', help='Path to a Megadetector .json')
@click.option(
    '--run_detector',
    help='Indicate whether to run the Megadetector before training',
    default=False,
)
def main(
    directory: str,
    label_file: str,
    detector_file: Optional[str],
    run_detector: bool = False,
) -> None:
    """Train a supervised classifier on wildlife data."""
    if detector_file is None and not run_detector:
        raise ValueError(
            'You did not pass a detector file and chose not to run the '
            'detector. Please pass a file path or set "run_detector" to '
            '"True"'
        )

    if run_detector:
        # ------------------------------------------------------------------------------
        # DETECT BOUNDING BOXES
        # ------------------------------------------------------------------------------
        md = MegaDetector(
            batch_size=CFG['detector_batch_size'],
            confidence_threshold=CFG['detector_confidence_threshold'],
        )
        md.predict_directory(directory=directory, output_file=detector_file)

    if detector_file is None:
        detector_file = directory + '_megadetector.json'

    # ----------------------------------------------------------------------------------
    # CREATE DATASETS
    # ----------------------------------------------------------------------------------

    # Split dataset keys in train and test.
    train_keys, test_keys = do_train_split(
        label_file_path=label_file,
        detector_file_path=detector_file,
        min_threshold=0.9,
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
    train_dataset = WildlifeDataset(
        keys=train_keys,
        label_file_path=label_file,
        detector_file_path=detector_file,
        batch_size=CFG['batch_size'],
        resolution=CFG['target_resolution'],
        do_cropping=CFG['cropping'],
        augmentation=augmentation,
        shuffle=True,
    )

    test_dataset = WildlifeDataset(
        keys=train_keys,
        label_file_path=label_file,
        detector_file_path=detector_file,
        batch_size=CFG['batch_size'],
        resolution=CFG['target_resolution'],
        do_cropping=CFG['cropping'],
        shuffle=False,
    )

    # ----------------------------------------------------------------------------------
    # CONDUCT TRAINING
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
    )
    trainer.fit(train_dataset, test_dataset)


if __name__ == '__main__':
    main()
