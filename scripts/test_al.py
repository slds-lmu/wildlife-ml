"""This script serves as an example to train an active learner with wildlifeml."""
from typing import (
    Dict,
    Final,
    Optional,
)

import albumentations as A
import click
from tensorflow import keras

from wildlifeml import (
    ActiveLearner,
    MegaDetector,
    WildlifeTrainer,
)
from wildlifeml.data import (
    WildlifeDataset,
    do_train_split,
    filter_detector_keys,
)

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
    'splits': (0.7, 0.1, 0.2),
    'al_iterations': 1,
    'al_batch_size': 32,
    'al_acquisitor': 'random',
}


@click.command()
@click.option(
    '--directory_train', '-dt', help='Directory with train images.', required=True
)
@click.option(
    '--label_file_train', '-lft', help='Path to a csv with train labels.', required=True
)
@click.option('--detector_file_train', '-dft', help='Path to train Megadetector .json')
@click.option(
    '--run_detector_train',
    help='Indicate whether to run the Megadetector before training',
    default=False,
)
@click.option(
    '--directory_pool', '-dp', help='Directory with pool images.', required=True
)
@click.option(  # only for experiments without human oracle
    '--label_file_pool', '-lfp', help='Path to a csv with pool labels.', required=True
)
@click.option('--detector_file_pool', '-dfp', help='Path to pool Megadetector .json')
@click.option(
    '--run_detector_pool',
    help='Indicate whether to run the Megadetector for pool data',
    default=False,
)
@click.option(
    '--directory_active', '-da', help='Directory for labeling process.', required=True
)
def main(
    directory_train: str,
    label_file_train: str,
    detector_file_train: Optional[str],
    directory_pool: str,
    label_file_pool: str,
    detector_file_pool: Optional[str],
    directory_active: str,
    run_detector_train: bool = False,
    run_detector_pool: bool = False,
) -> None:
    """Train a supervised classifier on wildlife data."""
    md_error = (
        'You did not pass a detector file and chose not to run the detector '
        'for the data. Please pass a file path or set "run_detector" to "True"'
    )
    if detector_file_train is None and not run_detector_train:
        raise ValueError(md_error + ' for the training data')
    if detector_file_pool is None and not run_detector_pool:
        raise ValueError(md_error + ' for the pool data')

    if run_detector_train:
        # ------------------------------------------------------------------------------
        # DETECT BOUNDING BOXES
        # ------------------------------------------------------------------------------
        md_train = MegaDetector(
            batch_size=CFG['detector_batch_size'],
            confidence_threshold=CFG['detector_confidence_threshold'],
        )
        md_train.predict_directory(
            directory=directory_train, output_file=detector_file_train
        )

    if detector_file_train is None:
        detector_file_train = directory_train + '_megadetector.json'

    if run_detector_pool:
        # ------------------------------------------------------------------------------
        # DETECT BOUNDING BOXES
        # ------------------------------------------------------------------------------
        md_pool = MegaDetector(
            batch_size=CFG['detector_batch_size'],
            confidence_threshold=CFG['detector_confidence_threshold'],
        )
        md_pool.predict_directory(
            directory=directory_pool, output_file=detector_file_pool
        )

    if detector_file_pool is None:
        detector_file_pool = directory_pool + '_megadetector.json'

    # ----------------------------------------------------------------------------------
    # CREATE DATASETS
    # ----------------------------------------------------------------------------------

    # Split active training dataset keys in train, val and test, discarding keys below
    # detection threshold
    train_keys, val_keys, test_keys = do_train_split(
        label_file_path=label_file_train,
        detector_file_path=detector_file_train,
        min_threshold=0.9,
        splits=CFG['splits'],
        strategy=CFG['split_strategy'],
    )
    # Get keys of pool data above detection threshold
    pool_keys = filter_detector_keys(
        label_file_path=label_file_pool,
        detector_file_path=detector_file_pool,
        min_threshold=0.9,
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
        image_dir=directory_train,
        label_file_path=label_file_train,
        detector_file_path=detector_file_train,
        batch_size=CFG['batch_size'],
        resolution=CFG['target_resolution'],
        do_cropping=CFG['cropping'],
        augmentation=augmentation,
        shuffle=True,
    )

    val_dataset = WildlifeDataset(
        keys=val_keys,
        image_dir=directory_train,
        label_file_path=label_file_train,
        detector_file_path=detector_file_train,
        batch_size=CFG['batch_size'],
        resolution=CFG['target_resolution'],
        do_cropping=CFG['cropping'],
        augmentation=augmentation,
        shuffle=False,
    )

    test_dataset = WildlifeDataset(
        keys=test_keys,
        image_dir=directory_train,
        label_file_path=label_file_train,
        detector_file_path=detector_file_train,
        batch_size=CFG['batch_size'],
        resolution=CFG['target_resolution'],
        do_cropping=CFG['cropping'],
        shuffle=False,
    )

    pool_dataset = WildlifeDataset(
        keys=pool_keys,
        image_dir=directory_pool,
        label_file_path=label_file_pool,
        detector_file_path=detector_file_pool,
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
    )

    active_learner = ActiveLearner(
        trainer=trainer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        pool_dataset=pool_dataset,
        al_batch_size=CFG['al_batch_size'],
        active_directory=directory_active,
        acquisitor_name=CFG['al_acquisitor'],
        start_fresh=True,
        test_dataset=test_dataset,
        state_cache='.activecache.json',
        random_state=123,
    )
    active_learner.run()
    breakpoint()
    active_learner.start_fresh = False

    for i in range(CFG['active_learning_iterations']):
        # TODO: put labeled list into staging area
        active_learner.run()


if __name__ == '__main__':
    main()
