"""Classes and functions for Active Learning."""
import os
import random
from typing import (
    Dict,
    List,
    Optional,
)

import numpy as np

from wildlifeml import WildlifeTrainer
from wildlifeml.data import (
    WildlifeDataset,
    do_train_split,
    update_dataset,
)
from wildlifeml.preprocessing.cropping import Cropper
from wildlifeml.training.acquisitor import AcquisitorFactory
from wildlifeml.utils.io import (
    load_csv,
    load_image,
    load_json,
    save_as_csv,
    save_as_json,
)
from wildlifeml.utils.misc import render_bbox


class ActiveLearner:
    """Interface for Active Learning on wildlife data."""

    def __init__(
        self,
        trainer: WildlifeTrainer,
        train_dataset: WildlifeDataset,
        val_dataset: WildlifeDataset,
        pool_dataset: WildlifeDataset,
        al_batch_size: int = 10,
        active_directory: str = 'active-wildlife',
        acquisitor_name: str = 'random',
        start_fresh: bool = True,
        start_keys: List[str] = None,
        test_dataset: Optional[WildlifeDataset] = None,
        state_cache: str = '.activecache.json',
        random_state: Optional[int] = None,
    ) -> None:
        """Instantiate an ActiveLearner object."""
        self.pool_dataset = pool_dataset
        self.img_dir = self.pool_dataset.img_dir
        self.act_dir = active_directory

        self.trainer = trainer
        # Save initial trainer state with which to begin each iteration
        self.model_dir = os.path.join(active_directory, 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        trainer.save_model(os.path.join(self.model_dir, 'untrained_model.h5'))

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.acquisitor = AcquisitorFactory.get(
            acquisitor_name, top_k=al_batch_size, random_state=random_state
        )
        self.test_dataset = test_dataset
        self.al_batch_size = al_batch_size
        self.random_state = random_state

        self.state_cache_file = state_cache
        self.do_fresh_start = start_fresh
        self.start_keys = start_keys

        # Serves as storage for all active keys and labels.
        self.active_labels: Dict[str, float] = {}
        # Count active learning iterations
        self.active_counter = 1

    def run(self) -> None:
        """Trigger Active Learning process."""
        # ------------------------------------------------------------------------------
        # INITIALIZE RUN
        # ------------------------------------------------------------------------------
        if self.do_fresh_start:
            self.initialize()
            return
        else:
            self.load_state()

        # ------------------------------------------------------------------------------
        # COLLECT RESULTS
        # ------------------------------------------------------------------------------
        self.collect()

        # ------------------------------------------------------------------------------
        # FIT MODEL INCLUDING FRESH DATA
        # ------------------------------------------------------------------------------
        self.fit()

        # ------------------------------------------------------------------------------
        # EVALUATE MODEL
        # ------------------------------------------------------------------------------
        if self.test_dataset is not None:
            self.evaluate()

        # ------------------------------------------------------------------------------
        # SELECT NEW CANDIDATES
        # ------------------------------------------------------------------------------
        candidate_keys = self.pool_dataset.keys
        # TODO: Filter self.active_labels from full target dataset.
        preds = self.predict(candidate_keys)
        staging_keys = self.acquisitor(candidate_keys, preds)
        self.fill_active_stage(staging_keys)
        self.save_state()
        self.active_counter += 1

    def initialize(self) -> None:
        """Initialize AL run as fresh start."""
        if os.path.exists(self.state_cache_file):
            print(
                f'---> Found active learning state {self.state_cache_file}, '
                f'but you also specified to start a new active learning run. '
                f'The file will be deleted and your progress will be lost.'
            )
            print('Do you want to continue? [y / n]')
            if 'y' not in input().lower():
                print('Active Learning setup is aborted.')
                exit()

        os.makedirs(self.act_dir, exist_ok=True)
        os.makedirs(os.path.join(self.act_dir, 'images'), exist_ok=True)

        if self.start_keys is None:
            print(
                f'For this fresh start, {self.al_batch_size} images are randomly '
                f'chosen from your unlabeled dataset.'
            )
            all_keys = self.pool_dataset.keys
            if self.random_state is not None:
                random.seed(self.random_state)
            staging_keys = random.sample(all_keys, self.al_batch_size)
        else:
            staging_keys = self.start_keys

        # Move initial data and exit.
        self.fill_active_stage(staging_keys)
        self.save_state()
        self.active_counter += 1

    def load_state(self) -> None:
        """Initialize active learner from state file."""
        try:
            state = load_json(self.state_cache_file)

            self.img_dir = self.pool_dataset.img_dir
            self.al_batch_size = state['al_batch_size']
            self.act_dir = state['active_directory']
            self.random_state = state['random_state']
            self.active_labels = state['active_labels']
            self.active_counter = state['active_counter']

            self.acquisitor = AcquisitorFactory.get(
                state['acquisitor_name'],
                top_k=self.al_batch_size,
                random_state=self.random_state,
            )

        except IOError:
            print(
                'There is a problem with your state cache file. '
                'Make sure the path is correct and the file is not damaged.'
            )

    def save_state(self) -> None:
        """Save active learner state file in a file."""
        state = {
            'image_directory': self.img_dir,
            'al_batch_size': self.al_batch_size,
            'active_directory': self.act_dir,
            'acquisitor_name': self.acquisitor.__str__(),
            'random_state': self.random_state,
            'active_labels': self.active_labels,
            'active_counter': self.active_counter,
        }
        save_as_json(state, self.state_cache_file)

    def fill_active_stage(self, keys: List[str]) -> None:
        """
        Move data into the active directory.

        After selecting candidate keys, selected images are transferred to the active
        directory, ready for human labeling. The images are located in the directory
        `images` and the `active_labels.csv` contains a template for adding labels.
        """
        target_path = os.path.join(self.act_dir, 'images')

        for key in keys:
            entry = self.pool_dataset.detector_dict[key]
            img = load_image(entry['file'])
            width, height = img.size

            x_coords, y_coords = Cropper.get_absolute_coords(
                entry['detections'][0]['bbox'], (height, width)
            )
            img = render_bbox(img, x_coords, y_coords)

            img.save(os.path.join(target_path, key))

        label_template = [(key, '') for key in keys]
        save_as_csv(
            rows=label_template, target=os.path.join(self.act_dir, 'active_labels.csv')
        )

        print(
            f'A selection of images is now waiting in "{self.act_dir}" for your '
            f'labeling expertise!\nRerun the program when you are done.'
        )

    def collect(self) -> None:
        """
        Collect manual labels from staging area.

        The function checks if the active labels were filled in correctly and
        completely. If collecting labels was successful, the staging area is wiped.
        """
        try:
            labels_supplied = {
                key: value
                for key, value in load_csv(
                    os.path.join(self.act_dir, 'active_labels.csv')
                )
            }
        except IOError:
            'There is a problem with your label file.'
            'Make sure you have supplied a label for every entry.'

        # Check whether label type is valid
        set_labels_supplied = set(labels_supplied.values())
        if not all([isinstance(item, str) for item in set_labels_supplied]):
            raise ValueError('Supplied labels must be of type "integer".')
        set_labels_train = set(self.train_dataset.label_dict.values())
        unknown_labels = set.difference(set_labels_supplied, set_labels_train)
        if len(unknown_labels) > 0:
            print(
                f'Please note: Your supplied labels contain classes "{unknown_labels}" '
                f'which have so far not been part of the training data.'
            )

        # Update datasets
        # TODO think about stratification here (at least by class)
        n_t = len(self.train_dataset)
        n_v = len(self.val_dataset)
        train_keys, _, val_keys = do_train_split(
            label_file_path=os.path.join(self.act_dir, 'active_labels.csv'),
            splits=(n_t / (n_t + n_v), 0, n_v / (n_t + n_v)),
            strategy='random',
            random_state=self.random_state,
        )
        self.train_dataset = update_dataset(
            dataset=self.train_dataset,
            new_label_dict={
                k: v for k, v in labels_supplied.items() if k in train_keys
            },
        )
        self.val_dataset = update_dataset(
            dataset=self.train_dataset,
            new_label_dict={k: v for k, v in labels_supplied.items() if k in val_keys},
        )
        self.pool_dataset = update_dataset(
            dataset=self.pool_dataset,
            keys=[k for k in self.pool_dataset.keys if k not in labels_supplied.keys()],
        )

        # Wipe staging area
        for f in os.listdir(os.path.join(self.act_dir, 'images')):
            os.remove(os.path.join(self.act_dir, 'images', f))
        os.remove(os.path.join(self.act_dir, 'active_labels.csv'))

    def fit(self) -> None:
        """Fit the model with active data."""
        self.trainer.load_model(os.path.join(self.model_dir, 'untrained_model.h5'))
        # TODO: fit trainer

    def evaluate(self) -> None:
        """Evaluate the model on the eval dataset."""
        pass

    def predict(self, keys: List[str]) -> np.ndarray:
        """Obtain predictions for a list of keys."""
        pass
