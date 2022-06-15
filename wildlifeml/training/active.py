"""Classes and functions for Active Learning."""
import os
import random
from typing import (
    Dict,
    List,
    Optional,
)

import numpy as np

from wildlifeml.data import WildlifeDataset
from wildlifeml.preprocessing.cropping import Cropper
from wildlifeml.training.acquisitor import AcquisitorFactory
from wildlifeml.utils.io import (
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
        target_dataset: WildlifeDataset,
        image_directory: str,
        al_batch_size: int = 10,
        active_directory: str = 'active-wildlife',
        acquisitor_name: str = 'random',
        start_fresh: bool = True,
        eval_dataset: Optional[WildlifeDataset] = None,
        state_cache: str = '.activecache.json',
        random_state: Optional[int] = None,
    ) -> None:
        """Instantiate an ActiveLearner object."""
        self.target_dataset = target_dataset
        self.img_dir = image_directory
        self.act_dir = active_directory

        self.acquisitor = AcquisitorFactory.get(
            acquisitor_name, top_k=al_batch_size, random_state=random_state
        )
        self.eval_dataset = eval_dataset
        self.al_batch_size = al_batch_size
        self.random_state = random_state

        self.state_cache_file = state_cache
        self.do_fresh_start = start_fresh

        # Serves as storage for all active keys and labels.
        self.active_labels: Dict[str, float] = {}

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
        # SELECT NEW CANDIDATES
        # ------------------------------------------------------------------------------
        candidate_keys = self.target_dataset.keys
        # TODO: Filter self.active_labels from full target dataset.
        preds = self.predict(candidate_keys)
        staging_keys = self.acquisitor(candidate_keys, preds)
        self.fill_active_stage(staging_keys)
        self.save_state()

    def initialize(self) -> None:
        """Initialize AL run as fresh start."""
        if os.path.exists(self.state_cache_file):
            print(
                '---> Found active learning state {}, but you also specified to '
                'start a new Active Learning run. The file will be deleted and your '
                'progress is lost.',
                format(self.state_cache_file),
            )
            print('Do you want to continue? [y / n]')
            if 'y' not in input().lower():
                print('Active Learning setup is aborted.')
                exit()

        os.makedirs(self.act_dir, exist_ok=True)
        os.makedirs(os.path.join(self.act_dir, 'images'), exist_ok=True)

        print(
            'For this fresh start, {} images are randomly chosen from your unlabeled '
            'dataset.'.format(self.al_batch_size)
        )
        all_keys = self.target_dataset.keys
        if self.random_state is not None:
            random.seed(self.random_state)
        staging_keys = random.sample(all_keys, self.al_batch_size)

        # Move initial data and exit.
        self.fill_active_stage(staging_keys)
        self.save_state()

    def load_state(self) -> None:
        """Initialize active learner from state file."""
        try:
            state = load_json(self.state_cache_file)

            self.img_dir = state['image_directory']
            self.al_batch_size = state['al_batch_size']
            self.act_dir = state['active_directory']
            self.random_state = state['random_state']
            self.active_keys = state['active_keys']

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
            'active_keys': self.active_keys,
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
            entry = self.target_dataset.detector_dict[key]
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
            'A selection of images is now waiting in "{}" for your labeling '
            'expertise!\nRerun the program, when you are done'.format(self.act_dir)
        )

    def collect(self) -> None:
        """
        Collect manual labels from staging area.

        The function checks if the active labels were filled in correctly and
        completely. If collecting labels was successful, the staging area is wiped.
        """
        pass

    def fit(self) -> None:
        """Fit the model with active data."""
        pass

    def predict(self, keys: List[str]) -> np.ndarray:
        """Obtain predictions for a list of keys."""
        pass
