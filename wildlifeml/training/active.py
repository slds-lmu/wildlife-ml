"""Classes and functions for Active Learning."""
import os
import random
from typing import (
    Dict,
    List,
    Optional,
)

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

from wildlifeml.data import (
    WildlifeDataset,
    append_dataset,
    do_train_split,
    subset_dataset,
)
from wildlifeml.preprocessing.cropping import Cropper
from wildlifeml.training.acquisitor import AcquisitorFactory
from wildlifeml.training.trainer import WildlifeTrainer
from wildlifeml.utils.io import (
    load_csv,
    load_image,
    load_json,
    load_pickle,
    save_as_csv,
    save_as_json,
    save_as_pickle,
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
        label_file_path: str,
        al_batch_size: int = 10,
        active_directory: str = 'active-wildlife',
        acquisitor_name: str = 'random',
        start_fresh: bool = True,
        start_keys: List[str] = None,
        test_dataset: Optional[WildlifeDataset] = None,
        test_logfile_path: Optional[str] = None,
        state_cache: str = '.activecache.json',
        random_state: Optional[int] = None,
    ) -> None:
        """Instantiate an ActiveLearner object."""
        self.pool_dataset = pool_dataset
        self.dir_img = self.pool_dataset.img_dir
        self.dir_act = active_directory

        self.trainer = trainer

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.label_file_path = label_file_path

        self.acquisitor = AcquisitorFactory.get(
            acquisitor_name, top_k=al_batch_size, random_state=random_state
        )
        self.test_dataset = test_dataset
        self.test_logfile_path = test_logfile_path
        self.al_batch_size = al_batch_size
        self.random_state = random_state

        self.state_cache_file = state_cache
        self.do_fresh_start = start_fresh
        self.start_keys = start_keys

        # Serves as storage for all active keys and labels.
        self.active_labels: Dict[str, float] = {}
        # Count active learning iterations
        self.active_counter = 0

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
        preds = dict(zip(self.pool_dataset.keys, self.predict(self.pool_dataset)))
        staging_keys = self.acquisitor(preds)
        self.fill_active_stage(staging_keys)
        self.save_state()

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

        os.makedirs(self.dir_act, exist_ok=True)
        os.makedirs(os.path.join(self.dir_act, 'images'), exist_ok=True)

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

    def load_state(self) -> None:
        """Initialize active learner from state file."""
        try:
            state = load_json(self.state_cache_file)

            self.dir_img = self.pool_dataset.img_dir
            self.al_batch_size = state['al_batch_size']
            self.dir_act = state['active_directory']
            self.random_state = state['random_state']
            self.active_labels = state['active_labels']
            self.active_counter = state['active_counter']
            self.active_counter += 1

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
            'image_directory': self.dir_img,
            'al_batch_size': self.al_batch_size,
            'active_directory': self.dir_act,
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
        target_path = os.path.join(self.dir_act, 'images')

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
            rows=label_template, target=os.path.join(self.dir_act, 'active_labels.csv')
        )

        print(
            f'A selection of images is now waiting in "{self.dir_act}" for your '
            f'labeling expertise!\nRerun the program when you are done.'
        )

    def collect(self) -> None:
        """
        Collect manual labels from staging area.

        The function checks if the active labels were filled in correctly and
        completely. If collecting labels was successful, the staging area is wiped.
        """
        labels_supplied = {}
        try:
            labels_supplied.update(
                {
                    key: float(value)
                    for key, value in load_csv(
                        os.path.join(self.dir_act, 'active_labels.csv')
                    )
                }
            )
        except IOError:
            'There is a problem with your label file.'
            'Make sure you have supplied a label for every entry.'

        # Check whether label type is valid
        set_labels_supplied = set(labels_supplied.values())
        set_labels_train = set(self.train_dataset.label_dict.values())
        unknown_labels = set.difference(set_labels_supplied, set_labels_train)
        if len(unknown_labels) > 0:
            print(
                f'Please note: Your supplied labels contain classes "{unknown_labels}" '
                f'which have so far not been part of the training data.'
            )

        # Update datasets (add new instances to train and validation data in constant
        # proportion; remove them from pool dataset)
        # TODO think about stratification here (at least by class)
        n_t = len(self.train_dataset.keys)
        n_v = len(self.val_dataset.keys)
        train_keys, _, val_keys = do_train_split(
            label_file_path=os.path.join(self.dir_act, 'active_labels.csv'),
            splits=(n_t / (n_t + n_v), 0, n_v / (n_t + n_v)),
            strategy='random',
            random_state=self.random_state,
        )
        self.train_dataset = append_dataset(
            dataset=self.train_dataset,
            new_label_dict={
                k: v for k, v in labels_supplied.items() if k in train_keys
            },
        )
        self.val_dataset = append_dataset(
            dataset=self.train_dataset,
            new_label_dict={k: v for k, v in labels_supplied.items() if k in val_keys},
        )
        self.pool_dataset = subset_dataset(
            dataset=self.pool_dataset,
            keys=[k for k in self.pool_dataset.keys if k not in labels_supplied.keys()],
        )
        # Update label file
        labels_existing = {
            key: float(value) for key, value in load_csv(self.label_file_path)
        }
        labels_existing.update(labels_supplied)
        save_as_csv(
            rows=[(key, value) for key, value in labels_existing.items()],
            target=self.label_file_path,
        )

        # Wipe staging area
        for f in os.listdir(os.path.join(self.dir_act, 'images')):
            os.remove(os.path.join(self.dir_act, 'images', f))
        os.remove(os.path.join(self.dir_act, 'active_labels.csv'))

    def fit(self) -> None:
        """Fit the model with active data."""
        self.trainer.reset_model()
        self.trainer.fit(self.train_dataset, self.val_dataset)

    def evaluate(self) -> None:
        """Evaluate the model on the eval dataset."""
        logfile = {}
        if self.test_logfile_path is not None and os.path.exists(
            self.test_logfile_path
        ):
            logfile = load_pickle(self.test_logfile_path)

        print('---> Evaluating on test data')
        # TODO find out whether keras metrics are valid (seem very optimistic)
        keras_metrics = dict(
            zip(
                self.trainer.model.metrics_names,
                self.trainer.model.evaluate(self.test_dataset),
            )
        )
        if self.test_dataset is not None:
            y_true = np.array(
                [
                    value
                    for key, value in self.test_dataset.label_dict.items()
                    if key in self.test_dataset.keys
                ]
            )
            preds = self.predict(self.test_dataset)
            y_pred = np.argmax(preds, axis=1)
            acc = accuracy_score(y_true=y_true, y_pred=y_pred)
            prec = precision_score(
                y_true=y_true,
                y_pred=y_pred,
                average='macro',
                zero_division=0,
            )
            rec = recall_score(
                y_true=y_true,
                y_pred=y_pred,
                average='macro',
                zero_division=0,
            )
            custom_metrics = {
                'accuracy_skl': acc,
                'precision': prec,
                'recall': rec,
                'confusion_matrix': confusion_matrix(y_true=y_true, y_pred=y_pred),
            }
            results = dict(keras_metrics, **custom_metrics)
            print(f'accuracy: {acc:.3f}, precision: {prec:.3f}, recall: {rec:.3f}')
            if self.test_logfile_path is not None:
                logfile.update({f'iteration {self.active_counter}': results})
                save_as_pickle(logfile, self.test_logfile_path)

    def predict(self, dataset: WildlifeDataset) -> np.ndarray:
        """Obtain predictions for a list of keys."""
        return self.trainer.predict(dataset)