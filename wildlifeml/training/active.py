"""Classes and functions for Active Learning."""
import os
import random
from copy import deepcopy
from typing import (
    Dict,
    List,
    Optional,
)

from tensorflow.keras import Model

from wildlifeml.data import WildlifeDataset, subset_dataset
from wildlifeml.preprocessing.cropping import Cropper
from wildlifeml.training.acquisitor import AcquisitorFactory
from wildlifeml.training.evaluator import Evaluator
from wildlifeml.training.trainer import BaseTrainer, WildlifeTrainer
from wildlifeml.utils.datasets import (
    do_stratified_splitting,
    map_bbox_to_img,
    map_preds_to_img,
    render_bbox,
)
from wildlifeml.utils.io import (
    load_csv,
    load_image,
    load_json,
    save_as_csv,
    save_as_json,
    save_as_pickle,
)
from wildlifeml.utils.misc import flatten_list


class ActiveLearner:
    """Interface for Active Learning on wildlife data."""

    def __init__(
        self,
        trainer: BaseTrainer,
        pool_dataset: WildlifeDataset,
        label_file_path: str,
        conf_threshold: float,
        empty_class_id: int,
        al_batch_size: int = 10,
        active_directory: str = 'active-wildlife',
        acquisitor_name: str = 'random',
        start_fresh: bool = True,
        start_keys: Optional[List[str]] = None,
        train_size: float = 0.7,
        test_dataset: Optional[WildlifeDataset] = None,
        test_logfile_path: Optional[str] = None,
        acq_logfile_path: Optional[str] = None,
        meta_dict: Optional[Dict] = None,
        state_cache: str = '.activecache.json',
        random_state: Optional[int] = None,
    ) -> None:
        """Instantiate an ActiveLearner object."""
        self.pool_dataset = pool_dataset
        self.unlabeled_dataset = deepcopy(self.pool_dataset)
        self.labeled_dataset = subset_dataset(self.pool_dataset, keys=[])
        self.conf_threshold = conf_threshold

        self.dir_img = self.pool_dataset.img_dir
        self.dir_act = active_directory

        self.trainer = trainer
        self.label_file_path = label_file_path
        self.train_size = train_size

        self.test_dataset = test_dataset
        self.test_logfile_path = test_logfile_path
        self.acq_logfile_path = acq_logfile_path
        self.empty_class_id = empty_class_id

        self.meta_dict = meta_dict
        self.acquisitor = AcquisitorFactory.get(
            acquisitor_name,
            top_k=al_batch_size,
            random_state=random_state,
            stratified=True if self.meta_dict is not None else False,
            meta_dict=self.meta_dict,
        )
        self.al_batch_size = al_batch_size
        self.random_state = random_state
        self.state_cache_file = state_cache
        self.do_fresh_start = start_fresh
        self.start_keys = start_keys
        self.train_size = train_size

        # Serves as storage for all active keys and labels.
        self.active_labels: Dict[str, float] = {}
        # Count active learning iterations
        self.active_counter = 0
        # Set up evaluator
        if test_dataset is not None:
            if test_dataset.label_file_path is None:
                raise ValueError('Test dataset must have label file.')
            else:
                self.evaluator = Evaluator(
                    label_file_path=test_dataset.label_file_path,
                    detector_file_path=test_dataset.detector_file_path,
                    dataset=test_dataset,
                    empty_class_id=self.empty_class_id,
                    num_classes=trainer.get_num_classes(),
                    conf_threshold=self.conf_threshold,
                )

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
        preds = self.predict_img(
            dataset=self.unlabeled_dataset,
            detector_file_path=self.unlabeled_dataset.detector_file_path,
        )
        staging_keys = self.acquisitor(preds)
        self.fill_active_stage(staging_keys)
        self.save_state()

        if self.acq_logfile_path is not None:
            log_acq: Dict = {}
            if os.path.exists(self.acq_logfile_path):
                log_acq.update(load_json(self.acq_logfile_path))
            log_acq.update(
                {
                    f'iteration {self.active_counter + 1}': {
                        'acq_predictions': {
                            k: list(v.round(5))
                            for k, v in preds.items()
                            if k in staging_keys
                        }
                    }
                }
            )
            save_as_json(log_acq, self.acq_logfile_path)

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
            else:
                os.remove(self.state_cache_file)

        os.makedirs(self.dir_act, exist_ok=True)
        os.makedirs(os.path.join(self.dir_act, 'images'), exist_ok=True)

        if self.start_keys is None:
            print(
                f'For this fresh start, {self.al_batch_size} images are randomly '
                f'chosen from your unlabeled dataset.'
            )
            all_keys = self.unlabeled_dataset.keys
            if self.random_state is not None:
                random.seed(self.random_state)
            staging_keys = random.sample(all_keys, self.al_batch_size)
        else:
            staging_keys = self.start_keys
        staging_keys = list(set([map_bbox_to_img(k) for k in staging_keys]))

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

        # Save images with bboxes highlighted for user to label
        for key in keys:
            bbox_keys = self.pool_dataset.mapping_dict[key]
            first_entry = self.pool_dataset.detector_dict[bbox_keys[0]]
            img = load_image(first_entry['file'])
            width, height = img.size
            x_coords, y_coords = [], []
            breakpoint()
            for bkey in bbox_keys:
                x, y = Cropper.get_absolute_coords(
                    self.pool_dataset.detector_dict[bkey].get('bbox'), (height, width)
                )
                x_coords.append(x)
                y_coords.append(y)
            img = render_bbox(img, x_coords, y_coords)
            img.save(os.path.join(target_path, key))

        # Save list for user to fill in labels
        label_template = [(key, '') for key in keys]
        save_as_csv(
            rows=label_template, target=os.path.join(self.dir_act, 'active_labels.csv')
        )

        print(
            f'A selection of images is now waiting in "{self.dir_act}" for your '
            f'labeling expertise! \nPlease provide class labels in integer format.'
            f'\n Special cases: please use the label label "-1" for images with more '
            f'than one animal species present.'
            f'\nRerun the program when you are done.'
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
            print(
                'There is a problem with your label file.'
                'Make sure you have supplied a label for every entry.'
            )

        # Check whether label type is valid
        set_labels_supplied = set(labels_supplied.values())
        set_labels_train = set(self.active_labels.values())
        unknown_labels = set.difference(set_labels_supplied, set_labels_train)
        if len(unknown_labels) > 0:
            print(
                f'Please note: your supplied labels contain classes "{unknown_labels}" '
                f'which have so far not been part of the training data.'
            )

        # Update acquisition log
        if self.acq_logfile_path is not None:
            log_acq: Dict = {}
            if os.path.exists(self.acq_logfile_path):
                log_acq = load_json(self.acq_logfile_path)
            if log_acq.get(f'iteration {self.active_counter}') is not None:
                log_acq[f'iteration {self.active_counter}'].update(
                    {'acq_true_labels': {k: v for k, v in labels_supplied.items()}}
                )
            else:
                log_acq.update(
                    {
                        f'iteration {self.active_counter}': {
                            'acq_true_labels': {
                                k: v for k, v in labels_supplied.items()
                            }
                        }
                    }
                )
            save_as_json(log_acq, self.acq_logfile_path)

        # Update label dict and file (NB: labels_existing might contain data outside the
        # training procedure and is updated with all new information; the labels learned
        # during training are stored in active_labels)
        self.active_labels.update(labels_supplied)
        if os.path.exists(self.label_file_path):
            labels_existing = {
                key: float(value) for key, value in load_csv(self.label_file_path)
            }
        else:
            labels_existing = {}
        labels_existing.update(labels_supplied)
        save_as_csv(
            rows=[(key, value) for key, value in labels_existing.items()],
            target=self.label_file_path,
        )

        # Update datasets, omitting mixed-class imgs for training
        img_keys_labeled = [k for k, v in self.active_labels.items() if v != -2]
        bbox_keys_labeled = flatten_list(
            [self.pool_dataset.mapping_dict[k] for k in img_keys_labeled]
        )
        bbox_keys_unlabeled = list(
            set(self.unlabeled_dataset.keys) - set(bbox_keys_labeled)
        )
        self.labeled_dataset = subset_dataset(self.pool_dataset, bbox_keys_labeled)
        self.labeled_dataset.label_dict = self.active_labels
        self.labeled_dataset.is_supervised = True
        self.unlabeled_dataset = subset_dataset(self.pool_dataset, bbox_keys_unlabeled)

        # Wipe staging area
        for f in os.listdir(os.path.join(self.dir_act, 'images')):
            os.remove(os.path.join(self.dir_act, 'images', f))
        os.remove(os.path.join(self.dir_act, 'active_labels.csv'))

    def fit(self) -> None:
        """Fit the model with active data."""
        # Get new train and val sets
        meta_dict = self.meta_dict if self.meta_dict is not None else self.active_labels
        keys_train, _, keys_val = do_stratified_splitting(
            img_keys=list(self.active_labels.keys()),
            splits=(self.train_size, 0.0, 1 - self.train_size),
            random_state=self.random_state,
            meta_dict=meta_dict,
        )
        train_dataset = subset_dataset(
            self.labeled_dataset,
            flatten_list([self.labeled_dataset.mapping_dict[k] for k in keys_train]),
        )
        val_dataset = subset_dataset(
            self.labeled_dataset,
            flatten_list([self.labeled_dataset.mapping_dict[k] for k in keys_val]),
        )
        val_dataset.shuffle = False
        val_dataset.augmentation = None

        # Train model
        self.trainer.reset_model()
        self.trainer.fit(train_dataset, val_dataset)

    def evaluate(self) -> None:
        """Evaluate the model on the eval dataset."""
        if self.test_dataset is None:
            print('No test dataset was specified. Evaluation is skipped.')
            return
        if self.test_logfile_path is not None:
            self.evaluator.evaluate(self.trainer)
            details = self.evaluator.get_details()
            filename = os.path.join(
                self.test_logfile_path, f'results_iteration_{self.active_counter}.pkl'
            )
            # Remove old log file
            if os.path.exists(filename):
                os.remove(filename)
            save_as_pickle(details, filename)

    def predict_bbox(self, dataset: WildlifeDataset) -> Dict:
        """Obtain bbox-level predictions."""
        return dict(zip(dataset.keys, self.trainer.predict(dataset)))

    def predict_img(
        self,
        dataset: WildlifeDataset,
        detector_file_path: str,
    ) -> Dict:
        """Obtain img-level predictions."""
        preds_bboxes = self.trainer.predict(dataset)
        detector_dict = load_json(detector_file_path)
        preds_imgs = map_preds_to_img(
            preds=preds_bboxes,
            bbox_keys=dataset.keys,
            detector_dict=detector_dict,
            empty_class_id=self.empty_class_id,
        )
        return preds_imgs

    def get_model(self) -> Model:
        """Return current model instance."""
        return self.trainer.get_model()

    def set_trainer(self, trainer: WildlifeTrainer):
        """Reset trainer object."""
        self.trainer = trainer
