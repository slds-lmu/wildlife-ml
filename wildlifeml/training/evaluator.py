"""Evaluating model outputs."""
from typing import (
    Dict,
    List,
    Optional,
)

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from wildlifeml.data import (
    BBoxMapper,
    WildlifeDataset,
    subset_dataset,
)
from wildlifeml.training.trainer import BaseTrainer
from wildlifeml.utils.datasets import (
    map_bbox_to_img,
    map_preds_to_img,
    separate_empties,
)
from wildlifeml.utils.io import load_csv, load_json


class Evaluator:
    """Evaluating model predictions on suitable metrics."""

    def __init__(
        self,
        label_file_path: str,
        detector_file_path: str,
        dataset: WildlifeDataset,
        num_classes: int,
        empty_class_id: Optional[int] = None,
        conf_threshold: Optional[float] = None,
        batch_size: int = 64,
    ) -> None:
        """Initialize evaluator object."""
        self.detector_dict = load_json(detector_file_path)
        self.label_dict = {key: float(val) for key, val in load_csv(label_file_path)}
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold

        # Index what images are contained in the eval dataset
        self.dataset_imgs = set([map_bbox_to_img(k) for k in dataset.keys])
        # Get mapping of img -> bboxs for dataset
        self.bbox_map = BBoxMapper(detector_file_path).get_keymap()
        # Remove keys from bbox_map that are not in the eval dataset
        removable_keys = set(self.bbox_map) - self.dataset_imgs
        for k in removable_keys:
            del self.bbox_map[k]

        # Get keys that have no detected bbox from the MD, with optional new threshold
        empty_keys_md, _ = separate_empties(detector_file_path, self.conf_threshold)
        self.empty_keys = list(set(dataset.keys).intersection(set(empty_keys_md)))

        # Remove keys from dataset where MD detects no bbox
        self.nonempty_keys = list(set(dataset.keys) - set(self.empty_keys))

        # Only register samples that are not filtered by MD
        self.dataset = subset_dataset(dataset, keys=self.nonempty_keys)
        self.dataset.shuffle = False

        # Dirty fix for determining which Keras class predictions
        # corresponds to the empty class label
        # If no empty class id is supplied, the maximum class index is taken as empty
        # class
        if empty_class_id is None:
            self.empty_class_id = num_classes - 1
        else:
            self.empty_class_id = empty_class_id

        # Pre-allocated array of shape (num_empty_samples, num_classes), which is 1
        # in the empty class index and zero otherwise. This is a setup for majority
        # voting via confidence and softmax scores in the evaluation phase. Consider
        # images that are filtered out by the MD with confidence 1.0.
        self.empty_pred_arr = np.zeros(
            shape=(len(self.empty_keys), num_classes), dtype=float
        )
        self.empty_pred_arr[:, self.empty_class_id] = 1.0
        self.preds = np.empty(0)
        self.preds_imgs_clf: Dict = {}
        self.preds_imgs_ppl: Dict = {}
        self.truth_imgs_clf: List = []
        self.truth_imgs_ppl: List = []

    def evaluate(self, trainer: BaseTrainer) -> None:
        """Obtain metrics for a supplied model."""
        # Get predictions for bboxs
        self.preds = trainer.predict(self.dataset)

        # Above predictions are on bbox level, but image level prediction is desired.
        # For this every prediction is reweighted with the MD confidence score.
        # All predictions for one image are summed up and then argmaxed to obtain the
        # final prediction.

        # Aggregate empty and bbox predictions
        all_preds = np.concatenate([self.empty_pred_arr, self.preds])

        # Compute majority voting predictions on image level
        self.preds_imgs_clf = map_preds_to_img(
            bbox_keys=self.nonempty_keys,
            preds=self.preds,
            mapping_dict=self.bbox_map,
            detector_dict=self.detector_dict,
            empty_class_id=self.empty_class_id,
        )
        self.preds_imgs_ppl = map_preds_to_img(
            bbox_keys=self.empty_keys + self.nonempty_keys,
            preds=all_preds,
            mapping_dict=self.bbox_map,
            detector_dict=self.detector_dict,
            empty_class_id=self.empty_class_id,
        )
        self.truth_imgs_clf = [self.label_dict[k] for k in self.preds_imgs_clf.keys()]
        self.truth_imgs_ppl = [self.label_dict[k] for k in self.preds_imgs_ppl.keys()]

    def get_details(self) -> Dict:
        """Obtain further details about predictions."""
        return {
            'keys_bbox_empty': self.empty_keys,
            'keys_bbox_nonempty': self.nonempty_keys,
            'preds_bbox_empty': self.empty_pred_arr,
            'preds_bbox_nonempty': self.preds,
            'preds_imgs_clf': self.preds_imgs_clf,
            'preds_imgs_ppl': self.preds_imgs_ppl,
            'truth_imgs_clf': self.truth_imgs_clf,
            'truth_imgs_ppl': self.truth_imgs_ppl,
        }

    def compute_metrics(self) -> Dict:
        """Compute eval metrics for predictions."""
        return self._compute_metrics(
            np.array(self.truth_imgs_ppl),
            np.array([np.argmax(v) for v in self.preds_imgs_ppl.values()]),
        )

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Compute eval metrics for predictions."""
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
        f1 = f1_score(
            y_true=y_true,
            y_pred=y_pred,
            average='macro',
            zero_division=0,
        )
        tp, tn, fp, fn = 0, 0, 0, 0
        for true, pred in zip(y_true, y_pred):
            if true == self.empty_class_id:
                if true == pred:
                    tn += 1
                else:
                    fp += 1
            else:
                if pred == self.empty_class_id:
                    fn += 1
                else:
                    tp += 1
        conf_empty = {
            'tnr': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'fnr': fn / (tp + fn) if (tp + fn) > 0 else 0.0,
            'fpr': fp / (tn + fp) if (tn + fp) > 0 else 0.0,
        }
        return {
            'acc': acc,
            'prec': prec,
            'rec': rec,
            'f1': f1,
            'conf_empty': conf_empty,
        }
