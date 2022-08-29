"""Evaluating model outputs."""
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras import Model

from wildlifeml.data import (
    BBoxMapper,
    WildlifeDataset,
    subset_dataset,
)
from wildlifeml.utils.datasets import map_bbox_to_img, map_preds_to_img
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
        batch_size: int = 64,
    ) -> None:
        """Initialize evaluator object."""
        self.detector_dict = load_json(detector_file_path)
        self.label_dict = {key: float(val) for key, val in load_csv(label_file_path)}
        self.batch_size = batch_size
        self.num_classes = num_classes

        # Index what images are contained in the eval dataset
        self.dataset_imgs = set([map_bbox_to_img(k) for k in dataset.keys])
        # Get mapping of img -> bboxs for dataset
        self.bbox_map = BBoxMapper(detector_file_path).get_keymap()
        # Remove keys from bbox_map that are not in the eval dataset
        removable_keys = set(self.bbox_map) - self.dataset_imgs
        for k in removable_keys:
            del self.bbox_map[k]

        # Get keys that have no detected bbox from the MD
        self.empty_keys = [
            k for k in dataset.keys if self.detector_dict[k]['category'] == -1
        ]

        # Remove keys from dataset where MD detects no bbox
        self.bbox_keys = list(set(dataset.keys) - set(self.empty_keys))

        # Only register samples that are not filtered by MD
        self.dataset = subset_dataset(dataset, keys=self.bbox_keys)
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
        # voting via confidence and softmax scores in the evaluate phase. Consider
        # images that are filtered out by the MD with confidence 1.0.
        self.empty_pred_arr = np.zeros(
            len(self.empty_keys), num_classes, dtype=np.float
        )
        self.empty_pred_arr[:, self.empty_class_id] = 1.0

    def evaluate(
        self,
        model: Model,
        verbose: bool = True,
    ) -> Dict:
        """Obtain metrics for a supplied model."""
        # Get predictions for bboxs
        preds = model.predict(self.dataset)

        # Above predictions are on bbox level, but image level prediction is desired.
        # For this every prediction is reweighted with the MD confidence score.
        # All predictions for one image are summed up and then argmaxed to obtain the
        # final prediction.

        # Aggregate empty and bbox predictions
        all_keys_idx = {k: i for i, k in enumerate(self.empty_keys + self.bbox_keys)}
        all_preds = np.concatenate([self.empty_pred_arr, preds])

        # Compute majority voting predictions on image level
        _, y_preds = map_preds_to_img(
            preds_bboxes=all_preds,
            mapping_dict=self.bbox_map,
            detector_dict=self.detector_dict,
        )
        y_trues = [self.label_dict[k] for k in all_keys_idx]

        # Compute metrics on final predictions
        metrics = Evaluator.compute_metrics(np.asarray(y_trues), np.asarray(y_preds))

        # Lastly, add keras metrics
        keras_metrics = {zip(model.metrics_names, model.evaluate(self.dataset))}
        metrics.update({'keras_metrics': keras_metrics})

        if verbose:
            print(
                'ACC: {:.3f}\t|\tPREC: {:.3f}\t|\tREC: {:.3f}\t|\tF1 : {:.3f}'.format(
                    metrics['acc'], metrics['prec'], metrics['rec'], metrics['f1']
                )
            )

        return metrics

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
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

        return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}
