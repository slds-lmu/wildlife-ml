"""Custom metrics suitable for Sparse Categorical computations."""
import warnings
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras


class BaseMetric(ABC):
    """Base class for deriving metrics."""

    def __init__(self, reduction: Optional[str] = 'macro') -> None:
        """Initialize a BaseMetric object."""
        self.reduction = reduction

    @abstractmethod
    @property
    def name(self) -> str:
        """Get name of the metric."""
        pass

    @abstractmethod
    def _compute(self, tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
        """Compute metric."""
        pass

    @staticmethod
    def _catch_nans(arr: np.ndarray) -> np.ndarray:
        """Catch NaN in metric functions."""
        return np.nan_to_num(arr)

    def compute_from_confusion(
        self, tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Reduce batched metric with the predefined strategy."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            if self.reduction is None:
                metric = self._compute(tp, fp, fn)
            elif self.reduction == 'micro':
                metric = self._compute(tp.sum(), fp.sum(), fn.sum())
            elif self.reduction == 'macro':
                metric = self._compute(tp, fp, fn).mean()
            elif self.reduction == 'weighted':
                metric = (self._compute(tp, fp, fn) * weights).sum()
            return BaseMetric._catch_nans(metric)

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute metric from predictions."""
        conf = tf.math.confusion_matrix(
            y_true, tf.math.argmax(y_pred, -1), num_classes=y_pred.shape[1]
        ).numpy()
        tp = np.diagonal(conf)
        fp = conf.sum(0) - tp
        fn = conf.sum(1) - tp
        weights = conf.sum(1) / conf.sum()

        return self.compute_from_confusion(tp, fp, fn, weights)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Trigger forward pass."""
        return self.forward(y_true, y_pred)


class RecallMetric(BaseMetric):
    """Class for computing Recall."""

    @property
    def name(self) -> str:
        """Get name of the metric."""
        return 'recall'

    def _compute(self, tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
        """Compute metric."""
        return tp / (tp + fn)


class PrecisionMetric(BaseMetric):
    """Class for computing Precision."""

    @property
    def name(self) -> str:
        """Get name of the metric."""
        return 'precision'

    def _compute(self, tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
        """Compute metric."""
        return tp / (tp + fp)


class F1Metric(BaseMetric):
    """Class for computing F1 score."""

    @property
    def name(self) -> str:
        """Get name of the metric."""
        return 'f1'

    def _compute(self, tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
        """Compute metric."""
        return 2 * tp / (2 * tp + fp + fn)


class SparseCategoricalRecall(keras.metrics.Metric):
    """Keras metric for computing sparse recall."""

    def __init__(
        self,
        name: str = 'sparse_categorical_recall',
        reduction: Optional[str] = 'macro',
        **kwargs
    ) -> None:
        """Instantiate a SparseCategoricalRecall object."""
        super().__init__(name=name, **kwargs)
        self.computer = RecallMetric(reduction)
        self.reduction = reduction

    def update_state(self, y_true, y_pred, sample_weight: Any = None) -> None:
        """Update the metric state."""
        self.value = self.computer(y_true, y_pred)

    def result(self) -> np.ndarray:
        """Get result."""
        return self.value


class SparseCategoricalPrecision(keras.metrics.Metric):
    """Keras metric for computing sparse precision."""

    def __init__(
        self,
        name: str = 'sparse_categorical_precision',
        reduction: Optional[str] = 'macro',
        **kwargs
    ) -> None:
        """Instantiate a SparseCategoricalRecall object."""
        super().__init__(name=name, **kwargs)
        self.computer = PrecisionMetric(reduction)
        self.reduction = reduction

    def update_state(self, y_true, y_pred, sample_weight: Any = None) -> None:
        """Update the metric state."""
        self.value = self.computer(y_true, y_pred)

    def result(self) -> np.ndarray:
        """Get result."""
        return self.value


class SparseCategoricalF1(keras.metrics.Metric):
    """Keras metric for computing sparse f1 score."""

    def __init__(
        self,
        name: str = 'sparse_categorical_f1',
        reduction: Optional[str] = 'macro',
        **kwargs
    ) -> None:
        """Instantiate a SparseCategoricalF1 object."""
        super().__init__(name=name, **kwargs)
        self.computer = F1Metric(reduction)
        self.reduction = reduction

    def update_state(self, y_true, y_pred, sample_weight: Any = None) -> None:
        """Update the metric state."""
        self.value = self.computer(y_true, y_pred)

    def result(self) -> np.ndarray:
        """Get result."""
        return self.value
