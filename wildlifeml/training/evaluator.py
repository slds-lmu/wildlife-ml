"""Evaluating model outputs."""
from typing import Dict, List

from tensorflow.keras import Model
from tensorflow.keras.utils import Sequence


class Evaluator:
    """Evaluating model predictions on suitable metrics."""

    def __init__(self, metrics: List) -> None:
        """Initialize evaluator object."""
        self.metrics = metrics

    @staticmethod
    def predict(
        model: Model,
        dataset: Sequence,
    ):
        """Make predictions on dataset with supplied model."""
        pass

    def compute_metrics(self, predictions: Dict[str, float]):
        """Compute eval metrics for predictions."""
        pass

    @staticmethod
    def log_metrics(self, logfile_path: str = None):
        """Write eval results to file."""
        pass
