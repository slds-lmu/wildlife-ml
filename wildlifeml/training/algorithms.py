"""Classes and functions for creating search and schedule algorithms."""

from typing import Callable, Final

from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

AVAILABLE_ALGORITHMS: Final = {
    'hyperoptsearch': HyperOptSearch,
    'ashascheduler': ASHAScheduler,
}


class AlgorithmFactory:
    """Factory for creating RayTune algorithm objects."""

    @staticmethod
    def get(alg_id: str) -> Callable:
        """Return an algorithm object from an identifier."""
        return AVAILABLE_ALGORITHMS.get(alg_id)  # type: ignore

    def __call__(self, alg_id: str) -> Callable:
        """Return an algorithm object from an identifier."""
        return AlgorithmFactory.get(alg_id)
