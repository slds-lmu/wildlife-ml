"""Acquisitor classes for Active Learning."""
import random
from abc import ABC, abstractmethod
from typing import (
    Dict,
    List,
    Optional,
)

import numpy as np


class BaseAcquisitor(ABC):
    """Base class for acquisition."""

    def __init__(self, top_k: int = 10, random_state: Optional[int] = None) -> None:
        """Initialize BaseAcquisitor."""
        self.top_k = top_k
        self.random_state = random_state

    @abstractmethod
    def _run_acquisition(self, predictions: Dict[str, float]) -> List[str]:
        """Return the top candidates for AL loop."""
        pass

    def __call__(self, predictions: Dict[str, float]) -> List[str]:
        """Return the top candidates for AL loop."""
        return self._run_acquisition(predictions)

    @abstractmethod
    def __str__(self) -> str:
        """Print object name."""
        pass

    @staticmethod
    # TODO rewrite for lists of keys
    def get_top_k_indices(x: np.ndarray, k: int = 1) -> np.ndarray:
        """Return top k indices (or all, if x has fewer than k elements)."""
        assert len(x.shape) == 1
        m = min(k, x.shape[0])
        return np.argpartition(x, -m)[-m:]


class RandomAcquisitor(BaseAcquisitor):
    """Random acquisition."""

    def _run_acquisition(self, predictions: Dict[str, float]) -> List[str]:
        if self.random_state is not None:
            random.seed(self.random_state)
        return random.sample(list(predictions.keys()), self.top_k)

    def __str__(self):
        """Print object name."""
        return 'random'


class AcquisitorFactory:
    """Factory for creating acquisitors."""

    @staticmethod
    def get(
        acquisitor_name: str, top_k: int = 10, random_state: Optional[int] = None
    ) -> BaseAcquisitor:
        """Create acquisitor."""
        if acquisitor_name == 'random':
            return RandomAcquisitor(top_k=top_k, random_state=random_state)
        else:
            raise ValueError(
                'Acquisitor with name "{}" is unknown.'.format(acquisitor_name)
            )
