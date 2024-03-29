"""Acquisitor classes for Active Learning."""
import heapq
import random
from abc import ABC, abstractmethod
from typing import (
    Dict,
    List,
    Optional,
)

import numpy as np
from scipy.stats import entropy

from wildlifeml.utils.datasets import do_stratified_sampling


class BaseAcquisitor(ABC):
    """Base class for acquisition."""

    def __init__(
        self,
        top_k: int = 10,
        random_state: Optional[int] = None,
        stratified: bool = False,
        meta_dict: Optional[Dict] = None,
    ) -> None:
        """Initialize BaseAcquisitor."""
        self.top_k = top_k
        self.random_state = random_state
        self.stratified = stratified
        self.meta_dict = meta_dict

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
    def get_top_k_keys(
        acq_scores: Dict[str, float],
        top_k: int,
        reverse: bool = False,
    ) -> List[str]:
        """Return keys of top k values."""
        k = min(top_k, len(acq_scores))
        if not reverse:
            list_top_k = heapq.nlargest(k, acq_scores.items(), key=lambda i: i[1])
        else:
            list_top_k = heapq.nsmallest(k, acq_scores.items(), key=lambda i: i[1])
        return [x[0] for x in list_top_k]


class RandomAcquisitor(BaseAcquisitor):
    """Random acquisition."""

    def _run_acquisition(
        self,
        predictions: Dict[str, float],
        stratified: bool = False,
        meta_dict: Optional[Dict] = None,
    ) -> List[str]:
        if self.random_state is not None:
            random.seed(self.random_state)
        if stratified and meta_dict is None:
            raise ValueError(
                'Stratified acquisition requires variable for stratification.'
            )
        if stratified:
            keys = do_stratified_sampling(
                img_keys=list(predictions.keys()),
                n_samples=self.top_k,
                meta_dict=meta_dict,
                random_state=self.random_state,
            )
        else:
            keys = random.sample(list(predictions.keys()), self.top_k)
        return keys

    def __str__(self):
        """Print object name."""
        return 'random'


class EntropyAcquisitor(BaseAcquisitor):
    """Softmax-confidence-based acquisition."""

    def _run_acquisition(self, predictions: Dict[str, float]) -> List[str]:
        entropy_dict = {
            key: entropy(np.array(value), base=2) for key, value in predictions.items()
        }
        return self.get_top_k_keys(entropy_dict, self.top_k)

    def __str__(self):
        """Print object name."""
        return 'entropy'


class BreakingTiesAcquisitor(BaseAcquisitor):
    """Acquisition based on discrimination between top-2 classes."""

    def _run_acquisition(self, predictions: Dict[str, float]) -> List[str]:
        ties_dict = {
            key: self._break_ties(np.array(value)) for key, value in predictions.items()
        }
        return self.get_top_k_keys(ties_dict, self.top_k, reverse=True)

    def __str__(self):
        """Print object name."""
        return 'breakingties'

    @staticmethod
    def _break_ties(x: np.ndarray) -> float:
        """Compute difference between top-2-largest values in vector."""
        top_two: List[float] = heapq.nlargest(2, x)
        return abs(top_two[0] - top_two[1])


class AcquisitorFactory:
    """Factory for creating acquisitors."""

    @staticmethod
    def get(
        acquisitor_name: str,
        top_k: int = 10,
        random_state: Optional[int] = None,
        stratified: bool = False,
        meta_dict: Optional[Dict] = None,
    ) -> BaseAcquisitor:
        """Create acquisitor."""
        if acquisitor_name == 'random':
            return RandomAcquisitor(
                top_k=top_k,
                random_state=random_state,
                stratified=stratified,
                meta_dict=meta_dict,
            )
        elif acquisitor_name == 'entropy':
            return EntropyAcquisitor(top_k=top_k)
        elif BreakingTiesAcquisitor == 'breakingties':
            return BreakingTiesAcquisitor(top_k=top_k)
        else:
            raise ValueError(f'Acquisitor with name "{acquisitor_name}" is unknown.')
