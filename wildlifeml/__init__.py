"""Modules for the 'wildlifeml' package."""
from wildlifeml.data import WildlifeDataset
from wildlifeml.preprocessing.megadetector import MegaDetector
from wildlifeml.training.active import ActiveLearner

__all__ = ['MegaDetector', 'ActiveLearner', 'WildlifeDataset']
