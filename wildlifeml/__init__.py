"""Modules for the 'wildlifeml' package."""
from wildlifeml.preprocessing.megadetector import MegaDetector
from wildlifeml.training.active import ActiveLearner
from wildlifeml.training.trainer import WildlifeTrainer

__all__ = ['MegaDetector', 'WildlifeTrainer', 'ActiveLearner']
