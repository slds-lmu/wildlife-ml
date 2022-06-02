"""Classes for accessing data."""
from typing import Tuple

import numpy as np
from tensorflow.keras.utils import Sequence


class WildlifeDataset(Sequence):
    """Dataset object for handling wildlife datasets with bounding boxes."""

    def __init__(self, batch_size: int, shuffle: bool = True) -> None:
        """Initialize a WildlifeDataset object."""
        self.batch_size = batch_size
        self.shuffle = shuffle

    def on_epoch_end(self) -> None:
        """Execute after every epoch in the keras `.fit()` method."""
        if self.shuffle:
            pass

    def __len__(self) -> int:
        """Return the number of batches in the dataset."""
        pass

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return a batch with training data and labels."""
        pass
