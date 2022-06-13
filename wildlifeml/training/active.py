"""Classes and functions for Active Learning."""
import os
from typing import List, Optional

from wildlifeml.data import WildlifeDataset
from wildlifeml.training.acquisitor import AcquisitorFactory


class ActiveLearner:
    """Interface for Active Learning on wildlife data."""

    def __init__(
        self,
        keys: List[str],
        image_directory: str,
        al_batch_size: int = 10,
        active_directory: str = 'active-wildlife',
        acquisitor_name: str = 'random',
        start_fresh: bool = True,
        eval_dataset: Optional[WildlifeDataset] = None,
        state_cache: str = '.activecache.yml',
        random_state: Optional[int] = None,
    ) -> None:
        """Instantiate an ActiveLearner object."""
        self.keys = keys
        self.img_dir = image_directory
        self.act_dir = active_directory

        self.acquisitor = AcquisitorFactory.get(
            acquisitor_name, top_k=al_batch_size, random_state=random_state
        )
        self.eval_dataset = eval_dataset

        self.state_cache_file = state_cache
        self.do_fresh_start = start_fresh

    def initialize(self) -> None:
        """Initialize AL run as fresh start."""
        os.makedirs(self.act_dir, exist_ok=True)

    def run(self) -> None:
        """Trigger Active Learning process."""
        pass
