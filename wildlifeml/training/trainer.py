"""Classes for managing training."""
from abc import ABC, abstractmethod
from typing import (
    Any,
    Final,
    List,
    Optional,
    Tuple,
)

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence

from wildlifeml.training.models import ModelFactory

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

TUNABLE: Final[List[str]] = [
    'batch_size',
    'transfer_learning_rate',
    'finetune_learning_rate',
    'backbone',
]


class BaseTrainer(ABC):
    """Base class for trainer objects."""

    @abstractmethod
    def fit(self, train_dataset: Sequence, val_dataset: Sequence) -> Model:
        """Fit the model on the provided dataset."""
        pass

    @abstractmethod
    def get_model(self) -> Model:
        """Return the model instance."""
        pass

    @abstractmethod
    def compile_model(self) -> None:
        """Compile the model for evaluation."""
        pass

    @abstractmethod
    def reset_model(self) -> None:
        """Set model to initial state as obtained from model factory."""
        pass

    @abstractmethod
    def save_model(self, file_path: str) -> None:
        """Save a model checkpoint."""
        pass

    @abstractmethod
    def load_model(self, file_path: str) -> None:
        """Load a model from a checkpoint."""
        pass

    @abstractmethod
    def predict(self, dataset: Sequence) -> np.ndarray:
        """Make predictions according to trained model."""
        pass

    @abstractmethod
    def get_num_classes(self):
        """Return number of classes."""
        pass


class WildlifeTrainer(BaseTrainer):
    """Trainer for assisting with fitting neural networks."""

    def __init__(
        self,
        batch_size: int,
        loss_func: Any,
        num_classes: int,
        transfer_epochs: int,
        finetune_epochs: int,
        transfer_optimizer: Any,
        finetune_optimizer: Any,
        finetune_layers: int,
        model_backbone: str = 'resnet50',
        transfer_callbacks: Optional[List] = None,
        finetune_callbacks: Optional[List] = None,
        num_workers: int = 0,
        eval_metrics: Optional[List] = None,
        pretraining_checkpoint: Optional[str] = None,
        input_shape: Optional[Tuple] = (1, 224, 224, 3),
    ) -> None:
        """Initialize trainer object."""
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model_backbone = model_backbone
        self.pretraining_checkpoint = pretraining_checkpoint
        self.model = Sequential()
        self.reset_model()

        self.transfer_optimizer = transfer_optimizer
        self.finetune_optimizer = finetune_optimizer
        self.loss_func = loss_func

        self.finetune_layers = finetune_layers
        self.finetune_epochs = finetune_epochs
        self.transfer_epochs = transfer_epochs
        self.transfer_callbacks = transfer_callbacks
        self.finetune_callbacks = finetune_callbacks
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.eval_metrics = eval_metrics

    def get_num_classes(self) -> int:
        """Return number of classes."""
        return self.num_classes

    def fit(
        self, train_dataset: Sequence, val_dataset: Optional[Sequence] = None
    ) -> Model:
        """Fit the model on the provided dataset."""
        if self.transfer_epochs > 0:
            print('---> Compiling model')
            self.model.compile(
                optimizer=self.transfer_optimizer,
                loss=self.loss_func,
                metrics=self.eval_metrics,
            )
            if self.pretraining_checkpoint is not None:
                # first line necessary to get architecture (not stored in .hd5)
                self.model(np.zeros(self.input_shape))
                self.load_model(self.pretraining_checkpoint)

            print('---> Starting transfer learning')
            for layer in self.model.get_layer(self.model_backbone).layers:
                layer.trainable = False

            self.model.fit(
                x=train_dataset,
                validation_data=val_dataset,
                batch_size=self.batch_size,
                epochs=self.transfer_epochs,
                callbacks=self.transfer_callbacks,
                workers=self.num_workers,
                use_multiprocessing=self.num_workers > 0,
                shuffle=False,
            )

        if self.finetune_epochs > 0 and self.finetune_layers > 0:
            print(f'---> Unfreezing last {self.finetune_layers} layers')
            bbone_layers = self.model.get_layer(self.model_backbone).layers
            for layer in bbone_layers[: -self.finetune_layers]:
                layer.trainable = False
            for layer in bbone_layers[-self.finetune_layers :]:
                layer.trainable = True

            print('---> Compiling model')
            self.model.compile(
                optimizer=self.finetune_optimizer,
                loss=self.loss_func,
                metrics=self.eval_metrics,
                # run_eagerly=True,
            )

            print('---> Starting fine tuning')
            self.model.fit(
                x=train_dataset,
                validation_data=val_dataset,
                batch_size=self.batch_size,
                epochs=self.finetune_epochs,
                callbacks=self.finetune_callbacks,
                workers=self.num_workers,
                use_multiprocessing=self.num_workers > 0,
                shuffle=False,
            )

        return self.model

    def get_model(self) -> Model:
        """Return the model instance."""
        return self.model

    def compile_model(self) -> None:
        """Compile model for evaluation."""
        self.model.compile(
            optimizer=self.transfer_optimizer,
            loss=self.loss_func,
            metrics=self.eval_metrics,
            run_eagerly=True,
        )

    def reset_model(self) -> None:
        """Set model to initial state as obtained from model factory."""
        self.model = ModelFactory.get(
            model_id=self.model_backbone, num_classes=self.num_classes
        )

    def save_model(self, file_path: str) -> None:
        """Save a model checkpoint."""
        self.model.save(file_path)

    def save_model_weights(self, file_path: str) -> None:
        """Save model weights."""
        self.model.save_weights(file_path)

    def load_model(self, file_path: str) -> None:
        """Load a model from a checkpoint."""
        self.model.load_weights(file_path)

    def predict(self, dataset: Sequence) -> np.ndarray:
        """Make predictions according to trained model."""
        return self.model.predict(dataset)
