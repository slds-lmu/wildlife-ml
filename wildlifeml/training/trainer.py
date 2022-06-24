"""Classes for managing training."""
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.utils import Sequence

from wildlifeml.training.models import ModelFactory


class WildlifeTrainer:
    """Trainer object for assisting with fitting neural networks."""

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
    ) -> None:
        """Initialize trainer object."""
        self.num_classes = num_classes
        self.model_backbone = model_backbone
        self.model, self.preproc_func = ModelFactory.get(
            model_id=model_backbone, num_classes=num_classes
        )

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

    def fit(self, train_dataset: Sequence, val_dataset: Sequence) -> Model:
        """Fit the model on the provided dataset."""
        if self.transfer_epochs > 0:
            print('---> Compiling model')
            self.model.compile(
                optimizer=self.transfer_optimizer,
                loss=self.loss_func,
                metrics=['accuracy'],
            )

            print('---> Starting transfer learning')
            for layer in self.model.layers[:-1]:
                layer.trainable = False

            self.model.fit(
                x=train_dataset,
                validation_data=val_dataset,
                batch_size=self.batch_size,
                epochs=self.transfer_epochs,
                callbacks=self.transfer_callbacks,
                workers=self.num_workers,
                use_multiprocessing=self.num_workers > 0,
            )

        if self.finetune_epochs > 0:
            print(f'---> Unfreezing last {self.finetune_layers} layers')
            for layer in self.model.layers[: -self.finetune_layers]:
                layer.trainable = False
            for layer in self.model.layers[-self.finetune_layers :]:
                layer.trainable = True

            print('---> Compiling model')
            self.model.compile(
                optimizer=self.finetune_optimizer,
                loss=self.loss_func,
                metrics=['accuracy'],
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
            )

        return self.model

    def reset_model(self) -> None:
        """Set model to initial state as obtained from model factory."""
        self.model, self.preproc_func = ModelFactory.get(
            model_id=self.model_backbone, num_classes=self.num_classes
        )

    def save_model(self, file_path: str) -> None:
        """Save a model checkpoint."""
        self.model.save(file_path)

    def load_model(self, file_path: str) -> None:
        """Load a model from a checkpoint."""
        self.model = keras.models.load_model(file_path)

    def predict(self, dataset: Sequence) -> Dict[str, float]:
        """Make predictions according to trained model."""
        preds = self.model.predict(dataset)
        return dict(zip(dataset.keys, preds))
