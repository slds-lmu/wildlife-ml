"""Classes for managing training."""
from typing import Any, List

from tensorflow.keras import Model, Sequence

from wildlifeml.training.models import ModelFactory


class WildlifeTrainer:
    """Trainer object for assisting with fitting neural networks."""

    def __init__(
        self,
        model_backbone: str,
        loss_func: Any,
        num_classes: int,
        transfer_epochs: int,
        transfer_optimizer: Any,
        transfer_callbacks: List,
        finetune_epochs: int,
        finetune_optimizer: Any,
        finetune_callbacks: List,
        finetune_layers: int,
        batch_size: int,
        num_workers: int,
    ) -> None:
        """Initialize trainer object."""
        self.num_classes = num_classes
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

    def fit(self, dataset: Sequence) -> Model:
        """Fit the model on the provided dataset."""
        if self.transfer_epochs > 0:
            print('---> Compiling model')
            self.model.compile(optimizer=self.transfer_optimizer, loss=self.loss_func)

            print('---> Starting transfer learning')
            for layer in self.model.layers[:-1]:
                layer.trainable = False

            self.model.fit(
                x=dataset,
                batch_size=self.batch_size,
                epochs=self.transfer_epochs,
                workers=self.num_workers,
                use_multiprocessing=self.num_workers > 0,
            )

        if self.finetune_epochs > 0:
            print('---> Unfreezing last {} layers'.format(self.finetune_layers))
            for layer in self.model.layers[: -self.finetune_layers]:
                layer.trainable = False
            for layer in self.model.layers[-self.finetune_layers :]:
                layer.trainable = True

            print('---> Compiling model')
            self.model.compile(optimizer=self.finetune_optimizer, loss=self.loss_func)

            print('---> Starting fine tuning')
            self.model.fit(
                x=dataset,
                batch_size=self.batch_size,
                epochs=self.finetune_epochs,
                workers=self.num_workers,
                use_multiprocessing=self.num_workers > 0,
            )

        return self.model
