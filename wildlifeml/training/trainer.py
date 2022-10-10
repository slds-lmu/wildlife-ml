"""Classes for managing training."""
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Final,
    List,
    Optional,
    Tuple,
)

import numpy as np
import ray
from ray.tune.integration.keras import TuneReportCallback
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

from wildlifeml.data import (
    map_bbox_to_img,
    merge_datasets,
    subset_dataset,
)
from wildlifeml.training.algorithms import AlgorithmFactory
from wildlifeml.training.models import ModelFactory
from wildlifeml.utils.datasets import do_stratified_cv
from wildlifeml.utils.misc import flatten_list

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
        eval_metrics: Optional[List] = None,
        pretraining_checkpoint: Optional[str] = None,
    ) -> None:
        """Initialize trainer object."""
        self.num_classes = num_classes
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
                run_eagerly=True,
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
                metrics=self.eval_metrics,
                run_eagerly=True,
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
        if self.pretraining_checkpoint is not None:
            self.model.load_weights(self.pretraining_checkpoint)

    def save_model(self, file_path: str) -> None:
        """Save a model checkpoint."""
        self.model.save(file_path)

    def save_model_weights(self, file_path: str) -> None:
        """Save model weights."""
        self.model.save_weights(file_path)

    def load_model(self, file_path: str) -> None:
        """Load a model from a checkpoint."""
        self.model = keras.models.load_model(file_path)

    def predict(self, dataset: Sequence) -> np.ndarray:
        """Make predictions according to trained model."""
        return self.model.predict(dataset)


class WildlifeTuningTrainer(BaseTrainer):
    """Trainer class for hyperparameter tuning."""

    def __init__(
        self,
        search_space: Dict,
        loss_func: Any,
        num_classes: int,
        transfer_epochs: int,
        finetune_epochs: int,
        finetune_layers: int,
        transfer_optimizer: Any,
        finetune_optimizer: Any,
        num_workers: int = 0,
        transfer_callbacks: Optional[List] = None,
        finetune_callbacks: Optional[List] = None,
        eval_metrics: Optional[List] = None,
        pretraining_checkpoint: Optional[str] = None,
        local_dir: str = './ray_results/',
        random_state: int = 123,
        resources_per_trial: Optional[Dict] = None,
        max_concurrent_trials: int = 2,
        objective: str = 'val_accuracy',
        mode: str = 'max',
        n_trials: int = 2,
        transfer_epochs_per_trial: Optional[int] = 1,
        finetune_epochs_per_trial: Optional[int] = 1,
        time_budget: int = 3600,
        verbose: int = 0,
        search_alg_id: str = 'hyperoptsearch',
        scheduler_alg_id: str = 'ashascheduler',
    ) -> None:
        """Initialize tuner object."""
        if set(TUNABLE) != set(search_space):
            raise IOError(
                f'Please provide search ranges for all HP in {TUNABLE}. '
                f'To exclude a HP from tuning, specify a single-element choice.'
            )
        self.search_space = search_space
        self.num_classes = num_classes
        self.transfer_epochs = transfer_epochs
        self.finetune_epochs = finetune_epochs
        self.transfer_optimizer = transfer_optimizer
        self.finetune_optimizer = finetune_optimizer
        self.finetune_layers = finetune_layers
        self.transfer_callbacks = transfer_callbacks
        self.finetune_callbacks = finetune_callbacks
        self.pretraining_checkpoint = pretraining_checkpoint

        self.loss_func = loss_func

        self.objective = objective
        self.mode = mode
        self.n_trials = n_trials
        self.time_budget = time_budget
        self.local_dir = local_dir
        self.random_state = random_state
        self.verbose = verbose

        self.transfer_epochs_per_trial = transfer_epochs_per_trial or transfer_epochs
        self.finetune_epochs_per_trial = finetune_epochs_per_trial or finetune_epochs
        self.max_concurrent_trials = max_concurrent_trials
        self.resources_per_trial = resources_per_trial or {'cpu': 1}
        self.num_workers = num_workers

        self.search_algorithm = AlgorithmFactory.get(search_alg_id)()
        self.scheduler_algorithm = AlgorithmFactory.get(scheduler_alg_id)()

        self.eval_metrics = eval_metrics

        if eval_metrics is not None:
            eval_metrics_names = []
            for metric in eval_metrics:
                name = metric if isinstance(metric, str) else metric.name
                eval_metrics_names.append('val_' + name)
                eval_metrics_names.append(name)

            if objective not in eval_metrics_names:
                raise IOError(
                    'The objective must be among the evaluation metrics. '
                    'Please add the corresponding objective function to eval_metrics.'
                )
            else:
                self.report_metrics = {objective: objective}

        self.optimal_config: Optional[Dict] = None
        self.model: Optional[Model] = None

    def get_num_classes(self) -> int:
        """Return number of classes."""
        return self.num_classes

    def fit(self, train_dataset: Sequence, val_dataset: Sequence) -> Model:
        """Fit the model on the provided dataset."""
        analysis = ray.tune.run(
            ray.tune.with_parameters(
                self._fit_trial,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            ),
            config=self.search_space,
            search_alg=self.search_algorithm,
            scheduler=self.scheduler_algorithm,
            metric=self.objective,
            mode=self.mode,
            num_samples=self.n_trials,
            time_budget_s=self.time_budget,
            verbose=self.verbose,
            local_dir=self.local_dir,
            resources_per_trial=self.resources_per_trial,
            max_concurrent_trials=self.max_concurrent_trials,
        )
        self.optimal_config = analysis.best_config
        if self.optimal_config is None:
            raise ValueError('Tuning produced no optimal configuration.')
        else:
            self.transfer_optimizer.learning_rate = self.optimal_config[
                'transfer_learning_rate'
            ]
            self.finetune_optimizer.learning_rate = self.optimal_config[
                'finetune_learning_rate'
            ]
            optimal_trainer = WildlifeTrainer(
                batch_size=self.optimal_config['batch_size'],
                loss_func=self.loss_func,
                num_classes=self.num_classes,
                transfer_epochs=self.transfer_epochs,
                finetune_epochs=self.finetune_epochs,
                transfer_optimizer=self.transfer_optimizer,
                finetune_optimizer=self.finetune_optimizer,
                finetune_layers=self.finetune_layers,
                model_backbone=self.optimal_config['backbone'],
                transfer_callbacks=self.transfer_callbacks,
                finetune_callbacks=self.finetune_callbacks,
                num_workers=self.num_workers,
                eval_metrics=self.eval_metrics,
                pretraining_checkpoint=self.pretraining_checkpoint,
            )
            merged_dataset = merge_datasets(train_dataset, val_dataset)
            merged_dataset.batch_size = self.optimal_config['batch_size']
            self.model = optimal_trainer.fit(train_dataset=merged_dataset)
            return self.model

    def _fit_trial(
        self, config: Dict, train_dataset: Sequence, val_dataset: Sequence
    ) -> None:
        """Worker function for ray trials."""
        # Set current HP configs (if being tuned)
        trainer = WildlifeTrainer(
            batch_size=config['batch_size'],
            loss_func=self.loss_func,
            num_classes=self.num_classes,
            transfer_epochs=self.transfer_epochs_per_trial,
            finetune_epochs=self.finetune_epochs_per_trial,
            transfer_optimizer=Adam(config['transfer_learning_rate']),
            finetune_optimizer=Adam(config['finetune_learning_rate']),
            finetune_layers=self.finetune_layers,
            model_backbone=config['backbone'],
            transfer_callbacks=[
                TuneReportCallback(metrics=self.report_metrics, on='epoch_end')
            ],
            finetune_callbacks=[
                TuneReportCallback(metrics=self.report_metrics, on='epoch_end')
            ],
            num_workers=self.num_workers,
            eval_metrics=self.eval_metrics,
            pretraining_checkpoint=self.pretraining_checkpoint,
        )
        train_dataset.batch_size = config['batch_size']
        val_dataset.batch_size = config['batch_size']
        trainer.fit(train_dataset=train_dataset, val_dataset=val_dataset)

    def get_model(self) -> Model:
        """Return the model instance."""
        if self.model is None:
            raise ValueError('There is no model yet. Please fit the trainer.')
        return self.model

    def compile_model(self) -> None:
        """Compile model for evaluation."""
        if self.model is None:
            raise ValueError('There is no model yet. Please fit the trainer.')
        self.model.compile(
            optimizer=self.transfer_optimizer,
            loss=self.loss_func,
            metrics=self.eval_metrics,
            run_eagerly=True,
        )

    def reset_model(self) -> None:
        """Set model to initial state as obtained from model factory."""
        if self.model is not None and self.optimal_config is not None:
            self.model = ModelFactory.get(
                model_id=self.optimal_config['model_backbone'],
                num_classes=self.num_classes,
            )

    def save_model(self, file_path: str) -> None:
        """Save a model checkpoint."""
        if self.model is not None:
            self.model.save(file_path)

    def save_model_weights(self, file_path: str) -> None:
        """Save model weights."""
        if self.model is not None:
            self.model.save_weights(file_path)

    def load_model(self, file_path: str) -> None:
        """Load a model from a checkpoint."""
        self.model = keras.models.load_model(file_path)

    def predict(self, dataset: Sequence) -> np.ndarray:
        """Make predictions according to trained model."""
        if self.model is None:
            raise ValueError('There is no model yet. Please fit the trainer.')
        return self.model.predict(dataset)

    def cal_epochs(
        self,
        dataset: Sequence,
        folds: int = 5,
        n_runs: int = 2,
        patience: int = 1,
        max_epochs: int = 50,
    ) -> Tuple[int, int]:
        """Calculate the optimal number of epochs with fixed hyperparameters."""
        if self.optimal_config is None:
            raise ValueError('There is no optimal config yet. Please fit the trainer.')

        batch_size = self.optimal_config['batch_size']
        dataset.batch_size = batch_size
        img_keys = [map_bbox_to_img(k) for k in dataset.keys()]

        transfer_stopped_epoch_list = []
        finetune_stopped_epoch_list = []
        for index_run in range(n_runs):

            keys_train, keys_val = do_stratified_cv(
                img_keys=img_keys,
                folds=folds,
                meta_dict={k: {'label': v} for k, v in dataset.label_dict.items()},
            )

            for index_fold in range(folds):
                dataset_train = subset_dataset(
                    dataset,
                    flatten_list(
                        [dataset.mapping_dict[k] for k in keys_train[index_fold]]
                    ),
                )
                dataset_val = subset_dataset(
                    dataset,
                    flatten_list(
                        [dataset.mapping_dict[k] for k in keys_val[index_fold]]
                    ),
                )

                transfer_earlystop = EarlyStopping(
                    monitor=self.objective, patience=patience, mode=self.mode
                )

                finetune_earlystop = EarlyStopping(
                    monitor=self.objective, patience=patience, mode=self.mode
                )

                trainer = WildlifeTrainer(
                    batch_size=batch_size,
                    loss_func=self.loss_func,
                    num_classes=self.num_classes,
                    transfer_epochs=max_epochs if self.transfer_epochs else 0,
                    finetune_epochs=max_epochs if self.finetune_epochs else 0,
                    transfer_optimizer=Adam(
                        self.optimal_config['transfer_learning_rate']
                    ),
                    finetune_optimizer=Adam(
                        self.optimal_config['finetune_learning_rate']
                    ),
                    finetune_layers=self.finetune_layers,
                    model_backbone=self.optimal_config['model_backbone'],
                    transfer_callbacks=[transfer_earlystop],
                    finetune_callbacks=[finetune_earlystop],
                    num_workers=self.num_workers,
                    eval_metrics=self.eval_metrics,
                )

                trainer.fit(train_dataset=dataset_train, val_dataset=dataset_val)

                transfer_stopped_epoch = transfer_earlystop.stopped_epoch
                if transfer_stopped_epoch == 0:
                    transfer_stopped_epoch = max_epochs
                transfer_stopped_epoch_list.append(transfer_stopped_epoch)

                finetune_stopped_epoch = finetune_earlystop.stopped_epoch
                if finetune_stopped_epoch == 0:
                    finetune_stopped_epoch = max_epochs
                finetune_stopped_epoch_list.append(finetune_stopped_epoch)

        optimal_transfer_epochs = int(np.mean(transfer_stopped_epoch_list))
        optimal_finetune_epochs = int(np.mean(finetune_stopped_epoch_list))

        print(
            f'Found optimal epochs: {optimal_transfer_epochs} for transfer learning, '
            f'{optimal_finetune_epochs} for fine-tuning.'
        )

        optimal_hps = self.optimal_config.copy()
        optimal_hps['transfer_epochs'] = optimal_transfer_epochs
        optimal_hps['finetune_epochs'] = optimal_finetune_epochs
        return optimal_transfer_epochs, optimal_finetune_epochs
