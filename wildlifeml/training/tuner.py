"""Classes for managing tuning."""

from typing import (Any, Dict, List, Optional)
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import Sequence
from wildlifeml.training.trainer import WildlifeTrainer
from wildlifeml.data import modify_dataset
from wildlifeml.utils.datasets import do_stratified_cv
from wildlifeml.training.algorithms import AlgorithmFactory
import ray
from ray.tune import ExperimentAnalysis
from ray.tune.integration.keras import TuneReportCallback


class WildlifeTuner:
    """Tuner object for assisting with hypertuning neural networks."""
    
    def __init__(
        self,
        search_space: Dict,
        loss_func: Any,
        num_classes: int,
        transfer_epochs: int,
        finetune_epochs: int,
        finetune_layers: int,
        num_workers: int = 0,
        eval_metrics: Optional[List] = None,
        local_dir: str = './ray_results/',
        random_state: int = 123,
        resources_per_trial: Dict = {'cpu': 8},
        max_concurrent_trials: int = 2,
        objective: str = 'val_accuracy',
        mode: str = 'max',
        n_trials: int = 2,
        epochs_per_trial: int = 1,
        time_budget: int = 3600,
        verbose: int = 0,
        search_alg_id: str = 'hyperoptsearch', 
        scheduler_alg_id: str = 'ashascheduler',
        ) -> None:
        """Initialize tuner object."""
        
        self.search_space = search_space
        self.loss_func = loss_func
        self.num_classes = num_classes
        self.transfer_epochs = transfer_epochs
        self.finetune_epochs = finetune_epochs
        self.finetune_layers = finetune_layers
        
        self.objective = objective
        self.mode = mode
        self.n_trials = n_trials
        self.time_budget = time_budget
        self.local_dir = local_dir
        self.random_state = random_state
        self.verbose = verbose
        
        self.epochs_per_trial = epochs_per_trial
        self.max_concurrent_trials = max_concurrent_trials
        self.resources_per_trial = resources_per_trial
        self.num_workers = num_workers

        self.search_algorithm    = None if search_alg_id is None else AlgorithmFactory.get(search_alg_id)(metric=self.objective, mode=self.mode, random_state_seed = self.random_state) 
        self.scheduler_algorithm = None if scheduler_alg_id is None else AlgorithmFactory.get(scheduler_alg_id)()
        self.eval_metrics = eval_metrics
        self.report_metrics = {**{metric: metric for metric in self.eval_metrics}, **{'val_'+metric:'val_'+metric for metric in self.eval_metrics}}
        
        if transfer_epochs == 0: self.search_space.pop('transfer_learning_rate')
        if finetune_epochs == 0: self.search_space.pop('finetune_learning_rate')   
        
    def search(self, dataset_train: Sequence, dataset_val: Sequence)-> ExperimentAnalysis:
        """Search for the best HP combination via drawing and evaluating several HP combinations (i.e., trials). """

        self.analysis = ray.tune.run(
            ray.tune.with_parameters(WildlifeTuner.evaluate, dataset_train=dataset_train, dataset_val=dataset_val, self = self),
            config                = self.search_space, 
            search_alg            = self.search_algorithm, 
            scheduler             = self.scheduler_algorithm, 
            metric                = self.objective, 
            mode                  = self.mode, 
            num_samples           = self.n_trials, 
            time_budget_s         = self.time_budget, 
            verbose               = self.verbose, 
            local_dir             = self.local_dir, 
            resources_per_trial   = self.resources_per_trial, 
            max_concurrent_trials = self.max_concurrent_trials)
        return self.analysis

    def evaluate(config, dataset_train, dataset_val, self, checkpoint_dir = None):
        """Evalaute the performance of a model built with a particular set of hyperparameters."""
        
        batch_size = config['batch_size']
        trainer = WildlifeTrainer(
                batch_size         = batch_size,
                loss_func          = self.loss_func,
                num_classes        = self.num_classes,
                transfer_epochs    = self.epochs_per_trial if self.transfer_epochs else 0,
                finetune_epochs    = self.epochs_per_trial if self.finetune_epochs else 0,
                transfer_optimizer = keras.optimizers.Adam(config['transfer_learning_rate']) if self.transfer_epochs else None,
                finetune_optimizer = keras.optimizers.Adam(config['finetune_learning_rate']) if self.finetune_epochs else None,
                finetune_layers    = self.finetune_layers,
                model_backbone     = config['model_backbone'],
                transfer_callbacks = [TuneReportCallback(metrics=self.report_metrics)] if self.transfer_epochs else None,
                finetune_callbacks = [TuneReportCallback(metrics=self.report_metrics)] if self.finetune_epochs else None,
                num_workers        = self.num_workers,
                eval_metrics       = self.eval_metrics)
        dataset_train.batch_size = dataset_val.batch_size = batch_size 
        trainer.fit(train_dataset=dataset_train, val_dataset=dataset_val)
        
    def extract_trial(self, experiment_dir, trial_id) -> Dict:
        """Return the optimal trial by loading the previously performed experiment."""

        if experiment_dir=='infer': 
            if self.analysis is not None:
                analysis = self.analysis
            else:
                experiment_dir = os.path.join(self.local_dir, os.listdir(self.local_dir)[-1])
                analysis = ExperimentAnalysis(experiment_dir)
        df_trials = analysis.results_df
        df_trials.sort_values(self.objective, ascending={'max': False, 'min':True}[self.mode], inplace=True)
        df_trials.reset_index(inplace= True) 

        if trial_id =='infer':
            trial = df_trials.iloc[0].to_dict()
        else:
            trial = df_trials[df_trials['trial_id']==trial_id].iloc[0].to_dict()

        return trial
    
    def cal_epochs(self, dataset_train: Sequence, dataset_val: Sequence, experiment_dir: str = 'infer', trial_id: str ='infer', folds: int = 5, n_runs: int = 2, patience: int = 1, max_epochs: int = 50) -> int:
        """Calculate the optimal number of epochs when other hyperparameters are kept fixed."""
        
        trial = self.extract_trial(experiment_dir, trial_id)
        batch_size = trial['config.batch_size']
        dataset_tv = modify_dataset(dataset_train, extend=True, keys=list(dataset_val.keys), new_label_dict= dataset_val.label_dict)
        
        dataset_tv.batch_size = batch_size
        
        stopped_epoch_list=[]
        for index_run in range(n_runs):
            
            keys_train, keys_val = do_stratified_cv(
                mapping_dict = dataset_tv.mapping_dict, 
                img_keys = ['_'.join(key.split('_')[:-1]) for key in dataset_tv.keys],
                folds=folds, 
                meta_dict = {k: {'label': v} for k, v in dataset_tv.label_dict.items()})

            for index_fold in range(folds):
                dataset_train = modify_dataset(dataset=dataset_tv, keys=keys_train[index_fold])
                dataset_val   = modify_dataset(dataset=dataset_tv, keys=keys_val  [index_fold])
                
                callback_EarlyStop = keras.callbacks.EarlyStopping(monitor=self.objective, patience=patience, mode=self.mode)
                trainer = WildlifeTrainer(
                    batch_size         = batch_size,
                    loss_func          = self.loss_func,
                    num_classes        = self.num_classes,
                    transfer_epochs    = max_epochs if self.transfer_epochs else 0,
                    finetune_epochs    = max_epochs if self.finetune_epochs else 0,
                    transfer_optimizer = keras.optimizers.Adam(trial['config.transfer_learning_rate']) if self.transfer_epochs else None,
                    finetune_optimizer = keras.optimizers.Adam(trial['config.finetune_learning_rate']) if self.finetune_epochs else None,
                    finetune_layers    = self.finetune_layers,
                    model_backbone     = trial['config.model_backbone'],
                    transfer_callbacks = [callback_EarlyStop] if self.transfer_epochs else None,
                    finetune_callbacks = [callback_EarlyStop] if self.finetune_epochs else None,
                    num_workers        = self.num_workers,
                    eval_metrics       = self.eval_metrics)    
                
                trainer.fit(train_dataset=dataset_train, val_dataset=dataset_val)
                stopped_epoch = max_epochs if callback_EarlyStop.stopped_epoch==0 else callback_EarlyStop.stopped_epoch  
                stopped_epoch_list.append(stopped_epoch)
            print('Completed evaluation on run: {} => Optimal Epochs: {}'.format(1+index_run, stopped_epoch))
        optimal_epochs = int(np.mean(stopped_epoch_list))
        print('Different values for Epochs: {} => Epoch = {}'.format(stopped_epoch_list, optimal_epochs))

        optimal_hps = dict()
        optimal_hps['transfer_epochs'] =  optimal_epochs if self.transfer_epochs else 0
        optimal_hps['finetune_epochs'] =  optimal_epochs if self.finetune_epochs else 0
        for key, value in trial.items():
            if key.startswith('config.'):
                optimal_hps[key.split('config.')[-1]] = value
        self.optimal_hps = optimal_hps
        return optimal_epochs
