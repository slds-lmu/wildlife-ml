"""Custom metrics suitable for Sparse Categorical computations."""

import warnings
import tensorflow as tf
from tensorflow import keras
import numpy as np


class ComputeMetric:
    def __init__(self, metric_char='recall', average='macro'):
        self._func = {'recall': self._recall, 'precision': self._precision, 'f1': self._f1}[metric_char]
        self.average = average
    def _recall(self, tp, fp, fn):
        return np.nan_to_num(tp/(tp+fn))
    def _precision(self, tp, fp, fn):
        return np.nan_to_num(tp/(tp+fp))
    def _f1(self, tp, fp, fn):
        return np.nan_to_num(2*tp/(2*tp+fp+fn))
    def reduce(self, tp, fp, fn, weights):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if self.average is None:
                return self._func(tp, fp, fn)
            elif self.average=='micro':
                return self._func(tp.sum(), fp.sum(), fn.sum())
            elif self.average=='macro':
                return self._func(tp, fp, fn).mean()
            elif self.average=='weighted':
                return (self._func(tp, fp, fn)*weights).sum()
    def forward(self, y_true, y_pred):
        conf = tf.math.confusion_matrix(
            y_true, 
            tf.math.argmax(y_pred, -1), 
            num_classes=y_pred.shape[1]).numpy()
        tp = np.diagonal(conf)
        fp = conf.sum(0) - tp
        fn = conf.sum(1) - tp
        weights = conf.sum(1)/conf.sum()
        return self.reduce(tp, fp, fn, weights)

class SparseCategoricalRecall(keras.metrics.Metric):
    def __init__(self, name='sparse_categorical_recall', average='macro', **kwargs):
        super(SparseCategoricalRecall, self).__init__(name=name, **kwargs)
        self.computer = ComputeMetric('recall', average)
        self.average = average
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.value = self.computer.forward(y_true, y_pred)
    def result(self):
        return self.value
    
class SparseCategoricalPrecision(keras.metrics.Metric):
    def __init__(self, name='sparse_categorical_precision', average='macro', **kwargs):
        super(SparseCategoricalPrecision, self).__init__(name=name, **kwargs)
        self.computer = ComputeMetric('precision', average)
        self.average = average
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.value = self.computer.forward(y_true, y_pred)
    def result(self):
        return self.value
    
class SparseCategoricalF1(keras.metrics.Metric):
    def __init__(self, name='sparse_categorical_f1', average='macro', **kwargs):
        super(SparseCategoricalF1, self).__init__(name=name, **kwargs)
        self.computer = ComputeMetric('f1', average)
        self.average = average
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.value = self.computer.forward(y_true, y_pred)
    def result(self):
        return self.value