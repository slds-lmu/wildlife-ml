"""Classes and functions for creating model backbones."""
from typing import (
    Callable,
    Final,
    Tuple,
)

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

AVAILABLE_MODELS: Final = {
    'resnet50': {
        'model': tf.keras.applications.ResNet50V2,
        'input_shape': (224, 224),
        'preproc_func': tf.keras.applications.resnet_v2.preprocess_input,
    }
}


class ModelFactory:
    """Factory for creating Keras model objects."""

    @staticmethod
    def get(
        model_id: str,
        num_classes: int,
        weights: str = 'imagenet',
        include_top: bool = False,
        pooling: str = 'avg',
    ) -> Tuple[tf.keras.Model, Callable]:
        """
        Return an initialized model instance from an identifier.

        Note: The model still needs to be compiled.
        """
        model_entry = AVAILABLE_MODELS[model_id]
        model_cls = model_entry['model']
        model = model_cls(weights=weights, include_top=include_top, pooling=pooling)
        prediction = Dense(num_classes, activation='softmax')(model.output)

        return (
            Model(inputs=model.input, outputs=prediction),
            model_entry['preproc_func'],
        )
