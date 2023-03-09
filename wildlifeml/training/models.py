"""Classes and functions for creating model backbones."""
from typing import Final

import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.models import Sequential

AVAILABLE_MODELS: Final = {
    'resnet50': {
        'model': tf.keras.applications.ResNet50V2,
        'preproc_func': tf.keras.applications.resnet_v2.preprocess_input,
    },
    'inception_resnet_v2': {
        'model': tf.keras.applications.InceptionResNetV2,
        'preproc_func': tf.keras.applications.inception_resnet_v2.preprocess_input,
    },
    'vgg19': {
        'model': tf.keras.applications.VGG19,
        'preproc_func': tf.keras.applications.vgg19.preprocess_input,
    },
    'xception': {
        'model': tf.keras.applications.Xception,
        'preproc_func': tf.keras.applications.xception.preprocess_input,
    },
    'densenet121': {
        'model': tf.keras.applications.DenseNet121,
        'preproc_func': tf.keras.applications.densenet.preprocess_input,
    },
    'densenet201': {
        'model': tf.keras.applications.DenseNet201,
        'preproc_func': tf.keras.applications.densenet.preprocess_input,
    },
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
    ) -> Sequential:
        """
        Return an initialized model instance from an identifier.

        Note: The model still needs to be compiled.
        """
        model_entry = AVAILABLE_MODELS[model_id]
        model_cls = model_entry['model']

        model = Sequential()
        model.add(Lambda(model_entry['preproc_func']))
        model.add(model_cls(weights=weights, include_top=include_top, pooling=pooling))
        model.add(Dense(num_classes, activation='softmax'))

        return model
