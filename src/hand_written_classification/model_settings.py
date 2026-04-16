"""Model construction utilities for handwritten digit classification."""

import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Input

from hand_written_classification.config import TrainingConfig


class ModelFactory:
    """Builds Keras models for MNIST classification."""

    def __init__(self, config: TrainingConfig) -> None:
        """Initializes the model factory.

        Args:
            config: Training configuration.
        """
        self._config = config

    def build_baseline_model(self) -> Sequential:
        """Builds a baseline softmax classifier.

        Returns:
            A compiled Keras Sequential model.
        """
        model = Sequential(
            [
                Input(shape=(self._config.input_dim,)),
                Dense(self._config.num_classes, activation="softmax"),
            ]
        )
        return self._compile(model)

    def build_deep_model(self) -> Sequential:
        """Builds a deeper feed-forward neural network.

        Returns:
            A compiled Keras Sequential model.
        """
        model = Sequential(
            [
                Input(shape=(self._config.input_dim,)),
                Dense(128, activation="relu"),
                Dense(64, activation="relu"),
                Dense(32, activation="relu"),
                Dense(self._config.num_classes, activation="softmax"),
            ]
        )
        return self._compile(model)

    def _compile(self, model: Sequential) -> Sequential:
        """Compiles a Keras model.

        Args:
            model: Uncompiled Keras model.

        Returns:
            A compiled Keras model.
        """
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self._config.learning_rate
        )
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model