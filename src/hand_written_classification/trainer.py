"""Training and evaluation utilities for handwritten digit models."""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from keras.callbacks import EarlyStopping
from keras.models import Sequential

from hand_written_classification.config import TrainingConfig


class MnistTrainer:
    """Handles model training, evaluation, and diagnostics."""

    def __init__(self, config: TrainingConfig) -> None:
        """Initializes the trainer.

        Args:
            config: Training configuration.
        """
        self._config = config

    def train(
        self,
        model: Sequential,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ) -> tf.keras.callbacks.History:
        """Trains a model on the training set.

        Args:
            model: Compiled Keras model.
            x_train: Training features.
            y_train: Training labels.

        Returns:
            Training history returned by Keras.
        """
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            )
        ]

        history = model.fit(
            x_train,
            y_train,
            batch_size=self._config.batch_size,
            epochs=self._config.epochs,
            validation_split=self._config.validation_split,
            callbacks=callbacks,
            verbose=1,
        )
        return history

    def evaluate(
        self,
        model: Sequential,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluates a trained model on the test set.

        Args:
            model: Trained Keras model.
            x_test: Test features.
            y_test: Test labels.

        Returns:
            A dictionary with evaluation metrics.
        """
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        return {
            "test_loss": float(loss),
            "test_accuracy": float(accuracy),
        }

    def predict_labels(
        self,
        model: Sequential,
        x_test: np.ndarray,
    ) -> np.ndarray:
        """Predicts class labels for the given inputs.

        Args:
            model: Trained Keras model.
            x_test: Test features.

        Returns:
            Predicted label array.
        """
        probabilities = model.predict(x_test, verbose=0)
        return np.argmax(probabilities, axis=1)

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """Plots a confusion matrix.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
        """
        matrix = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        plt.title("Confusion Matrix")
        plt.show()

    def plot_training_history(
        self,
        history: tf.keras.callbacks.History,
    ) -> None:
        """Plots training and validation accuracy.

        Args:
            history: Training history returned by Keras.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(history.history["accuracy"], label="Train Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training History")
        plt.legend()
        plt.grid(True)
        plt.show()