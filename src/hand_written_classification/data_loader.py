"""Data loading and visualization utilities for MNIST."""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist


class MnistDataLoader:
    """Loads, preprocesses, and visualizes the MNIST dataset."""

    def __init__(self, input_dim: int) -> None:
        """Initializes the data loader.

        Args:
            input_dim: Expected flattened input dimension.
        """
        self._input_dim = input_dim

    def load_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Loads and preprocesses the MNIST dataset.

        Returns:
            A tuple containing training features, training labels,
            test features, and test labels.
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = self._flatten_and_normalize(x_train)
        x_test = self._flatten_and_normalize(x_test)

        return x_train, y_train, x_test, y_test

    def show_sample(self, images: np.ndarray, index: int = 0) -> None:
        """Displays a single MNIST image.

        Args:
            images: Array of original 28x28 images.
            index: Index of the image to display.
        """
        plt.figure(figsize=(4, 4))
        plt.imshow(images[index], cmap="gray")
        plt.title(f"Image at index {index}")
        plt.axis("off")
        plt.show()

    def show_grid(self, images: np.ndarray, n: int = 20) -> None:
        """Displays the first n images in a grid.

        Args:
            images: Array of original 28x28 images.
            n: Number of images to display.
        """
        cols = 5
        rows = int(np.ceil(n / cols))

        plt.figure(figsize=(10, 8))
        for i in range(n):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(images[i], cmap="gray")
            plt.axis("off")
            plt.title(f"#{i}")
        plt.tight_layout()
        plt.show()

    def _flatten_and_normalize(self, images: np.ndarray) -> np.ndarray:
        """Flattens images and normalizes pixel values to [0, 1].

        Args:
            images: Input image array of shape (num_samples, 28, 28).

        Returns:
            Preprocessed image array of shape (num_samples, input_dim).
        """
        flattened = images.reshape(images.shape[0], self._input_dim)
        return flattened.astype("float32") / 255.0