"""Configuration objects for the handwritten classification project."""

from dataclasses import dataclass

@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for model training and evaluation.

    Attributes:
        input_dim: Flattened image dimension.
        num_classes: Number of output classes.
        batch_size: Number of samples per gradient update.
        epochs: Maximum number of training epochs.
        validation_split: Fraction of training data used for validation.
        learning_rate: Optimizer learning rate.
        seed: Random seed for reproducibility.
    """

    input_dim: int = 28 * 28
    num_classes: int = 10
    batch_size: int = 128
    epochs: int = 20
    validation_split: float = 0.1
    learning_rate: float = 1e-3
    seed: int = 42