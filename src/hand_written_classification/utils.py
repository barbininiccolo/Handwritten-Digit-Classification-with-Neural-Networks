"""Utility helpers for the handwritten classification project."""

import numpy as np
import tensorflow as tf


def set_global_seed(seed: int) -> None:
    """Sets global random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)