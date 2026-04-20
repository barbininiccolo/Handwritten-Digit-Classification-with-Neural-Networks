import numpy as np

from hand_written_classification.utils import set_global_seed


def test_set_global_seed_makes_numpy_reproducible() -> None:
    set_global_seed(42)
    first = np.random.rand(5)

    set_global_seed(42)
    second = np.random.rand(5)

    assert np.array_equal(first, second)