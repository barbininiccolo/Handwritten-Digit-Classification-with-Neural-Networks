import numpy as np

from hand_written_classification.data_loader import MnistDataLoader


def test_flatten_and_normalize_output_shape_and_range() -> None:
    loader = MnistDataLoader(input_dim=784)
    images = np.full((2, 28, 28), 255, dtype=np.uint8)

    output = loader._flatten_and_normalize(images)

    assert output.shape == (2, 784)
    assert output.dtype == np.float32
    assert np.all(output >= 0.0)
    assert np.all(output <= 1.0)
    assert np.allclose(output, 1.0)