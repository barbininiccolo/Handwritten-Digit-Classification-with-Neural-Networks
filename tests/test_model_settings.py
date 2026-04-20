from hand_written_classification.config import TrainingConfig
from hand_written_classification.model_settings import ModelFactory


def test_baseline_model_output_shape() -> None:
    config = TrainingConfig()
    factory = ModelFactory(config)

    model = factory.build_baseline_model()

    assert model.input_shape == (None, 784)
    assert model.output_shape == (None, 10)