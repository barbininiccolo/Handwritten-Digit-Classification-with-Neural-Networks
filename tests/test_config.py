from hand_written_classification.config import TrainingConfig


def test_training_config_defaults() -> None:
    config = TrainingConfig()

    assert config.input_dim == 28 * 28
    assert config.num_classes == 10
    assert config.batch_size > 0
    assert 0 < config.validation_split < 1