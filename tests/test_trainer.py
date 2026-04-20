import numpy as np

from hand_written_classification.config import TrainingConfig
from hand_written_classification.model_settings import ModelFactory
from hand_written_classification.trainer import MnistTrainer


def test_predict_labels_returns_expected_shape() -> None:
    config = TrainingConfig(epochs=1, batch_size=8)
    factory = ModelFactory(config)
    trainer = MnistTrainer(config)

    model = factory.build_baseline_model()

    x_train = np.random.rand(16, 784).astype("float32")
    y_train = np.random.randint(0, 10, size=(16,))
    x_test = np.random.rand(4, 784).astype("float32")

    trainer.train(model, x_train, y_train)
    predictions = trainer.predict_labels(model, x_test)

    assert predictions.shape == (4,)