"""Application entrypoint for MNIST handwritten digit classification."""

from hand_written_classification.config import TrainingConfig
from hand_written_classification.data_loader import MnistDataLoader
from hand_written_classification.model_settings import ModelFactory
from hand_written_classification.trainer import MnistTrainer
from hand_written_classification.utils import set_global_seed

import matplotlib.pyplot as plt
from keras.datasets import mnist


def main() -> None:
    """Runs the full MNIST training and evaluation pipeline."""
    config = TrainingConfig()
    set_global_seed(config.seed)

    (raw_x_train, _), (raw_x_test, _) = mnist.load_data()

    data_loader = MnistDataLoader(input_dim=config.input_dim)
    x_train, y_train, x_test, y_test = data_loader.load_data()

    print(f"x_train shape: {x_train.shape}")
    print(f"x_test shape: {x_test.shape}")

    data_loader.show_sample(raw_x_train, index=0)
    data_loader.show_grid(raw_x_train, n=20)

    model_factory = ModelFactory(config)
    trainer = MnistTrainer(config)

    model = model_factory.build_deep_model()
    model.summary()

    history = trainer.train(model, x_train, y_train)
    metrics = trainer.evaluate(model, x_test, y_test)

    print("\nEvaluation results:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    y_pred = trainer.predict_labels(model, x_test)
    trainer.plot_confusion_matrix(y_test, y_pred)
    trainer.plot_training_history(history)

    sample_index = 3
    plt.figure(figsize=(4, 4))
    plt.imshow(raw_x_test[sample_index], cmap="gray")
    plt.title(f"Predicted label: {y_pred[sample_index]}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()