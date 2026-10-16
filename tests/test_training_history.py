import torch
from torch.utils.data import DataLoader, TensorDataset

from mnist_mlp.model import MLPClassifier
from mnist_mlp.train import train_model


def test_train_model_returns_epoch_metrics_for_train_and_validation():
    torch.manual_seed(0)
    features = torch.randn(64, 1, 28, 28)
    labels = torch.randint(0, 10, (64,))

    train_loader = DataLoader(TensorDataset(features[:48], labels[:48]), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(features[48:], labels[48:]), batch_size=16, shuffle=False)

    model = MLPClassifier(hidden_sizes=(32, 16))
    history, training_time_seconds, _ = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        lr=1e-3,
        optimizer_name="adam",
        device=torch.device("cpu"),
    )

    assert training_time_seconds >= 0
    for key in ["train_loss", "train_accuracy", "val_loss", "val_accuracy"]:
        assert key in history
        assert len(history[key]) == 2
