import torch

from mnist_mlp.model import MLPClassifier


def test_mlp_has_minimum_two_hidden_layers_and_relu():
    model = MLPClassifier(hidden_sizes=(256, 128))
    linear_layers = [m for m in model.network if isinstance(m, torch.nn.Linear)]
    relu_layers = [m for m in model.network if isinstance(m, torch.nn.ReLU)]

    assert len(linear_layers) == 3  # 2 hidden + 1 output
    assert len(relu_layers) == 2


def test_output_dimension_is_10_classes():
    model = MLPClassifier(hidden_sizes=(64, 32))
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    assert y.shape == (4, 10)
