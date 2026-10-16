from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    def __init__(self, input_size: int = 28 * 28, hidden_sizes: tuple[int, ...] = (256, 128), num_classes: int = 10):
        super().__init__()
        if len(hidden_sizes) < 2:
            raise ValueError("Model must include at least two hidden layers.")

        layers: list[nn.Module] = []
        prev = input_size
        for width in hidden_sizes:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            prev = width
        layers.append(nn.Linear(prev, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.network(x)
