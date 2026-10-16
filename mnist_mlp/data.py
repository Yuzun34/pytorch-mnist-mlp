from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_mnist_loaders(
    data_dir: str,
    batch_size: int,
    val_split: float,
    seed: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )

    train_val = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    val_size = int(len(train_val) * val_split)
    train_size = len(train_val) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(train_val, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
