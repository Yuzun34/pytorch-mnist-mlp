import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int = 10):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.network(x)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> Tuple[float, float]:
    model.train(mode=train)
    total_loss = 0.0
    total_accuracy = 0.0
    total_batches = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_accuracy += accuracy_from_logits(logits, labels)
        total_batches += 1

    return total_loss / total_batches, total_accuracy / total_batches


@torch.no_grad()
def evaluate_test_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        predictions = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
    return correct / total


def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    val_ratio: float,
    train_subset: int,
    val_subset: int,
    test_subset: int,
    use_fake_data: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    if use_fake_data:
        fake_train = datasets.FakeData(
            size=60000,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
        )
        test_dataset = datasets.FakeData(
            size=10000,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
        )
        val_size = int(len(fake_train) * val_ratio)
        train_size = len(fake_train) - val_size
        train_split, val_split = random_split(
            fake_train,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        train_dataset = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transform)

        val_size = int(len(train_dataset) * val_ratio)
        train_size = len(train_dataset) - val_size
        train_split, val_split = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    if train_subset > 0:
        train_split = Subset(train_split, range(min(train_subset, len(train_split))))
    if val_subset > 0:
        val_split = Subset(val_split, range(min(val_subset, len(val_split))))
    if test_subset > 0:
        test_dataset = Subset(test_dataset, range(min(test_subset, len(test_dataset))))

    return (
        DataLoader(train_split, batch_size=batch_size, shuffle=True),
        DataLoader(val_split, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


def plot_history(history: Dict[str, List[float]], output_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Validation Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="Train Accuracy")
    axes[1].plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def train_single_config(args: argparse.Namespace, config_name: str, hidden_dims: List[int], learning_rate: float) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        test_subset=args.test_subset,
        use_fake_data=args.use_fake_data,
    )

    model = MLP(input_dim=28 * 28, hidden_dims=hidden_dims).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) if args.optimizer == "adam" else torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, train=False)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        print(
            f"[{config_name}] Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

    elapsed = time.time() - start_time
    test_accuracy = evaluate_test_accuracy(model, test_loader, device)

    output_dir = Path(args.output_dir) / config_name
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_history(history, output_dir / "loss_accuracy.png")
    torch.save(model.state_dict(), output_dir / "model.pt")

    metrics = {
        "config_name": config_name,
        "hidden_dims": hidden_dims,
        "optimizer": args.optimizer,
        "learning_rate": learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "training_loss": history["train_loss"][-1],
        "validation_loss": history["val_loss"][-1],
        "training_accuracy": history["train_accuracy"][-1],
        "validation_accuracy": history["val_accuracy"][-1],
        "test_accuracy": test_accuracy,
        "training_time_seconds": elapsed,
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST dataset with MLP classifier")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default="adam")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 128],
        help="At least two values are required",
    )
    parser.add_argument("--run-experiments", action="store_true")
    parser.add_argument("--train-subset", type=int, default=0, help="Use >0 for smoke tests")
    parser.add_argument("--val-subset", type=int, default=0, help="Use >0 for smoke tests")
    parser.add_argument("--test-subset", type=int, default=0, help="Use >0 for smoke tests")
    parser.add_argument(
        "--use-fake-data",
        action="store_true",
        help="Use torchvision FakeData instead of MNIST (offline smoke tests)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.hidden_dims) < 2:
        raise ValueError("MLP model must contain at least 2 hidden layers.")

    if args.run_experiments:
        configs = [
            ("baseline", [256, 128], args.learning_rate),
            ("wide", [512, 256], args.learning_rate),
            ("small_lr", [256, 128], args.learning_rate * 0.5),
        ]
    else:
        configs = [("single_run", args.hidden_dims, args.learning_rate)]

    results = []
    for config_name, hidden_dims, learning_rate in configs:
        metrics = train_single_config(args, config_name, hidden_dims, learning_rate)
        results.append(metrics)
        print(
            f"[{config_name}] test_accuracy={metrics['test_accuracy']:.4f} "
            f"training_time={metrics['training_time_seconds']:.2f}s"
        )

    summary_path = Path(args.output_dir) / "experiment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
