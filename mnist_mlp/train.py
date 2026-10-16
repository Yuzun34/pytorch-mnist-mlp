from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from .data import build_mnist_loaders, get_device, set_seed
from .model import MLPClassifier


def evaluate(model: nn.Module, data_loader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train_model(model: nn.Module, train_loader, val_loader, epochs: int, lr: float, optimizer_name: str, device: torch.device):
    criterion = nn.CrossEntropyLoss()
    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("optimizer must be 'sgd' or 'adam'")

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

    training_time_seconds = time.perf_counter() - start
    return history, training_time_seconds, criterion


def save_plots(history: dict[str, list[float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png")
    plt.close()


def run_training(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = get_device()

    train_loader, val_loader, test_loader = build_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
    )

    model = MLPClassifier(hidden_sizes=tuple(args.hidden_sizes)).to(device)
    history, training_time_seconds, criterion = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        optimizer_name=args.optimizer,
        device=device,
    )

    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_plots(history, output_dir)

    summary = {
        "seed": args.seed,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "hidden_sizes": args.hidden_sizes,
        "val_split": args.val_split,
        "training_time_seconds": training_time_seconds,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "history": history,
    }

    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.save_checkpoint:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": {
                "hidden_sizes": args.hidden_sizes,
                "input_size": 28 * 28,
                "num_classes": 10,
            },
        }
        torch.save(checkpoint, output_dir / "checkpoint.pt")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MNIST MLP from scratch.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./outputs/run")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-checkpoint", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_training(args)
    print(json.dumps({k: summary[k] for k in ["training_time_seconds", "test_loss", "test_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
