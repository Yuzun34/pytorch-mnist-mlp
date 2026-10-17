from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt

from .train import build_arg_parser, run_training


DEFAULT_EXPERIMENTS = [
    {"name": "adam_baseline", "optimizer": "adam", "lr": 1e-3, "hidden_sizes": [256, 128]},
    {"name": "adam_deeper_3layers", "optimizer": "adam", "lr": 1e-3, "hidden_sizes": [512, 256, 128]},
    {"name": "adam_compact", "optimizer": "adam", "lr": 1e-3, "hidden_sizes": [128, 64]},
    {"name": "sgd_baseline_arch", "optimizer": "sgd", "lr": 5e-2, "hidden_sizes": [256, 128]},
    {"name": "sgd_wider", "optimizer": "sgd", "lr": 5e-2, "hidden_sizes": [512, 256]},
]


def save_comparison_plot(results: list[dict], output_dir: Path) -> None:
    if not results:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for item in results:
        label = f"{item['experiment_name']} (test={item['test_accuracy']*100:.2f}%)"
        epochs_range = range(1, len(item["history"]["val_loss"]) + 1)
        axes[0].plot(epochs_range, item["history"]["val_loss"], marker="o", label=label)
        axes[1].plot(epochs_range, item["history"]["val_accuracy"], marker="o", label=label)

    axes[0].set_title("Deneyler Arası Validation Loss Karşılaştırması")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Deneyler Arası Validation Accuracy Karşılaştırması")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_curves.png", dpi=150)
    plt.close(fig)


def run_experiments(base_args, experiments: list[dict], output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for config in experiments:
        args = copy.deepcopy(base_args)
        args.optimizer = config.get("optimizer", args.optimizer)
        args.lr = config.get("lr", args.lr)
        args.hidden_sizes = config.get("hidden_sizes", args.hidden_sizes)
        args.output_dir = str(output_dir / config["name"])
        summary = run_training(args)
        summary["experiment_name"] = config["name"]
        results.append(summary)

    (output_dir / "experiment_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_comparison_plot(results, output_dir)
    return results


def main() -> None:
    parser = build_arg_parser()
    parser.description = "Run configurable MNIST MLP hyperparameter experiments."
    parser.add_argument("--experiments-json", default="", help="Path to experiment config JSON file.")
    parser.add_argument("--experiments-output-dir", default="./outputs/experiments")
    args = parser.parse_args()

    experiments = DEFAULT_EXPERIMENTS
    if args.experiments_json:
        experiments = json.loads(Path(args.experiments_json).read_text(encoding="utf-8"))

    results = run_experiments(args, experiments, Path(args.experiments_output_dir))
    compact = [
        {
            "name": item["experiment_name"],
            "train_loss": item["history"]["train_loss"][-1],
            "val_loss": item["history"]["val_loss"][-1],
            "train_accuracy": item["history"]["train_accuracy"][-1],
            "val_accuracy": item["history"]["val_accuracy"][-1],
            "test_loss": item["test_loss"],
            "test_accuracy": item["test_accuracy"],
            "training_time_seconds": item["training_time_seconds"],
        }
        for item in results
    ]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
