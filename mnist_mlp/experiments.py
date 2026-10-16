from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .train import build_arg_parser, run_training


DEFAULT_EXPERIMENTS = [
    {"name": "adam_baseline", "optimizer": "adam", "lr": 1e-3, "hidden_sizes": [256, 128]},
    {"name": "sgd_wider", "optimizer": "sgd", "lr": 5e-2, "hidden_sizes": [512, 256]},
]


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
            "test_accuracy": item["test_accuracy"],
            "training_time_seconds": item["training_time_seconds"],
        }
        for item in results
    ]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
