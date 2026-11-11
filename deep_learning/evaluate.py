"""Evaluate both trained models on the held-out designs and plot their learning curves."""

import argparse
import csv
import json
import math
from pathlib import Path
from time import perf_counter

import matplotlib
import torch
from matplotlib import pyplot as plt

from .data import read_fem_csv
from .model import load_surrogate


def plot_learning_curves(run_dir: Path, names: list[str]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained", sharey=True)
    for name in names:
        with (run_dir / f"{name}_history.csv").open(encoding="utf-8", newline="") as file:
            history = list(csv.DictReader(file))
        epochs = [int(row["epoch"]) for row in history]
        for axis, split in zip(axes, ("train", "validation")):
            errors = [float(row[f"{split}_mae_mpa"]) for row in history]
            axis.plot(epochs, errors, label=name.upper())
            axis.set_title(split.capitalize())
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)
            axis.legend()
    axes[0].set_ylabel("Mean absolute error (MPa)")
    figure.savefig(run_dir / "learning_curves.png", dpi=160)
    plt.close(figure)


@torch.inference_mode()
def evaluate(run_dir: Path, csv_path: Path, allowable_mpa: float | None) -> dict:
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    features, stress, digest = read_fem_csv(csv_path, run["schema"])
    if digest != run["dataset_sha256"]:
        raise ValueError("Use the same CSV file that was used to train these models.")
    test_indices = run["split"]["test"]
    test_x, test_y = features[test_indices], stress[test_indices]
    metrics = {
        "test_designs": len(test_indices),
        "selected_model": run["selected_model"],
        "allowable_mpa": allowable_mpa,
        "models": {},
    }
    predictions = {}
    for name in run["models"]:
        model = load_surrogate(run_dir / f"{name}.pt")
        started = perf_counter()
        predicted = model(test_x)
        elapsed = perf_counter() - started
        if not torch.isfinite(predicted).all():
            raise ValueError(f"{name}: predictions contain non-finite values.")
        predictions[name] = predicted.tolist()
        error = predicted.double() - test_y.double()
        result = {
            "mae_mpa": error.abs().mean().item(),
            "rmse_mpa": error.square().mean().sqrt().item(),
            "max_absolute_error_mpa": error.abs().max().item(),
            "test_batch_inference_seconds": elapsed,
        }
        if allowable_mpa is not None:
            predicted_feasible = predicted <= allowable_mpa
            result["predicted_feasible"] = int(predicted_feasible.sum())
            result["false_feasible"] = int((predicted_feasible & (test_y > allowable_mpa)).sum())
        metrics["models"][name] = result

    names = list(predictions)
    with (run_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("a_mm", "b_mm", "t_mm", "fem_mpa",
                         *(f"{name}_prediction_mpa" for name in names)))
        for index, (geometry, measured) in enumerate(zip(test_x.tolist(), test_y.tolist())):
            writer.writerow((*geometry, measured, *(predictions[name][index] for name in names)))
    plot_learning_curves(run_dir, names)
    (run_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--allowable-mpa", type=float)
    args = parser.parse_args()
    if args.allowable_mpa is not None and (
        not math.isfinite(args.allowable_mpa) or args.allowable_mpa <= 0
    ):
        parser.error("--allowable-mpa must be positive and finite")
    matplotlib.use("Agg")
    try:
        metrics = evaluate(args.run_dir, args.csv, args.allowable_mpa)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
