"""Compare MSE and MAE with identical data, initialization, and training settings."""

import argparse
import csv
import json
import math
import platform
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import add_data_arguments, read_fem_csv, split_designs
from .model import StressSurrogate

LOSS_FUNCTIONS = {"mse": nn.MSELoss, "mae": nn.L1Loss}


def train_one(loss_name, features, stress, split, args) -> dict:
    started = perf_counter()
    torch.manual_seed(args.seed)
    train_x, train_y = features[split["train"]], stress[split["train"]]
    val_x, val_y = features[split["validation"]], stress[split["validation"]]

    model = StressSurrogate()
    model.fit_scaling(train_x, train_y)
    train_z = (train_y - model.stress_mean) / model.stress_scale
    batches = DataLoader(
        TensorDataset(train_x, train_z),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loss_fn = LOSS_FUNCTIONS[loss_name]()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_mae = float("inf")
    best_epoch = 0
    checkpoint = args.output_dir / f"{loss_name}.pt"
    history_path = args.output_dir / f"{loss_name}_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as file:
        history = csv.writer(file)
        history.writerow(("epoch", "train_mae_mpa", "validation_mae_mpa"))
        for epoch in range(1, args.epochs + 1):
            model.train()
            for batch_x, batch_z in batches:
                optimizer.zero_grad()
                loss = loss_fn(model.standardized_prediction(batch_x), batch_z)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                train_mae = (model(train_x) - train_y).abs().mean().item()
                val_mae = (model(val_x) - val_y).abs().mean().item()
            if not all(math.isfinite(metric) for metric in (train_mae, val_mae)):
                raise ValueError(f"{loss_name}: training diverged at epoch {epoch}. Try a lower --lr.")
            history.writerow((epoch, train_mae, val_mae))
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint)
            if epoch - best_epoch >= args.patience:
                break

    return {
        "best_epoch": best_epoch,
        "validation_mae_mpa": best_val_mae,
        "training_seconds": perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1 or not 0 < args.lr < float("inf"):
        parser.error("epochs, batch size, patience, and learning rate must be positive and finite")

    schema = {
        "feature_columns": args.feature_columns,
        "stress_column": args.stress_column,
        "length_unit": args.length_unit,
        "stress_unit": args.stress_unit,
    }
    try:
        features, stress, digest = read_fem_csv(args.csv, schema)
        split = split_designs(len(features), args.seed)
        args.output_dir.mkdir(parents=True)
        results = {
            name: train_one(name, features, stress, split, args)
            for name in LOSS_FUNCTIONS
        }
    except (OSError, ValueError) as error:
        parser.error(str(error))

    metadata = {
        "dataset": args.csv.name,
        "dataset_sha256": digest,
        "schema": schema,
        "split": split,
        "seed": args.seed,
        "max_epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "learning_rate": args.lr,
        "optimizer": "Adam",
        "pytorch": torch.__version__,
        "python": platform.python_version(),
        "device": "cpu",
        "processor": platform.processor() or platform.machine(),
        "models": results,
        "selected_model": min(results, key=lambda name: results[name]["validation_mae_mpa"]),
    }
    pending_metadata = args.output_dir / "run.json.tmp"
    pending_metadata.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    pending_metadata.replace(args.output_dir / "run.json")
    for name, result in results.items():
        print(f"{name.upper()}: validation MAE {result['validation_mae_mpa']:.4f} MPa "
              f"at epoch {result['best_epoch']} ({result['training_seconds']:.2f} s)")
    print(f"Selected by validation MAE: {metadata['selected_model']}.")


if __name__ == "__main__":
    main()
