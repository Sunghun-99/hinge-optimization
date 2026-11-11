"""Load one FEM result per hinge geometry in mm and MPa."""

import argparse
import csv
import hashlib
import io
import math
from pathlib import Path

import torch


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--feature-columns", nargs=3, default=("a", "b", "t"),
                        metavar=("A", "B", "T"))
    parser.add_argument("--stress-column", default="sigma_max")
    parser.add_argument("--length-unit", choices=("mm", "m"), default="mm")
    parser.add_argument("--stress-unit", choices=("MPa", "Pa"), required=True)


def read_fem_csv(path: Path, schema: dict) -> tuple[torch.Tensor, torch.Tensor, str]:
    contents = path.read_bytes()
    reader = csv.DictReader(io.StringIO(contents.decode("utf-8-sig"), newline=""))
    columns = [*schema["feature_columns"], schema["stress_column"]]
    if len(set(columns)) != len(columns):
        raise ValueError("Specify three distinct feature columns and one stress column.")
    missing = set(columns) - set(reader.fieldnames or ())
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}.")
    if any(reader.fieldnames.count(column) != 1 for column in columns):
        raise ValueError("Each selected CSV column must occur exactly once.")

    length_factor = {"mm": 1.0, "m": 1_000.0}[schema["length_unit"]]
    stress_factor = {"MPa": 1.0, "Pa": 1e-6}[schema["stress_unit"]]
    geometries = []
    stresses = []
    seen = set()
    for row in reader:
        try:
            a, b, t, stress = (float(row[column]) for column in columns)
        except (TypeError, ValueError) as error:
            raise ValueError(f"CSV line {reader.line_num}: expected numeric values.") from error
        if not all(math.isfinite(value) for value in (a, b, t, stress)):
            raise ValueError(f"CSV line {reader.line_num}: found a non-finite value.")
        if min(a, b, t) <= 0 or stress < 0:
            raise ValueError(f"CSV line {reader.line_num}: invalid geometry or stress.")

        geometry = (a * length_factor, b * length_factor, t * length_factor)
        if geometry in seen:
            raise ValueError(
                f"CSV line {reader.line_num}: repeated geometry {geometry} mm. "
                "Keep one verified FEM stress per design under the same load."
            )
        seen.add(geometry)
        geometries.append(geometry)
        stresses.append(stress * stress_factor)

    if len(geometries) < 10:
        raise ValueError("Provide at least 10 distinct designs for the train/validation/test split.")
    features = torch.tensor(geometries, dtype=torch.float32)
    targets = torch.tensor(stresses, dtype=torch.float32)
    if not torch.isfinite(features).all() or not torch.isfinite(targets).all():
        raise ValueError("Converted CSV values exceed the supported numeric range.")
    return features, targets, hashlib.sha256(contents).hexdigest()


def split_designs(count: int, seed: int) -> dict[str, list[int]]:
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed)).tolist()
    held_out = round(0.15 * count)
    return {
        "test": order[:held_out],
        "validation": order[held_out:2 * held_out],
        "train": order[2 * held_out:],
    }
