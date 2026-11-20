"""Build the reaction-force training table for the team 3 surrogate.

The tip reaction R of the hinge grows roughly with t**3 (beam bending stiffness),
so R varies by a factor of about 14 across the design space. Training the team 2
MLP on R directly gives large relative errors for thin hinges, which are the
designs the optimizer cares about. This script writes R / t**3 instead, a target
with a much narrower range, and keeps the row order of the team 1 CSV so the
team 2 train/validation/test split (seed 42) selects the same designs.

Usage (from the repository root):
    python -m optimization.make_reaction_data
"""

import argparse
import csv
from pathlib import Path

REACTION_COLUMN = "tip_reaction_per_width_N_per_mm"
TARGET_COLUMN = "reaction_over_t3_N_per_mm4"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=Path("data/hinge_330_full.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/optimization/reaction_dataset.csv"))
    args = parser.parse_args()

    with args.csv.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit(f"No rows in {args.csv}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("a_mm", "b_mm", "t_mm", REACTION_COLUMN, TARGET_COLUMN))
        for row in rows:
            if row.get("status", "ok") != "ok":
                raise SystemExit(f"Unexpected status in {args.csv}: {row}")
            t = float(row["t_mm"])
            reaction = float(row[REACTION_COLUMN])
            writer.writerow((row["a_mm"], row["b_mm"], row["t_mm"], row[REACTION_COLUMN],
                             repr(reaction / t**3)))
    print(f"Wrote {len(rows)} designs to {args.out}")


if __name__ == "__main__":
    main()
