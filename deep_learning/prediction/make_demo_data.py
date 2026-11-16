"""Generate 330 synthetic designs for testing the training and inference pipeline."""

import argparse
import csv
from itertools import product
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("a", "b", "t", "sigma_max", "source"))
        for i, j, k in product(range(5), range(6), range(11)):
            a = 2.0 + 0.25 * i
            b = 0.8 + 0.2 * j
            t = 0.5 + 0.1 * k
            # A synthetic response for exercising the code, with no FEM calibration.
            stress = 30.0 + (12.0 + 2.0 * a + b**2) / t**2
            writer.writerow((round(a, 4), round(b, 4), round(t, 4),
                             round(stress, 6), "synthetic_demo"))
    print(f"Wrote 330 synthetic designs to {args.output}.")


if __name__ == "__main__":
    main()
