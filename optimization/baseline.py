"""Team 3: compare the deep-learning surrogate with non-deep-learning baselines.

Methods
* mlp            team 2 stress MLP x team 3 reaction MLP (R / t**3), see surrogates.py
* linear_interp  piecewise-linear interpolation (scipy LinearNDInterpolator) of the
                 same two targets, fitted on the team 2 train + validation designs
* rbf_interp     thin-plate-spline RBF interpolation (scipy RBFInterpolator), same data
* fem_grid       parameter sweep: pick directly from the 330 FEM designs (no model)

All models predict sigma_max and R / t**3 and combine them exactly like the MLP,
so the comparison isolates the regression method. Accuracy is measured on the
team 2 test designs (50) that none of the models saw. Each model then solves the
same optimization on the same design grid with the same stress limit. The FEM
sweep uses the true FEM stress, so it needs no surrogate margin and is checked
against the allowable stress itself.

Usage (from the repository root):
    python -m optimization.baseline [--config config/optimization.json]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator

from .optimize import axis_values, solve_load
from .surrogates import (DEFAULT_CONFIG, LoadedHingeSurrogate, allowable_limit,
                         fem_stress_under_load, load_config)

EVAL_CHUNK = 50_000


def read_fem(csv_path: Path) -> dict[str, np.ndarray]:
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    data = {key: np.array([float(r[col]) for r in rows]) for key, col in (
        ("a", "a_mm"), ("b", "b_mm"), ("t", "t_mm"), ("sigma", "sigma_max_MPa"),
        ("reaction", "tip_reaction_per_width_N_per_mm"), ("seconds", "total_seconds"))}
    data["x"] = np.column_stack([data["a"], data["b"], data["t"]])
    return data


class InterpolatedSurrogate:
    """Interpolates sigma_max and R / t**3 in coordinates scaled to [0, 1]."""

    def __init__(self, kind: str, x: np.ndarray, sigma: np.ndarray, reaction: np.ndarray,
                 bounds: dict) -> None:
        self.low = np.array([bounds[k][0] for k in "abt"])
        self.span = np.array([bounds[k][1] - bounds[k][0] for k in "abt"])
        z = self._scale(x)
        targets = np.column_stack([sigma, reaction / x[:, 2] ** 3])
        if kind == "linear_interp":
            self.main = LinearNDInterpolator(z, targets)
            self.fallback = NearestNDInterpolator(z, targets)
        elif kind == "rbf_interp":
            self.main = RBFInterpolator(z, targets, kernel="thin_plate_spline")
            self.fallback = None
        else:
            raise ValueError(kind)

    def _scale(self, x: np.ndarray) -> np.ndarray:
        return (x - self.low) / self.span

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = np.empty((len(x), 2))
        for start in range(0, len(x), EVAL_CHUNK):
            z = self._scale(x[start:start + EVAL_CHUNK])
            values = self.main(z)
            if self.fallback is not None:
                missing = np.isnan(values).any(axis=1)
                if missing.any():
                    values[missing] = self.fallback(z[missing])
            out[start:start + len(z)] = values
        sigma, g = out[:, 0], out[:, 1]
        return sigma, g * x[:, 2] ** 3

    def stress_per_load(self, x: np.ndarray) -> np.ndarray:
        sigma, reaction = self.predict(x)
        return sigma / reaction


def mlp_stress_per_load(model: LoadedHingeSurrogate, x: np.ndarray, batch: int) -> np.ndarray:
    out = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), batch):
        chunk = torch.from_numpy(x[start:start + batch].astype(np.float32))
        out[start:start + len(chunk)] = model.stress_per_load(chunk).numpy()
    return out


def relative_errors(pred: np.ndarray, true: np.ndarray) -> dict:
    error = np.abs(pred / true - 1.0) * 100.0
    return {"mean_pct": float(error.mean()), "max_pct": float(error.max())}


def fem_grid_solve(fem: dict, load: float, allowable: float) -> dict:
    stress = fem_stress_under_load(fem["sigma"], fem["reaction"], load)
    ok = stress <= allowable
    if not ok.any():
        return {"load_P_N_per_mm": load, "feasible": False}
    t_min = fem["t"][ok].min()
    candidates = np.flatnonzero(ok & np.isclose(fem["t"], t_min))
    best = candidates[np.argmin(stress[candidates])]
    return {"load_P_N_per_mm": load, "feasible": True, "rounded_design_inside_range": True,
            "a_mm": float(fem["a"][best]), "b_mm": float(fem["b"][best]),
            "t_mm": float(fem["t"][best]), "t_star_mm": float(t_min),
            "sigma_F_pred_MPa": float(stress[best])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    limit = allowable_limit(config)
    allowable = config["allowable_stress_MPa"]
    rounding = config["final_t_rounding_mm"]
    loads = [float(p) for p in config["load_sweep_N_per_mm"]]
    base_load = float(config["load_P_N_per_mm"])

    fem = read_fem(Path(config["data_csv"]))
    split = json.loads(Path(config["split_run_json"]).read_text(encoding="utf-8"))["split"]
    fit_idx = np.array(split["train"] + split["validation"])
    test_idx = np.array(split["test"])
    true_test = fem["sigma"][test_idx] / fem["reaction"][test_idx]

    bounds, steps = config["design_bounds_mm"], config["grid_step_mm"]
    a = axis_values(bounds["a"], steps["a"])
    b = axis_values(bounds["b"], steps["b"])
    t = axis_values(bounds["t"], steps["t"])
    grid = np.stack(np.meshgrid(a, b, t, indexing="ij"), axis=-1).reshape(-1, 3)
    shape = (len(a), len(b), len(t))

    results, sweeps = [], {}

    # Deep-learning surrogate (models already trained; training time reported from run.json).
    started = perf_counter()
    mlp = LoadedHingeSurrogate(config)
    fit_seconds = perf_counter() - started
    train_seconds = 0.0
    for run_json in (Path(config["split_run_json"]),
                     Path(config["reaction_model_dir"]) / "run.json"):
        run = json.loads(run_json.read_text(encoding="utf-8"))
        train_seconds += sum(m["training_seconds"] for m in run["models"].values())
    methods = {"mlp": lambda x: mlp_stress_per_load(mlp, x, config["batch_size"])}

    for kind in ("linear_interp", "rbf_interp"):
        started = perf_counter()
        model = InterpolatedSurrogate(kind, fem["x"][fit_idx], fem["sigma"][fit_idx],
                                      fem["reaction"][fit_idx], bounds)
        methods[kind] = (model.stress_per_load, perf_counter() - started)

    for name, entry in methods.items():
        predict, setup = (entry, fit_seconds) if name == "mlp" else entry
        accuracy = relative_errors(predict(fem["x"][test_idx]), true_test)
        started = perf_counter()
        stress_per_load = predict(grid).reshape(shape)
        grid_seconds = perf_counter() - started
        started = perf_counter()
        sweeps[name] = [solve_load(stress_per_load, a, b, t, p, limit, rounding) for p in loads]
        base = solve_load(stress_per_load, a, b, t, base_load, limit, rounding)
        search_seconds = perf_counter() - started
        results.append({
            "method": name,
            "test_sigma_F_rel_error_mean_pct": accuracy["mean_pct"],
            "test_sigma_F_rel_error_max_pct": accuracy["max_pct"],
            "stress_limit_MPa": limit,
            "t_star_mm": base.get("t_star_mm"),
            "a_mm": base.get("a_mm"), "b_mm": base.get("b_mm"), "t_mm": base.get("t_mm"),
            "sigma_F_pred_MPa": base.get("sigma_F_pred_MPa"),
            "fem_data_seconds": float(fem["seconds"].sum()),
            "training_or_fit_seconds": (train_seconds if name == "mlp" else setup),
            "grid_evaluation_seconds": grid_seconds,
            "search_seconds": search_seconds,
            "grid_points": int(len(grid)),
        })

    started = perf_counter()
    sweeps["fem_grid"] = [fem_grid_solve(fem, p, allowable) for p in loads]
    base = fem_grid_solve(fem, base_load, allowable)
    search_seconds = perf_counter() - started
    results.append({
        "method": "fem_grid",
        "test_sigma_F_rel_error_mean_pct": 0.0,
        "test_sigma_F_rel_error_max_pct": 0.0,
        "stress_limit_MPa": allowable,
        "t_star_mm": base.get("t_star_mm"),
        "a_mm": base.get("a_mm"), "b_mm": base.get("b_mm"), "t_mm": base.get("t_mm"),
        "sigma_F_pred_MPa": base.get("sigma_F_pred_MPa"),
        "fem_data_seconds": float(fem["seconds"].sum()),
        "training_or_fit_seconds": 0.0,
        "grid_evaluation_seconds": 0.0,
        "search_seconds": search_seconds,
        "grid_points": int(len(fem["t"])),
    })

    with (out_dir / "baseline_comparison.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with (out_dir / "baseline_sensitivity_P.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        names = list(sweeps)
        writer.writerow(["load_P_N_per_mm", *(f"{n}_t_mm" for n in names),
                         *(f"{n}_t_star_mm" for n in names)])
        for i, load in enumerate(loads):
            writer.writerow([load, *(sweeps[n][i].get("t_mm") for n in names),
                             *(sweeps[n][i].get("t_star_mm") for n in names)])

    if not args.no_plots:
        plot_sweeps(out_dir, loads, sweeps)

    print(f"Accuracy on {len(test_idx)} unseen test designs (sigma_F = sigma * P / R) "
          f"and the selected design at P = {base_load:g} N/mm:")
    print(f"{'method':<14}{'mean err':>10}{'max err':>9}{'t*':>8}{'t':>7}{'a':>7}{'b':>7}"
          f"{'sigma_F':>9}{'fit/train s':>13}{'grid s':>9}")
    for r in results:
        print(f"{r['method']:<14}{r['test_sigma_F_rel_error_mean_pct']:>9.2f}%"
              f"{r['test_sigma_F_rel_error_max_pct']:>8.2f}%{r['t_star_mm']:>8.3f}"
              f"{r['t_mm']:>7.2f}{r['a_mm']:>7.2f}{r['b_mm']:>7.2f}{r['sigma_F_pred_MPa']:>9.1f}"
              f"{r['training_or_fit_seconds']:>13.2f}{r['grid_evaluation_seconds']:>9.2f}")
    print(f"FEM data for all methods: 330 designs, {fem['seconds'].sum():.1f} s of solver time.")
    print(f"Saved: {out_dir / 'baseline_comparison.csv'}")


def plot_sweeps(out_dir: Path, loads: list[float], sweeps: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    ink, ink2 = "#0b0b0b", "#52514e"
    plt.rcParams.update({"axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2,
                         "ytick.color": ink2, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9})
    styles = {
        "mlp": ("#2a78d6", "o", "-", "MLP surrogate (deep learning)"),
        "linear_interp": ("#eb6834", "s", "--", "Linear interpolation"),
        "rbf_interp": ("#1baf7a", "^", ":", "RBF interpolation"),
        "fem_grid": ("#8a8984", "D", "none", "FEM 330-design sweep"),
    }
    fig, ax = plt.subplots(figsize=(6.0, 3.6), layout="constrained")
    for name, rows in sweeps.items():
        color, marker, line, label = styles[name]
        ax.plot(loads, [r.get("t_star_mm", np.nan) if name != "fem_grid" else r.get("t_mm", np.nan)
                        for r in rows], color=color, marker=marker, markersize=5,
                linestyle=line, linewidth=1.8, label=label)
    ax.set_xlabel("Tip load P (N/mm)")
    ax.set_ylabel("Minimum thickness t (mm)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Minimum thickness by method", loc="left", color=ink)
    fig.savefig(out_dir / "baseline_sensitivity_P.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
