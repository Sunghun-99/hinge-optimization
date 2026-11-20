"""Team 3: minimum-thickness hinge design with the surrogate models.

Problem (config/optimization.json):
    minimize    t
    subject to  sigma_F(a, b, t; P) = sigma_max * P / R <= (1 - margin) * sigma_allow
                a, b, t inside the team 1 design range (no extrapolation)
    tie-break   at the chosen t, the (a, b) with the lowest sigma_F

The design grid is evaluated once (sigma_F / P does not depend on P), then every
load in the sweep is solved from the same array. The final thickness is rounded
up to the manufacturing step (0.01 mm by default).

Usage (from the repository root):
    python -m optimization.optimize [--config config/optimization.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .surrogates import (DEFAULT_CONFIG, LoadedHingeSurrogate, allowable_limit,
                         fem_stress_under_load, load_config)

DECIMALS = 6


def axis_values(bounds: list[float], step: float) -> np.ndarray:
    low, high = bounds
    count = int(round((high - low) / step)) + 1
    values = np.round(low + step * np.arange(count), DECIMALS)
    if abs(values[-1] - high) > 1e-9:
        raise ValueError(f"Step {step} does not divide the range {bounds}.")
    return values


def evaluate_grid(model: LoadedHingeSurrogate, a: np.ndarray, b: np.ndarray, t: np.ndarray,
                  batch_size: int) -> np.ndarray:
    """sigma_F / P on the full grid, shape (len(a), len(b), len(t))."""
    grid = np.stack(np.meshgrid(a, b, t, indexing="ij"), axis=-1).reshape(-1, 3)
    features = torch.from_numpy(grid.astype(np.float32))
    out = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), batch_size):
        chunk = features[start:start + batch_size]
        out[start:start + len(chunk)] = model.stress_per_load(chunk).numpy()
    return out.reshape(len(a), len(b), len(t))


def best_ab_at(stress_per_load: np.ndarray, t_index: int) -> tuple[int, int]:
    plane = stress_per_load[:, :, t_index]
    i, j = np.unravel_index(int(np.argmin(plane)), plane.shape)
    return int(i), int(j)


def solve_load(stress_per_load: np.ndarray, a: np.ndarray, b: np.ndarray, t: np.ndarray,
               load: float, limit: float, rounding: float) -> dict:
    """Minimum feasible t for one load, then the rounded-up design."""
    feasible_any = (stress_per_load.min(axis=(0, 1)) * load) <= limit
    if not feasible_any.any():
        return {"load_P_N_per_mm": load, "feasible": False}

    k_star = int(np.argmax(feasible_any))
    i_star, j_star = best_ab_at(stress_per_load, k_star)
    t_star = float(t[k_star])

    t_step = float(t[1] - t[0])
    t_final = round(math.ceil(t_star / rounding - 1e-9) * rounding, DECIMALS)
    k_final = int(round((t_final - t[0]) / t_step))
    # Feasibility is not assumed monotonic in t: move up until the design passes.
    while k_final < len(t) and not feasible_any[k_final]:
        k_final = int(round((round(float(t[k_final]) + rounding, DECIMALS) - t[0]) / t_step))
    if k_final >= len(t):
        return {"load_P_N_per_mm": load, "feasible": True, "t_star_mm": t_star,
                "rounded_design_inside_range": False}
    i_final, j_final = best_ab_at(stress_per_load, k_final)

    return {
        "load_P_N_per_mm": load,
        "feasible": True,
        "t_star_mm": t_star,
        "a_at_t_star_mm": float(a[i_star]),
        "b_at_t_star_mm": float(b[j_star]),
        "sigma_F_at_t_star_MPa": float(stress_per_load[i_star, j_star, k_star] * load),
        "rounded_design_inside_range": True,
        "a_mm": float(a[i_final]),
        "b_mm": float(b[j_final]),
        "t_mm": float(t[k_final]),
        "sigma_F_pred_MPa": float(stress_per_load[i_final, j_final, k_final] * load),
        "feasible_ab_share_at_t": float(
            np.mean(stress_per_load[:, :, k_final] * load <= limit)),
    }


def fem_grid_table(csv_path: Path) -> dict[str, np.ndarray]:
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return {
        "t": np.array([float(r["t_mm"]) for r in rows]),
        "sigma": np.array([float(r["sigma_max_MPa"]) for r in rows]),
        "reaction": np.array([float(r["tip_reaction_per_width_N_per_mm"]) for r in rows]),
    }


def plot_results(out_dir: Path, sweep: list[dict], stress_per_load: np.ndarray, t: np.ndarray,
                 fem: dict, base: dict, config: dict, limit: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    blue, orange, gray = "#2a78d6", "#eb6834", "#8a8984"
    ink, ink2 = "#0b0b0b", "#52514e"
    plt.rcParams.update({"axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2,
                         "ytick.color": ink2, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9})

    # 1) Minimum thickness against load.
    solved = [row for row in sweep if row.get("rounded_design_inside_range")]
    loads = [row["load_P_N_per_mm"] for row in solved]
    fig, ax = plt.subplots(figsize=(6.0, 3.6), layout="constrained")
    ax.plot(loads, [row["t_star_mm"] for row in solved], color=blue, linewidth=2,
            marker="o", markersize=5, label="Surrogate t* (continuous)")
    ax.scatter(loads, [row["t_mm"] for row in solved], color=orange, marker="s", s=36,
               zorder=4, label=f"Final t (rounded up to {config['final_t_rounding_mm']} mm)")
    fem_min_t = []
    for load in loads:
        ok = (fem_stress_under_load(fem["sigma"], fem["reaction"], load)
              <= config["allowable_stress_MPa"])
        fem_min_t.append(fem["t"][ok].min() if ok.any() else np.nan)
    ax.scatter(loads, fem_min_t, color=gray, s=30, zorder=3,
               label="FEM 330-design grid (allowable, no margin)")
    ax.set_xlabel("Tip load P (N/mm)")
    ax.set_ylabel("Minimum thickness t (mm)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Minimum feasible hinge thickness vs load", loc="left", color=ink)
    fig.savefig(out_dir / "sensitivity_P.png", dpi=180)
    plt.close(fig)

    # 2) Stress against thickness at the base load.
    load = config["load_P_N_per_mm"]
    fig, ax = plt.subplots(figsize=(6.0, 3.8), layout="constrained")
    lowest = stress_per_load.min(axis=(0, 1)) * load
    highest = stress_per_load.max(axis=(0, 1)) * load
    ax.fill_between(t, lowest, highest, color=blue, alpha=0.18, linewidth=0,
                    label="Surrogate range over (a, b)")
    ax.plot(t, lowest, color=blue, linewidth=2, label="Surrogate, best (a, b) per t")
    ax.scatter(fem["t"], fem_stress_under_load(fem["sigma"], fem["reaction"], load),
               color=gray, s=10, alpha=0.7, zorder=3, label="FEM (330 designs)")
    ax.axhline(config["allowable_stress_MPa"], color=ink, linewidth=1, linestyle="--")
    ax.axhline(limit, color=ink2, linewidth=1, linestyle=":")
    ax.text(t[-1], config["allowable_stress_MPa"], "allowable 236.7 MPa", ha="right",
            va="bottom", color=ink, fontsize=8)
    ax.text(t[-1], limit, f"limit with {config['surrogate_margin']:.0%} margin {limit:.1f} MPa",
            ha="right", va="top", color=ink2, fontsize=8)
    if base.get("rounded_design_inside_range"):
        ax.axvline(base["t_mm"], color=orange, linewidth=1.5)
        ax.annotate(f"final t = {base['t_mm']:.2f} mm\n(a = {base['a_mm']:.2f}, b = {base['b_mm']:.2f})",
                    (base["t_mm"], base["sigma_F_pred_MPa"]), xytext=(12, 30),
                    textcoords="offset points", color=ink, fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": ink2, "linewidth": 0.8})
    ax.set_xlabel("Minimum thickness t (mm)")
    ax.set_ylabel(f"Max von Mises stress at P = {load:g} N/mm (MPa)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Load-controlled stress vs thickness", loc="left", color=ink)
    fig.savefig(out_dir / "stress_vs_t.png", dpi=180)
    plt.close(fig)


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
    bounds, steps = config["design_bounds_mm"], config["grid_step_mm"]
    a = axis_values(bounds["a"], steps["a"])
    b = axis_values(bounds["b"], steps["b"])
    t = axis_values(bounds["t"], steps["t"])

    started = perf_counter()
    model = LoadedHingeSurrogate(config)
    load_seconds = perf_counter() - started

    started = perf_counter()
    stress_per_load = evaluate_grid(model, a, b, t, config["batch_size"])
    grid_seconds = perf_counter() - started

    started = perf_counter()
    sweep = [solve_load(stress_per_load, a, b, t, float(p), limit,
                        config["final_t_rounding_mm"]) for p in config["load_sweep_N_per_mm"]]
    base = solve_load(stress_per_load, a, b, t, float(config["load_P_N_per_mm"]), limit,
                      config["final_t_rounding_mm"])
    search_seconds = perf_counter() - started

    if base.get("rounded_design_inside_range"):
        base["surrogate_check"] = model.predict_one(base["a_mm"], base["b_mm"], base["t_mm"],
                                                    config["load_P_N_per_mm"])

    result = {
        "problem": {
            "objective": "minimize t",
            "constraint": "sigma_max * P / R <= (1 - margin) * allowable",
            "tie_break": "lowest sigma_F at the chosen t",
            "load_P_N_per_mm": config["load_P_N_per_mm"],
            "allowable_stress_MPa": config["allowable_stress_MPa"],
            "surrogate_margin": config["surrogate_margin"],
            "stress_limit_MPa": limit,
        },
        "design": base,
        "grid": {
            "points": int(stress_per_load.size),
            "a_count": len(a), "b_count": len(b), "t_count": len(t),
            "steps_mm": steps, "bounds_mm": bounds,
        },
        "timing_seconds": {
            "load_models": load_seconds,
            "grid_evaluation": grid_seconds,
            "search_all_loads": search_seconds,
            "total": load_seconds + grid_seconds + search_seconds,
        },
        "models": {
            "stress": str(model.stress_model_path).replace("\\", "/"),
            "reaction": str(model.reaction_model_path).replace("\\", "/"),
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "processor": platform.processor() or platform.machine(),
            "torch_threads": torch.get_num_threads(),
        },
    }
    (out_dir / "optimal_design.json").write_text(json.dumps(result, indent=2) + "\n",
                                                  encoding="utf-8")

    columns = ["load_P_N_per_mm", "feasible", "t_star_mm", "a_at_t_star_mm", "b_at_t_star_mm",
               "sigma_F_at_t_star_MPa", "rounded_design_inside_range", "a_mm", "b_mm", "t_mm",
               "sigma_F_pred_MPa", "feasible_ab_share_at_t"]
    with (out_dir / "sensitivity_P.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(sweep)

    if not args.no_plots:
        plot_results(out_dir, sweep, stress_per_load, t, fem_grid_table(Path(config["data_csv"])),
                     base, config, limit)

    print(f"Grid: {stress_per_load.size:,} designs evaluated in {grid_seconds:.2f} s "
          f"(search over {len(sweep)} loads: {search_seconds:.3f} s)")
    print(f"Stress limit: {limit:.1f} MPa ({config['surrogate_margin']:.0%} margin)")
    for row in sweep:
        if row.get("rounded_design_inside_range"):
            print(f"  P = {row['load_P_N_per_mm']:>4} N/mm: t* = {row['t_star_mm']:.3f} mm -> "
                  f"t = {row['t_mm']:.2f}, a = {row['a_mm']:.2f}, b = {row['b_mm']:.2f}, "
                  f"sigma_F = {row['sigma_F_pred_MPa']:.1f} MPa")
        else:
            print(f"  P = {row['load_P_N_per_mm']:>4} N/mm: no design inside the range")
    if base.get("rounded_design_inside_range"):
        print(f"Selected design (P = {config['load_P_N_per_mm']} N/mm): a = {base['a_mm']:.2f}, "
              f"b = {base['b_mm']:.2f}, t = {base['t_mm']:.2f} mm, "
              f"predicted sigma_F = {base['sigma_F_pred_MPa']:.1f} MPa")
    print(f"Saved: {out_dir / 'optimal_design.json'}")


if __name__ == "__main__":
    main()
