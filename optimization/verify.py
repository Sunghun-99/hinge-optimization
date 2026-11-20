"""Team 3: re-verify optimized designs with the team 1 warp.fem solver.

For each design the team 1 script data/hinge_fem.py is run unchanged, with the
team 1 material and load settings passed explicitly (its built-in defaults are
test values). The FEM stress and reaction are converted to the load-controlled
stress sigma_F = sigma_max * P / R and compared with the surrogate prediction.

Checks per design
* FEM sigma_F <= allowable stress (236.667 MPa)            -> design is safe
* |prediction / FEM - 1| <= max_prediction_error_pct        -> surrogate trusted
* mesh refinement (factors 1.6, 1.0, 0.65): change of sigma_F
  from factor 1.0 to 0.65 <= max_mesh_change_pct              (selected design only)

Usage (from the repository root):
    python -m optimization.verify                 # selected design + mesh refinement
    python -m optimization.verify --sweep         # also every design of the P sweep
    python -m optimization.verify --design 3.12 1.66 0.70 --load 3
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from .surrogates import DEFAULT_CONFIG, LoadedHingeSurrogate, fem_stress_under_load, load_config

FEM_SCRIPT = Path("data/hinge_fem.py")


def run_fem(a: float, b: float, t: float, config: dict, out_dir: Path, *, refine: bool = False,
            plots: bool = False) -> tuple[dict | list[dict], float]:
    fem = config["fem"]
    command = [sys.executable, str(FEM_SCRIPT), "--a", f"{a:g}", "--b", f"{b:g}", "--t", f"{t:g}",
               "--E", f"{fem['E_MPa']:g}", "--nu", f"{fem['nu']:g}",
               "--delta", f"{fem['delta_mm']:g}", "--L", f"{fem['L_mm']:g}",
               "--device", fem["device"], "--out", str(out_dir)]
    if refine:
        command.append("--refine")
    elif plots:
        command.append("--plots")
    started = perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    seconds = perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(f"FEM run failed for a={a}, b={b}, t={t}:\n{completed.stderr[-2000:]}")
    if refine:
        with (out_dir / "convergence.csv").open(encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file)), seconds
    return json.loads((out_dir / "single_result.json").read_text(encoding="utf-8")), seconds


def check_design(model: LoadedHingeSurrogate, config: dict, a: float, b: float, t: float,
                 load: float, out_dir: Path, plots: bool) -> dict:
    predicted = model.predict_one(a, b, t, load)
    fem, seconds = run_fem(a, b, t, config, out_dir, plots=plots)
    fem_sigma_f = fem_stress_under_load(fem["sigma_max_MPa"],
                                        fem["tip_reaction_per_width_N_per_mm"], load)
    error_pct = (predicted["sigma_under_load_MPa"] / fem_sigma_f - 1.0) * 100.0
    limit_pct = config["verification"]["max_prediction_error_pct"]
    allowable = config["allowable_stress_MPa"]
    return {
        "load_P_N_per_mm": load, "a_mm": a, "b_mm": b, "t_mm": t,
        "pred_sigma_at_delta_MPa": predicted["sigma_at_delta_MPa"],
        "fem_sigma_at_delta_MPa": fem["sigma_max_MPa"],
        "pred_reaction_N_per_mm": predicted["reaction_N_per_mm"],
        "fem_reaction_N_per_mm": fem["tip_reaction_per_width_N_per_mm"],
        "pred_sigma_F_MPa": predicted["sigma_under_load_MPa"],
        "fem_sigma_F_MPa": fem_sigma_f,
        "prediction_error_pct": error_pct,
        "safety_margin_pct": (1.0 - fem_sigma_f / allowable) * 100.0,
        "fem_below_allowable": bool(fem_sigma_f <= allowable),
        "prediction_within_tolerance": bool(abs(error_pct) <= limit_pct),
        "triangles": fem["triangle_count"],
        "fem_seconds": seconds,
    }


def check_mesh(config: dict, a: float, b: float, t: float, load: float, out_dir: Path) -> dict:
    rows, seconds = run_fem(a, b, t, config, out_dir, refine=True)
    levels = []
    for row in rows:
        sigma = float(row["sigma_max_MPa"])
        reaction = float(row["tip_reaction_per_width_N_per_mm"])
        levels.append({"mesh_factor": float(row["mesh_factor"]),
                       "triangles": int(row["triangle_count"]),
                       "sigma_at_delta_MPa": sigma, "reaction_N_per_mm": reaction,
                       "sigma_F_MPa": fem_stress_under_load(sigma, reaction, load)})
    by_factor = {round(level["mesh_factor"], 3): level for level in levels}
    base, fine = by_factor[1.0], by_factor[0.65]
    change = abs(fine["sigma_F_MPa"] / base["sigma_F_MPa"] - 1.0) * 100.0
    return {"levels": levels, "change_1.0_to_0.65_pct": change,
            "converged": bool(change <= config["verification"]["max_mesh_change_pct"]),
            "fine_mesh_below_allowable": bool(fine["sigma_F_MPa"] <= config["allowable_stress_MPa"]),
            "fem_seconds": seconds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--design", nargs=3, type=float, metavar=("A", "B", "T"),
                        help="Verify this design instead of optimal_design.json")
    parser.add_argument("--load", type=float, help="Load P for --design (default: config)")
    parser.add_argument("--sweep", action="store_true",
                        help="Also verify every design in sensitivity_P.csv")
    parser.add_argument("--no-refine", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = Path(config["output_dir"])
    out_root = results_dir / config["verification"]["output_subdir"]
    out_root.mkdir(parents=True, exist_ok=True)
    model = LoadedHingeSurrogate(config)

    if args.design:
        a, b, t = args.design
        load = args.load if args.load is not None else float(config["load_P_N_per_mm"])
    else:
        design = json.loads((results_dir / "optimal_design.json").read_text(encoding="utf-8"))["design"]
        if not design.get("rounded_design_inside_range"):
            raise SystemExit("optimal_design.json has no design inside the range.")
        a, b, t, load = design["a_mm"], design["b_mm"], design["t_mm"], design["load_P_N_per_mm"]

    summary = {"allowable_stress_MPa": config["allowable_stress_MPa"],
               "fem_settings": config["fem"], "tolerances": config["verification"]}
    print(f"Verifying a = {a:.2f}, b = {b:.2f}, t = {t:.2f} mm at P = {load:g} N/mm ...")
    summary["selected"] = check_design(model, config, a, b, t, load, out_root / "selected", True)
    s = summary["selected"]
    print(f"  FEM sigma_F = {s['fem_sigma_F_MPa']:.1f} MPa, predicted {s['pred_sigma_F_MPa']:.1f} MPa "
          f"(error {s['prediction_error_pct']:+.2f}%), margin to allowable {s['safety_margin_pct']:.1f}% "
          f"[{'PASS' if s['fem_below_allowable'] else 'FAIL'}]")

    if not args.no_refine:
        print("Mesh refinement (factors 1.6, 1.0, 0.65) ...")
        summary["mesh_refinement"] = check_mesh(config, a, b, t, load, out_root / "selected_refine")
        m = summary["mesh_refinement"]
        for level in m["levels"]:
            print(f"  factor {level['mesh_factor']:<4}: {level['triangles']:>6} triangles, "
                  f"sigma_F = {level['sigma_F_MPa']:.1f} MPa")
        print(f"  change 1.0 -> 0.65: {m['change_1.0_to_0.65_pct']:.2f}% "
              f"[{'PASS' if m['converged'] else 'CHECK'}]")

    if args.sweep:
        with (results_dir / "sensitivity_P.csv").open(encoding="utf-8", newline="") as file:
            sweep = [r for r in csv.DictReader(file) if r["rounded_design_inside_range"] == "True"]
        rows = []
        print("Load sweep designs ...")
        for r in sweep:
            p = float(r["load_P_N_per_mm"])
            row = check_design(model, config, float(r["a_mm"]), float(r["b_mm"]), float(r["t_mm"]),
                               p, out_root / f"sweep_P{p:g}", False)
            rows.append(row)
            print(f"  P = {p:>4}: t = {row['t_mm']:.2f}, FEM sigma_F = {row['fem_sigma_F_MPa']:.1f} MPa, "
                  f"error {row['prediction_error_pct']:+.2f}% "
                  f"[{'PASS' if row['fem_below_allowable'] else 'FAIL'}]")
        with (out_root / "sweep_verification.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary["sweep"] = {"designs": len(rows),
                            "all_below_allowable": all(r["fem_below_allowable"] for r in rows),
                            "max_abs_prediction_error_pct": max(abs(r["prediction_error_pct"])
                                                                for r in rows)}

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    passed = s["fem_below_allowable"] and s["prediction_within_tolerance"]
    if "mesh_refinement" in summary:
        passed = passed and summary["mesh_refinement"]["fine_mesh_below_allowable"]
    print(f"Overall: {'PASS' if passed else 'CHECK RESULTS'} -> {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
