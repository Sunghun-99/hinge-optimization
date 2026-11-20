"""Load-controlled stress surrogate built from the team 2 and team 3 models.

The FEM data (team 1) are computed with a prescribed tip displacement delta.
Because the model is linear elastic, the maximum stress under a tip force P
(per unit width) follows from the displacement-controlled results:

    sigma_F(a, b, t; P) = sigma_max(a, b, t) * P / R(a, b, t)

* sigma_max: team 2 MLP (deep_learning/results/team2-fem/mse.pt), MPa
* R:         team 3 MLP trained with the unchanged team 2 code on R / t**3,
             so R = t**3 * g(a, b, t), N/mm per unit width

sigma_F / P (MPa per N/mm) is independent of delta, so one evaluation of the
design grid serves every load P.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from deep_learning.prediction.model import load_surrogate

DEFAULT_CONFIG = Path("config/optimization.json")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def allowable_limit(config: dict) -> float:
    """Stress limit used by the optimizer: (1 - margin) * allowable stress."""
    return (1.0 - config["surrogate_margin"]) * config["allowable_stress_MPa"]


class LoadedHingeSurrogate:
    """sigma_max, R and sigma_F predictions for batches of (a, b, t) in mm."""

    def __init__(self, config: dict) -> None:
        if config.get("reaction_model_target") != "reaction_over_t3":
            raise ValueError("This module expects a reaction model trained on R / t**3.")
        self.stress_model_path = Path(config["stress_model"])
        reaction_dir = Path(config["reaction_model_dir"])
        run = json.loads((reaction_dir / "run.json").read_text(encoding="utf-8"))
        self.reaction_model_name = run["selected_model"]
        self.reaction_model_path = reaction_dir / f"{self.reaction_model_name}.pt"
        self.stress_model = load_surrogate(self.stress_model_path)
        self.reaction_model = load_surrogate(self.reaction_model_path)

    @torch.inference_mode()
    def stress_at_delta(self, features_mm: torch.Tensor) -> torch.Tensor:
        """Maximum von Mises stress at the team 1 tip displacement, MPa."""
        return self.stress_model(features_mm)

    @torch.inference_mode()
    def reaction(self, features_mm: torch.Tensor) -> torch.Tensor:
        """Tip reaction per unit width at the team 1 tip displacement, N/mm."""
        return self.reaction_model(features_mm) * features_mm[..., 2] ** 3

    @torch.inference_mode()
    def stress_per_load(self, features_mm: torch.Tensor) -> torch.Tensor:
        """sigma_F / P in MPa per (N/mm): multiply by P to get sigma_F."""
        return self.stress_at_delta(features_mm) / self.reaction(features_mm)

    @torch.inference_mode()
    def stress_under_load(self, features_mm: torch.Tensor, load_n_per_mm: float) -> torch.Tensor:
        """Maximum stress under tip force P per unit width, MPa."""
        return self.stress_per_load(features_mm) * load_n_per_mm

    def predict_one(self, a_mm: float, b_mm: float, t_mm: float, load_n_per_mm: float) -> dict:
        features = torch.tensor([[a_mm, b_mm, t_mm]], dtype=torch.float32)
        sigma = float(self.stress_at_delta(features)[0])
        reaction = float(self.reaction(features)[0])
        return {
            "sigma_at_delta_MPa": sigma,
            "reaction_N_per_mm": reaction,
            "sigma_under_load_MPa": sigma * load_n_per_mm / reaction,
        }


def fem_stress_under_load(sigma_at_delta, reaction, load_n_per_mm):
    """The same load conversion applied to FEM results (scalars or arrays)."""
    return sigma_at_delta * load_n_per_mm / reaction
