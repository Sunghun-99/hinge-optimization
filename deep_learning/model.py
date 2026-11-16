"""Small MLP for (a, b, t) in mm to maximum stress in MPa."""

from pathlib import Path

import torch
from torch import nn


class StressSurrogate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        self.register_buffer("feature_mean", torch.zeros(3))
        self.register_buffer("feature_scale", torch.ones(3))
        self.register_buffer("stress_mean", torch.tensor(0.0))
        self.register_buffer("stress_scale", torch.tensor(1.0))

    @torch.no_grad()
    def fit_scaling(self, features: torch.Tensor, stress: torch.Tensor) -> None:
        feature_std = features.std(dim=0, unbiased=False)
        stress_std = stress.std(unbiased=False)
        self.feature_mean.copy_(features.mean(dim=0))
        self.feature_scale.copy_(torch.where(feature_std > 0, feature_std, 1.0))
        self.stress_mean.copy_(stress.mean())
        self.stress_scale.copy_(torch.where(stress_std > 0, stress_std, 1.0))

    def standardized_prediction(self, features_mm: torch.Tensor) -> torch.Tensor:
        normalized = (features_mm - self.feature_mean) / self.feature_scale
        return self.network(normalized).squeeze(-1)

    def forward(self, features_mm: torch.Tensor) -> torch.Tensor:
        return self.standardized_prediction(features_mm) * self.stress_scale + self.stress_mean

    @torch.inference_mode()
    def predict_stress(self, a_mm: float, b_mm: float, t_mm: float) -> float:
        features = self.feature_mean.new_tensor([a_mm, b_mm, t_mm])
        return float(self(features))


def load_surrogate(checkpoint: str | Path) -> StressSurrogate:
    model = StressSurrogate()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.eval()
