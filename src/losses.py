"""
losses.py
RO-2026-002 — ABPET Threshold
Frontier Translational Research Lab (Independent)
PI: Julian Borges, MD

Loss functions per model group (DEC-005):
    Models A, B, C: L1Loss (MAE) — matches hackathon for anchor comparability
    Models D, E, F: HuberLoss (delta=10.0) — noise robustness for ResNet group
"""

import torch
import torch.nn as nn


class MAELoss(nn.Module):
    """L1 loss. Used for Models A, B, C to match hackathon training."""

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


class HuberLoss(nn.Module):
    """
    Huber loss with delta=10.0 CL.
    Behaves as MAE for errors > delta, MSE for errors < delta.
    Reduces sensitivity to label noise outliers while maintaining
    MAE-scale gradient behavior for most predictions.
    Used for Models D, E, F.
    """

    def __init__(self, delta: float = 10.0):
        super().__init__()
        self.loss = nn.HuberLoss(delta=delta)
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


def get_loss(model_name: str) -> nn.Module:
    """Return correct loss function per model group (DEC-005)."""
    if model_name.upper() in ("A", "B", "C"):
        return MAELoss()
    elif model_name.upper() in ("D", "E", "F"):
        return HuberLoss(delta=10.0)
    else:
        raise ValueError(f"Unknown model name '{model_name}'")
