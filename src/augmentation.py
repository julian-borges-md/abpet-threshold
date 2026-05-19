"""
augmentation.py
RO-2026-002 — ABPET Threshold
Frontier Translational Research Lab (Independent)
PI: Julian Borges, MD

3D augmentation suite for Models C, D, E, F.
Models A and B use no augmentation (hackathon baseline).
"""

import torch
import torch.nn.functional as F
import numpy as np


class RandomFlip3D:
    """Random flip along each spatial axis independently."""

    def __init__(self, axes=(1, 2, 3), p=0.5):
        self.axes = axes
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for axis in self.axes:
            if torch.rand(1).item() < self.p:
                x = torch.flip(x, dims=[axis])
        return x


class RandomRotation3D:
    """
    Random rotation via affine grid sampling.
    Small rotations (+-10 degrees) around each axis independently.
    """

    def __init__(self, max_degrees=10.0):
        self.max_rad = max_degrees * (3.14159 / 180.0)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        angles = (torch.rand(3) * 2 - 1) * self.max_rad
        rx, ry, rz = angles

        # Build 3D rotation matrix (small angle approximation)
        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(rx), -torch.sin(rx)],
            [0, torch.sin(rx),  torch.cos(rx)]
        ], dtype=torch.float32)

        Ry = torch.tensor([
            [ torch.cos(ry), 0, torch.sin(ry)],
            [0, 1, 0],
            [-torch.sin(ry), 0, torch.cos(ry)]
        ], dtype=torch.float32)

        Rz = torch.tensor([
            [torch.cos(rz), -torch.sin(rz), 0],
            [torch.sin(rz),  torch.cos(rz), 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        R = Rz @ Ry @ Rx  # (3, 3)
        theta = torch.cat([R, torch.zeros(3, 1)], dim=1).unsqueeze(0)  # (1, 3, 4)

        grid = F.affine_grid(theta, x.unsqueeze(0).shape, align_corners=False)
        rotated = F.grid_sample(x.unsqueeze(0), grid, align_corners=False, mode="bilinear")
        return rotated.squeeze(0)


class GaussianNoise3D:
    """Add Gaussian noise with specified sigma."""

    def __init__(self, sigma=0.02):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.sigma
        return torch.clamp(x + noise, 0.0, 1.0)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def get_train_transform(model_name: str):
    """Return augmentation pipeline for training. None for Models A and B."""
    if model_name.upper() in ("A", "B"):
        return None
    return Compose([
        RandomFlip3D(axes=(1, 2, 3), p=0.5),
        RandomRotation3D(max_degrees=10.0),
        GaussianNoise3D(sigma=0.02),
    ])


def get_val_transform(model_name: str):
    """Validation always uses no augmentation."""
    return None
