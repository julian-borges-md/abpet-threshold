"""
dataset.py
RO-2026-002 — ABPET Threshold
Frontier Translational Research Lab (Independent)
PI: Julian Borges, MD

PETDataset class. Returns (image, centiloid, tracer_idx, zone) per sample.
Compatible with hackathon DataLoader API.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TRACER_MAP = {"FBP": 0, "FBB": 1, "PIB": 2, "NAV": 3}
ZONE_MAP = {"A": 0, "B": 1, "C": 2}


class PETDataset(Dataset):
    """
    Amyloid PET dataset for Centiloid regression.

    Args:
        csv_path: Path to adni_train.csv or adni_val.csv
        transform: Optional callable applied to image tensor
        return_zone: If True, also return CLINICAL_ZONE index

    Returns per item:
        image:      torch.FloatTensor (1, 128, 128, 128)
        centiloid:  torch.FloatTensor scalar
        tracer_idx: torch.LongTensor scalar
        zone_idx:   torch.LongTensor scalar (0=A, 1=B, 2=C)
    """

    def __init__(self, csv_path: str, transform=None, return_zone: bool = True):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.return_zone = return_zone

        missing = {"npy_path", "CENTILOIDS", "TRACER", "CLINICAL_ZONE"} - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image = np.load(row["npy_path"]).astype(np.float32)
        image = torch.from_numpy(image)  # (1, 128, 128, 128)

        if self.transform:
            image = self.transform(image)

        centiloid = torch.tensor(float(row["CENTILOIDS"]), dtype=torch.float32)
        tracer_idx = torch.tensor(TRACER_MAP.get(str(row["TRACER"]).upper(), 0), dtype=torch.long)
        zone_idx = torch.tensor(ZONE_MAP.get(str(row["CLINICAL_ZONE"]).upper(), 0), dtype=torch.long)

        return image, centiloid, tracer_idx, zone_idx

    def get_zone_indices(self, zone: str) -> list:
        """Return dataset indices for a specific clinical zone (A, B, or C)."""
        return self.df[self.df["CLINICAL_ZONE"] == zone.upper()].index.tolist()

    def zone_summary(self) -> dict:
        """Return count and Centiloid range per zone."""
        summary = {}
        for zone in ["A", "B", "C"]:
            subset = self.df[self.df["CLINICAL_ZONE"] == zone]
            summary[f"Zone_{zone}"] = {
                "n": len(subset),
                "cl_min": subset["CENTILOIDS"].min() if len(subset) else None,
                "cl_max": subset["CENTILOIDS"].max() if len(subset) else None,
                "cl_mean": subset["CENTILOIDS"].mean() if len(subset) else None,
            }
        return summary
