"""
preprocess.py
RO-2026-002 — ABPET Threshold
Frontier Translational Research Lab (Independent)
PI: Julian Borges, MD

Convert ADNI amyloid PET NIfTI files to preprocessed npy arrays.
Applies the exact 9-step hackathon pipeline for baseline comparability.
Adds CLINICAL_ZONE column to output CSV.

Usage:
    python src/preprocess.py --adni_dir /path/to/adni_nifti --output_dir data/processed/
    python src/preprocess.py --adni_dir /path/to/adni --output_dir data/processed/ --split_seed 42
"""

import argparse
import os
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from monai.transforms import (
    LoadImage, EnsureChannelFirst, Orientation, Spacing,
    CropForeground, Resize, SpatialPad, ScaleIntensityRange
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Clinical zone boundaries (RO-2026-002 Section 3.4) ──────────────────────
ZONE_A_MAX = 10.0    # < 10 CL: amyloid-negative with high certainty
ZONE_B_MIN = 10.0    # 10-30 CL: intermediate, treatment decision zone
ZONE_B_MAX = 30.0
ZONE_C_MIN = 30.0    # > 30 CL: amyloid-positive with high certainty

# ── Preprocessing constants (hackathon pipeline — DO NOT MODIFY) ─────────────
TARGET_SPACING = (2.0, 2.0, 2.0)   # Step 3: isotropic 2mm
FOREGROUND_MARGIN = 10              # Step 4: CropForeground margin
TARGET_SIZE = (128, 128, 128)       # Step 5: resize target
PAD_SIZE = (128, 128, 128)          # Step 6: spatial pad to exactly 128^3
NORM_MIN = 0.0                      # Step 9: min-max normalization
NORM_MAX = 1.0

def assign_clinical_zone(centiloid: float) -> str:
    """Assign clinical zone per RO-2026-002 Section 3.4."""
    if centiloid < ZONE_A_MAX:
        return "A"
    elif centiloid <= ZONE_B_MAX:
        return "B"
    else:
        return "C"


def preprocess_volume(nifti_path: str) -> np.ndarray:
    """
    Apply the 9-step hackathon preprocessing pipeline.
    Any deviation from this pipeline breaks baseline comparability.
    Document deviations in governance/decision_log.md before implementing.

    Steps:
        1. Load NIfTI and ensure channel first (C, H, W, D)
        2. Reorient to RAS
        3. Isotropic resampling to 2mm isotropic
        4. Foreground crop with 10-voxel margin
        5. Resize to 128x128x128 (trilinear)
        6. Spatial padding to exactly 128^3
        7. Dynamic frame averaging (4D PET only)
        8. Shape enforcement to (1, 128, 128, 128)
        9. Min-max normalization to [0, 1] per volume

    Returns:
        np.ndarray: shape (1, 128, 128, 128), dtype float32
    """
    loader = LoadImage(image_only=True)
    img = loader(nifti_path)

    # Step 1: channel first
    ensure_channel = EnsureChannelFirst()
    img = ensure_channel(img)

    # Step 7: dynamic frame averaging for 4D volumes
    if img.ndim == 5:
        img = img.mean(axis=-1, keepdims=False)

    # Step 2: reorient to RAS
    reorient = Orientation(axcodes="RAS")
    img = reorient(img)

    # Step 3: isotropic resampling to 2mm
    resample = Spacing(pixdim=TARGET_SPACING, mode="bilinear")
    img = resample(img)

    # Step 4: foreground crop with 10-voxel margin
    crop = CropForeground(margin=FOREGROUND_MARGIN)
    img = crop(img)

    # Step 5: resize to 128^3
    resize = Resize(spatial_size=TARGET_SIZE, mode="trilinear")
    img = resize(img)

    # Step 6: pad to exactly 128^3
    pad = SpatialPad(spatial_size=PAD_SIZE)
    img = pad(img)

    # Step 8: enforce shape
    arr = np.array(img)
    if arr.shape != (1, 128, 128, 128):
        raise ValueError(f"Shape enforcement failed: got {arr.shape}, expected (1,128,128,128)")

    # Step 9: min-max normalization per volume
    vmin, vmax = arr.min(), arr.max()
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr)

    return arr.astype(np.float32)

def process_adni_dataset(
    adni_dir: str,
    metadata_csv: str,
    output_dir: str,
    split_seed: int = 42,
    train_frac: float = 0.8
) -> None:
    """
    Process all ADNI amyloid PET scans and produce train/val CSVs.

    Expects metadata_csv with columns:
        SUBJECT_ID, NIFTI_PATH, CENTILOIDS, TRACER (FBP/FBB/PIB/NAV)

    Produces:
        output_dir/adni_train.csv
        output_dir/adni_val.csv
        Each with columns: npy_path, CENTILOIDS, TRACER, ID, SPLIT, CLINICAL_ZONE
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_dir = output_dir / "npy"
    npy_dir.mkdir(exist_ok=True)

    meta = pd.read_csv(metadata_csv)
    required = {"SUBJECT_ID", "NIFTI_PATH", "CENTILOIDS", "TRACER"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Metadata CSV missing columns: {missing}")

    records = []
    errors = []

    for _, row in meta.iterrows():
        nifti_path = row["NIFTI_PATH"]
        subject_id = row["SUBJECT_ID"]
        centiloid = float(row["CENTILOIDS"])
        tracer = str(row["TRACER"]).upper()

        npy_path = npy_dir / f"{subject_id}.npy"

        try:
            if not npy_path.exists():
                arr = preprocess_volume(nifti_path)
                np.save(str(npy_path), arr)
                log.info(f"Processed {subject_id} — CL={centiloid:.1f} tracer={tracer}")
            else:
                log.info(f"Skipping {subject_id} — npy exists")

            records.append({
                "npy_path": str(npy_path),
                "CENTILOIDS": centiloid,
                "TRACER": tracer,
                "ID": subject_id,
                "CLINICAL_ZONE": assign_clinical_zone(centiloid)
            })
        except Exception as e:
            log.error(f"Failed {subject_id}: {e}")
            errors.append({"ID": subject_id, "error": str(e)})

    df = pd.DataFrame(records)

    # Train/val split stratified by tracer
    rng = np.random.default_rng(split_seed)
    train_idx = []
    val_idx = []
    for tracer in df["TRACER"].unique():
        idx = df[df["TRACER"] == tracer].index.tolist()
        rng.shuffle(idx)
        n_train = int(len(idx) * train_frac)
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:])

    df.loc[train_idx, "SPLIT"] = "train"
    df.loc[val_idx, "SPLIT"] = "val"

    df[df["SPLIT"] == "train"].to_csv(output_dir / "adni_train.csv", index=False)
    df[df["SPLIT"] == "val"].to_csv(output_dir / "adni_val.csv", index=False)

    # Zone distribution report
    zone_counts = df["CLINICAL_ZONE"].value_counts()
    log.info(f"Zone distribution: {zone_counts.to_dict()}")
    zone_b_n = int(zone_counts.get("B", 0))
    if zone_b_n < 30:
        log.warning(f"Zone B n={zone_b_n} — BELOW MINIMUM of 30. See escalation protocol.")

    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "preprocessing_errors.csv", index=False)
        log.warning(f"{len(errors)} files failed preprocessing — see preprocessing_errors.csv")

    log.info(f"Done. Train={len(train_idx)}, Val={len(val_idx)}, Errors={len(errors)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADNI PET preprocessing — RO-2026-002")
    parser.add_argument("--adni_dir", required=True, help="Root directory of ADNI NIfTI files")
    parser.add_argument("--metadata_csv", required=True, help="CSV with SUBJECT_ID, NIFTI_PATH, CENTILOIDS, TRACER")
    parser.add_argument("--output_dir", required=True, help="Output directory for npy files and CSVs")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.8)
    args = parser.parse_args()

    process_adni_dataset(
        adni_dir=args.adni_dir,
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        split_seed=args.split_seed,
        train_frac=args.train_frac
    )
