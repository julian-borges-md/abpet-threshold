"""
ablation.py
RO-2026-002 — ABPET Threshold
Frontier Translational Research Lab (Independent)
PI: Julian Borges, MD

Sequential 6-model ablation runner. Trains A through F in order,
evaluates each, and produces ablation_table.csv and clinical_metrics_table.csv.

Usage:
    python src/ablation.py \
        --train_csv data/processed/adni_train.csv \
        --val_csv data/processed/adni_val.csv \
        --epochs 100 \
        --output_dir results/ablation/
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.train import train_model
from src.model import get_model
from src.dataset import PETDataset
from src.augmentation import get_val_transform
from src.evaluate import full_eval_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_ORDER = ["A", "B", "C", "D", "E", "F"]

# Baseline anchor check (DEC-004)
BASELINE_MAE_EXPECTED = 19.77
BASELINE_MAE_TOLERANCE = 3.0   # If Model A deviates >3 CL, halt and investigate


def run_ablation(train_csv: str, val_csv: str, epochs: int,
                 output_dir: str, medicalnet_weights: str = None,
                 batch_size: int = 4):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    ablation_rows = []
    clinical_rows = []

    for model_name in MODEL_ORDER:
        log.info(f"\n{'='*60}\nTraining Model {model_name}\n{'='*60}")

        model_dir = output_dir / f"model_{model_name}"
        mw = medicalnet_weights if model_name == "F" else None

        result = train_model(
            model_name=model_name,
            train_csv=train_csv,
            val_csv=val_csv,
            epochs=epochs,
            output_dir=str(model_dir),
            batch_size=batch_size,
            medicalnet_weights=mw
        )

        # Load best checkpoint for evaluation
        model = get_model(model_name)
        ckpt = model_dir / f"model_{model_name}_best.pt"
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        model.eval()

        val_ds = PETDataset(val_csv, transform=get_val_transform(model_name))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        all_preds, all_true, all_tracers = [], [], []
        with torch.no_grad():
            for imgs, targets, tracer_idx, _ in val_loader:
                preds = model(imgs, tracer_idx)
                all_preds.extend(preds.numpy())
                all_true.extend(targets.numpy())
                all_tracers.extend(tracer_idx.numpy())

        y_pred = np.array(all_preds)
        y_true = np.array(all_true)
        tracers = np.array(all_tracers)

        report = full_eval_report(y_true, y_pred, tracers, model_name)

        # Model A anchor check
        if model_name == "A":
            deviation = abs(report["overall_mae"] - BASELINE_MAE_EXPECTED)
            if deviation > BASELINE_MAE_TOLERANCE:
                log.error(
                    f"Model A MAE={report['overall_mae']:.2f} deviates {deviation:.2f} CL "
                    f"from expected {BASELINE_MAE_EXPECTED}. "
                    f"HALT: investigate preprocessing before continuing ablation. (DEC-007)"
                )
                raise RuntimeError("Model A anchor check failed. See DEC-007.")
            log.info(f"Model A anchor check PASSED: MAE={report['overall_mae']:.4f}")

        # Zone B minimum check
        if report["n_zone_b"] < 30:
            log.warning(
                f"Model {model_name}: Zone B n={report['n_zone_b']} < 30. "
                f"TSA_24 may not be statistically valid. See escalation protocol."
            )

        # Build table rows
        pt_mae = report["per_tracer_mae"]
        ablation_rows.append({
            "Model": model_name,
            "Overall_MAE": round(report["overall_mae"], 3),
            "MAE_CI_lower": round(report["mae_ci"]["lower"], 3),
            "MAE_CI_upper": round(report["mae_ci"]["upper"], 3),
            "RMSE": round(report["overall_rmse"], 3),
            "Pearson_r": round(report["pearson_r"], 3),
            "MAE_FBP": round(pt_mae.get("FBP", float("nan")), 3),
            "MAE_FBB": round(pt_mae.get("FBB", float("nan")), 3),
            "MAE_PIB": round(pt_mae.get("PIB", float("nan")), 3),
            "MAE_NAV": round(pt_mae.get("NAV", float("nan")), 3),
        })

        mc = report["misclassification"]
        clinical_rows.append({
            "Model": model_name,
            "Zone_A_MAE": round(report["zone_a_mae"] or float("nan"), 3),
            "Zone_B_MAE": round(report["zone_b_mae"] or float("nan"), 3),
            "Zone_C_MAE": round(report["zone_c_mae"] or float("nan"), 3),
            "n_Zone_B": report["n_zone_b"],
            "TSA_24_pct": round(report["tsa_24"] * 100, 1),
            "TSA_24_CI": f"[{report['tsa24_ci']['lower']*100:.1f}, {report['tsa24_ci']['upper']*100:.1f}]",
            "TSA_10_pct": round(report["tsa_10"] * 100, 1),
            "TSA_10_CI": f"[{report['tsa10_ci']['lower']*100:.1f}, {report['tsa10_ci']['upper']*100:.1f}]",
            "Misclassification_rate_pct": round((mc["misclassification_rate"] or 0) * 100, 1),
            "False_eligible": mc["false_eligible"],
            "False_ineligible": mc["false_ineligible"],
        })

        with open(output_dir / f"report_{model_name}.json", "w") as f:
            # Exclude non-serializable arrays before saving
            safe = {k: v for k, v in report.items()
                    if k not in ("bland_altman",)}
            json.dump(safe, f, indent=2, default=str)

    pd.DataFrame(ablation_rows).to_csv(output_dir / "ablation_table.csv", index=False)
    pd.DataFrame(clinical_rows).to_csv(output_dir / "clinical_metrics_table.csv", index=False)

    log.info(f"\nAblation complete. Results in {output_dir}")
    log.info("ablation_table.csv → Table 2 (manuscript)")
    log.info("clinical_metrics_table.csv → Table 3 (manuscript, primary contribution)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--medicalnet_weights", default=None)
    args = parser.parse_args()
    run_ablation(**vars(args))
