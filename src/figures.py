"""
figures.py
RO-2026-002 — ABPET Threshold
Frontier Translational Research Lab (Independent)
PI: Julian Borges, MD

All four manuscript figures. 300 DPI, publication-ready, restrained palette.

Figure 1: Treatment eligibility misclassification (scatter + bar)
Figure 2: Bland-Altman per tracer for best model (4 panels)
Figure 3: Training dynamics ablation (loss, val MAE, val r)
Figure 4: FiLM vs concat tracer feature space (t-SNE, 2 panels)
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

from src.label_noise import add_noise_reference_band, NOISE_FLOOR_POSITIVE
from src.evaluate import bland_altman

PALETTE = {
    "A": "#4878CF", "B": "#6ACC65", "C": "#D65F5F",
    "D": "#B47CC7", "E": "#C4AD66", "F": "#77BEDB",
    "zone_b": "#F5A623", "threshold": "#D0021B",
    "identity": "#AAAAAA", "noise": "#BBBBBB"
}
TRACER_COLORS = {"FBP": "#4878CF", "FBB": "#6ACC65", "PIB": "#D65F5F", "NAV": "#B47CC7"}
DPI = 300
FONT = {"family": "Arial", "size": 8}
plt.rc("font", **FONT)


def fig1_misclassification(results_dir: Path, output_dir: Path, best_model: str = "E"):
    """
    Figure 1: Treatment eligibility misclassification scatter + bar chart.
    Left: scatter true vs predicted, Zone B highlighted, 24 CL threshold line.
    Right: misclassification rate per model at 24 CL.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))

    # Left panel: scatter for best model
    report = json.load(open(results_dir / f"report_{best_model}.json"))
    # Load raw predictions from saved CSV
    pred_csv = results_dir / f"model_{best_model}" / "val_predictions.csv"
    if pred_csv.exists():
        df = pd.read_csv(pred_csv)
        y_true = df["y_true"].values
        y_pred = df["y_pred"].values
        zone = df["zone"].values

        mask_b = zone == "B"
        mask_other = ~mask_b

        ax = axes[0]
        ax.scatter(y_true[mask_other], y_pred[mask_other],
                   s=8, alpha=0.4, color="#AAAAAA", label="Zone A/C")
        ax.scatter(y_true[mask_b], y_pred[mask_b],
                   s=10, alpha=0.7, color=PALETTE["zone_b"], label="Zone B (10-30 CL)")
        lims = [min(y_true.min(), y_pred.min()) - 5, max(y_true.max(), y_pred.max()) + 5]
        ax.plot(lims, lims, color=PALETTE["identity"], lw=0.8, linestyle="--")
        ax.axvline(24, color=PALETTE["threshold"], lw=1.0, linestyle="--", label="24 CL threshold")
        ax.axhline(24, color=PALETTE["threshold"], lw=1.0, linestyle="--")
        ax.set_xlabel("True Centiloid (CL)")
        ax.set_ylabel("Predicted Centiloid (CL)")
        ax.set_title(f"Model {best_model}: True vs Predicted")
        ax.legend(fontsize=7, frameon=False)

    # Right panel: misclassification rate bar chart
    clinical = pd.read_csv(results_dir / "clinical_metrics_table.csv")
    ax = axes[1]
    colors = [PALETTE.get(m, "#888888") for m in clinical["Model"]]
    bars = ax.bar(clinical["Model"], clinical["Misclassification_rate_pct"],
                  color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Model")
    ax.set_ylabel("Misclassification Rate (%)")
    ax.set_title("Zone B Misclassification at 24 CL Threshold")
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, clinical["Misclassification_rate_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out = output_dir / "fig1_misclassification.png"
    plt.savefig(str(out), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def fig2_bland_altman(results_dir: Path, output_dir: Path, best_model: str = "E"):
    """
    Figure 2: 4-panel Bland-Altman per tracer for best model.
    +/-7.43 CL noise reference band on each panel.
    """
    pred_csv = results_dir / f"model_{best_model}" / "val_predictions.csv"
    if not pred_csv.exists():
        print(f"Skipping Figure 2: {pred_csv} not found")
        return

    df = pd.read_csv(pred_csv)
    tracers = ["FBP", "FBB", "PIB", "NAV"]
    fig, axes = plt.subplots(1, 4, figsize=(10.0, 3.0))

    for ax, tracer in zip(axes, tracers):
        sub = df[df["tracer"] == tracer]
        if len(sub) == 0:
            ax.set_title(f"{tracer} (n=0)")
            continue

        ba = bland_altman(sub["y_true"].values, sub["y_pred"].values)
        add_noise_reference_band(ax)
        ax.scatter(ba["mean_vals"], ba["diff"], s=6, alpha=0.5,
                   color=TRACER_COLORS.get(tracer, "#888888"))
        ax.axhline(ba["bias"], color="black", lw=1.0, label=f"Bias {ba['bias']:.1f}")
        ax.axhline(ba["loa_upper"], color="#D65F5F", lw=0.8, linestyle="--",
                   label=f"LoA {ba['loa_upper']:.1f}")
        ax.axhline(ba["loa_lower"], color="#D65F5F", lw=0.8, linestyle="--",
                   label=f"LoA {ba['loa_lower']:.1f}")
        ax.axhline(0, color="#AAAAAA", lw=0.5)
        ax.set_title(f"{tracer} (n={len(sub)})")
        ax.set_xlabel("Mean CL")
        if tracer == "FBP":
            ax.set_ylabel("Predicted − True (CL)")
        ax.legend(fontsize=6, frameon=False)

    plt.suptitle(f"Bland-Altman per Tracer — Model {best_model}", fontsize=9)
    plt.tight_layout()
    out = output_dir / "fig2_bland_altman.png"
    plt.savefig(str(out), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def fig3_training_dynamics(results_dir: Path, output_dir: Path):
    """Figure 3: Training dynamics — 3 panels for all 6 models."""
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.5))
    titles = ["Training Loss", "Validation MAE (CL)", "Validation Pearson r"]
    keys = ["train_loss", "val_mae", "val_r"]

    for model_name in ["A", "B", "C", "D", "E", "F"]:
        hist_path = results_dir / f"model_{model_name}" / f"history_{model_name}.json"
        if not hist_path.exists():
            continue
        history = json.load(open(hist_path))
        epochs = [h["epoch"] for h in history]
        for ax, key in zip(axes, keys):
            vals = [h[key] for h in history]
            ax.plot(epochs, vals, color=PALETTE[model_name],
                    label=f"Model {model_name}", lw=1.0)

    # Baseline reference line on MAE panel
    axes[1].axhline(19.77, color="#AAAAAA", linestyle="--", lw=0.8,
                    label="Baseline 19.77")

    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=6, frameon=False)

    plt.tight_layout()
    out = output_dir / "fig3_training_dynamics.png"
    plt.savefig(str(out), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def fig4_film_features(results_dir: Path, output_dir: Path):
    """
    Figure 4: t-SNE of penultimate features — Model D (concat) vs Model E (FiLM).
    Requires saved feature tensors from evaluate step.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    for ax, model_name, title in zip(axes, ["D", "E"],
                                     ["Model D: Concat Fusion", "Model E: FiLM Fusion"]):
        feat_path = results_dir / f"model_{model_name}" / "features.npy"
        label_path = results_dir / f"model_{model_name}" / "tracer_labels.npy"
        if not feat_path.exists():
            ax.set_title(f"{title}\n(features not saved)")
            continue
        feats = np.load(str(feat_path))
        labels = np.load(str(label_path))
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        emb = tsne.fit_transform(feats)
        tracer_names = ["FBP", "FBB", "PIB", "NAV"]
        for i, tname in enumerate(tracer_names):
            mask = labels == i
            ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.6,
                       color=list(TRACER_COLORS.values())[i], label=tname)
        ax.set_title(title)
        ax.legend(fontsize=7, frameon=False)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out = output_dir / "fig4_film_features.png"
    plt.savefig(str(out), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--best_model", default="E",
                        help="Model name for Figures 1 and 2")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)

    fig1_misclassification(results_dir, output_dir, args.best_model)
    fig2_bland_altman(results_dir, output_dir, args.best_model)
    fig3_training_dynamics(results_dir, output_dir)
    fig4_film_features(results_dir, output_dir)
    print("All figures complete.")
