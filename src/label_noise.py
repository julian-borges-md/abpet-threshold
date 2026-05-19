"""
label_noise.py
RO-2026-002 — ABPET Threshold
Frontier Translational Research Lab (Independent)
PI: Julian Borges, MD

Centiloid label noise documentation and reference line values for figures.
Source: Shekari M et al. Alzheimers Dement. 2024;20:5102-5113.
"""

# ── Noise floor constants (Shekari 2024) ─────────────────────────────────────
NOISE_FLOOR_NEGATIVE = 2.70   # 95% CI half-width, amyloid-negative subjects
NOISE_FLOOR_POSITIVE = 7.43   # 95% CI half-width, amyloid-positive subjects
NOISE_CITATION = "Shekari M et al. Alzheimers Dement. 2024;20:5102-5113."

# ── Published benchmark for noise context ────────────────────────────────────
YAMAO_2024_MAE = 8.54         # Best published single-tracer model (PIB, Brain Sci 2024)
YAMAO_ABOVE_NOISE = round(YAMAO_2024_MAE - NOISE_FLOOR_POSITIVE, 2)  # 1.11 CL


def noise_summary() -> dict:
    return {
        "noise_floor_negative_cl": NOISE_FLOOR_NEGATIVE,
        "noise_floor_positive_cl": NOISE_FLOOR_POSITIVE,
        "best_published_mae": YAMAO_2024_MAE,
        "best_published_above_noise": YAMAO_ABOVE_NOISE,
        "citation": NOISE_CITATION,
    }


def add_noise_reference_band(ax, alpha=0.08, color="gray"):
    """
    Add +/-7.43 CL noise reference band to a Bland-Altman axis.
    Required on all Bland-Altman panels (Figure 2).
    """
    ax.axhspan(
        -NOISE_FLOOR_POSITIVE, NOISE_FLOOR_POSITIVE,
        alpha=alpha, color=color,
        label=f"Label noise reference (±{NOISE_FLOOR_POSITIVE} CL, Shekari 2024)"
    )
    return ax


def add_noise_reference_line(ax, color="gray", linestyle="--", alpha=0.5):
    """Add horizontal dashed line at 7.43 CL for MAE comparison figures."""
    ax.axhline(NOISE_FLOOR_POSITIVE, color=color, linestyle=linestyle,
               alpha=alpha, label=f"Label noise floor ({NOISE_FLOOR_POSITIVE} CL, Shekari 2024)")
    return ax
