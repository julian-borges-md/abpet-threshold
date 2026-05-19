"""
evaluate.py
RO-2026-002 — ABPET Threshold
Frontier Translational Research Lab (Independent)
PI: Julian Borges, MD

Full metric suite. Primary clinical metrics are TSA and zone-stratified MAE.
Global MAE and Pearson r are reported for comparison with prior work only.
"""

import numpy as np
from scipy import stats
from scipy.stats import pearsonr


# ── Primary clinical metrics (paper's core contribution) ─────────────────────

def threshold_specific_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                                 threshold: float) -> float:
    """
    Threshold-Specific Accuracy (TSA).

    Measures the proportion of patients correctly classified as above or below
    a clinical Centiloid threshold. Clinically: correct eligibility classification rate.

    For lecanemab eligibility: threshold = 24.0 CL (TSA_24)
    For donanemab cessation:   threshold = 10.0 CL (TSA_10)

    Args:
        y_true:    True Centiloid values
        y_pred:    Predicted Centiloid values
        threshold: Clinical decision threshold in CL

    Returns:
        float: Proportion correctly classified (0 to 1)
    """
    true_pos = y_true >= threshold
    pred_pos = y_pred >= threshold
    return float((true_pos == pred_pos).mean())


def zone_stratified_mae(y_true: np.ndarray, y_pred: np.ndarray,
                        low: float = 10.0, high: float = 30.0) -> dict:
    """
    Zone-stratified MAE.

    Computes MAE separately for:
        Zone A: true CL < low         (amyloid-negative)
        Zone B: low <= true CL <= high (treatment decision zone)
        Zone C: true CL > high        (amyloid-positive)

    Zone B MAE is the paper's primary finding. If Zone B MAE does not differ
    substantially from overall MAE, see escalation protocol in RO-2026-002.

    Returns dict with zone_a, zone_b, zone_c MAE and sample counts.
    """
    mask_a = y_true < low
    mask_b = (y_true >= low) & (y_true <= high)
    mask_c = y_true > high

    def safe_mae(yt, yp, mask):
        if mask.sum() == 0:
            return None
        return float(np.abs(yt[mask] - yp[mask]).mean())

    return {
        "zone_a_mae": safe_mae(y_true, y_pred, mask_a),
        "zone_b_mae": safe_mae(y_true, y_pred, mask_b),
        "zone_c_mae": safe_mae(y_true, y_pred, mask_c),
        "n_zone_a": int(mask_a.sum()),
        "n_zone_b": int(mask_b.sum()),
        "n_zone_c": int(mask_c.sum()),
    }


def misclassification_rate(y_true: np.ndarray, y_pred: np.ndarray,
                            threshold: float = 24.0,
                            zone_low: float = 10.0,
                            zone_high: float = 30.0) -> dict:
    """
    Treatment eligibility misclassification rate in Zone B.

    For patients with true Centiloid in Zone B (10-30 CL):
    what fraction receive an incorrect eligibility classification at the threshold?

    This is the paper's Contribution 2 — clinical translation metric.
    """
    mask_b = (y_true >= zone_low) & (y_true <= zone_high)
    if mask_b.sum() == 0:
        return {"n_zone_b": 0, "misclassification_rate": None}

    yt_b = y_true[mask_b]
    yp_b = y_pred[mask_b]

    true_pos = yt_b >= threshold
    pred_pos = yp_b >= threshold
    misclassified = (true_pos != pred_pos)

    return {
        "n_zone_b": int(mask_b.sum()),
        "n_misclassified": int(misclassified.sum()),
        "misclassification_rate": float(misclassified.mean()),
        "false_eligible": int(((~true_pos) & pred_pos).sum()),
        "false_ineligible": int((true_pos & (~pred_pos)).sum()),
    }

# ── Global metrics (for comparison with prior work) ───────────────────────────

def global_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())


def global_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r, _ = pearsonr(y_true, y_pred)
    return float(r)


def per_tracer_mae(y_true: np.ndarray, y_pred: np.ndarray,
                   tracers: np.ndarray) -> dict:
    result = {}
    for t in np.unique(tracers):
        mask = tracers == t
        result[str(t)] = global_mae(y_true[mask], y_pred[mask])
    return result


# ── Bootstrap confidence intervals ───────────────────────────────────────────

def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 metric_fn, n_bootstrap: int = 1000,
                 ci: float = 0.95, **kwargs) -> dict:
    """
    Bootstrap 95% CI for any scalar metric function.

    Args:
        metric_fn: callable(y_true, y_pred, **kwargs) -> float
        n_bootstrap: number of bootstrap samples
        ci: confidence interval width

    Returns:
        dict with mean, lower, upper
    """
    rng = np.random.default_rng(42)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_pred[idx], **kwargs))
    scores = np.array(scores)
    alpha = (1 - ci) / 2
    return {
        "mean": float(metric_fn(y_true, y_pred, **kwargs)),
        "lower": float(np.quantile(scores, alpha)),
        "upper": float(np.quantile(scores, 1 - alpha)),
    }


# ── Bland-Altman ──────────────────────────────────────────────────────────────

def bland_altman(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Bland-Altman analysis.
    Returns mean bias, SD of differences, and 95% limits of agreement.
    Reference line at +/-7.43 CL noise floor added in figures.py.
    """
    diff = y_pred - y_true
    mean_vals = (y_pred + y_true) / 2
    bias = float(diff.mean())
    sd = float(diff.std())
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd
    return {
        "bias": bias,
        "sd": sd,
        "loa_upper": loa_upper,
        "loa_lower": loa_lower,
        "mean_vals": mean_vals,
        "diff": diff,
    }


# ── Full evaluation report ────────────────────────────────────────────────────

def full_eval_report(y_true: np.ndarray, y_pred: np.ndarray,
                     tracers: np.ndarray, model_name: str) -> dict:
    """
    Compute and return the complete metric set for one model.
    Produces Table 2 (global) and Table 3 (clinical) data.
    """
    report = {
        "model": model_name,
        "n": len(y_true),
        # Global metrics
        "overall_mae": global_mae(y_true, y_pred),
        "overall_rmse": global_rmse(y_true, y_pred),
        "pearson_r": pearson_r(y_true, y_pred),
        "per_tracer_mae": per_tracer_mae(y_true, y_pred, tracers),
        # Clinical metrics (primary contribution)
        "tsa_24": threshold_specific_accuracy(y_true, y_pred, threshold=24.0),
        "tsa_10": threshold_specific_accuracy(y_true, y_pred, threshold=10.0),
        **zone_stratified_mae(y_true, y_pred),
        "misclassification": misclassification_rate(y_true, y_pred, threshold=24.0),
        # Bootstrap CIs for primary metrics
        "mae_ci": bootstrap_ci(y_true, y_pred, global_mae),
        "tsa24_ci": bootstrap_ci(y_true, y_pred, threshold_specific_accuracy, threshold=24.0),
        "tsa10_ci": bootstrap_ci(y_true, y_pred, threshold_specific_accuracy, threshold=10.0),
        # Bland-Altman
        "bland_altman": bland_altman(y_true, y_pred),
    }
    return report
