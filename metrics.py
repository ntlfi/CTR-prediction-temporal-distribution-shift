"""Metrics for the CTR baseline comparison."""
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score)


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 15) -> float:
    """Binned |confidence - accuracy| calibration error (AMG-TP plan 5.4's
    "calibration or reliability error"). Equal-width probability bins,
    weighted by bin population."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_pred, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(y_pred[m].mean() - y_true[m].mean())
    return float(ece)


def day_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Log loss, Brier score, PR-AUC, ROC-AUC and calibration error for one
    day's predictions (AMG-TP plan 5.4). ROC-AUC / PR-AUC are undefined with a
    single class present and come back as NaN."""
    if len(np.unique(y_true)) < 2:
        pr_auc = float("nan")
        roc_auc = float("nan")
    else:
        pr_auc = average_precision_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
    return {
        "log_loss": log_loss(y_true, y_pred, labels=[0, 1]),
        "brier": brier_score_loss(y_true, y_pred),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "ece": expected_calibration_error(y_true, y_pred),
        "n": len(y_true),
    }
