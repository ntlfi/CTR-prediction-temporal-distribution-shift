"""Metrics for the CTR baseline comparison."""
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss


def day_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Log loss, Brier score, and PR-AUC for one day's predictions."""
    if len(np.unique(y_true)) < 2:
        # PR-AUC (and, degenerately, log loss) are undefined with a single class.
        pr_auc = float("nan")
    else:
        pr_auc = average_precision_score(y_true, y_pred)
    return {
        "log_loss": log_loss(y_true, y_pred, labels=[0, 1]),
        "brier": brier_score_loss(y_true, y_pred),
        "pr_auc": pr_auc,
        "n": len(y_true),
    }
