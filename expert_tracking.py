"""Classical prediction-with-expert-advice baselines over the five
window-family experts (expanding + rolling 1/3/7/14): Fixed Share and
Learn-alpha. These are the closest classical competitors to AMG-TP's idea
of maintaining expert mass under a changing environment and adapting the
switching rate online -- purely loss-driven, no context, no state features.

Causal contract (same as adamoe.py / han_arw.py): day t's prediction uses
the weight vector as of day t-1; weights update only after day t's labels
mature.

References:
  Herbster & Warmuth, "Tracking the Best Expert", Machine Learning 1998
    (Fixed-Share, the "share to uniform past" update).
  Monteleoni & Jaakkola, "Online Learning of Non-Stationary Sequences",
    NIPS 2003 (Learn-alpha: a hierarchy of Fixed-Share sub-algorithms over
    a grid of switching rates, with a top-level exponential-weights update
    that concentrates on whichever rate is tracking best).
"""
import numpy as np

from baselines import WINDOW_FAMILY
from han_arw import per_sample_log_loss

DEFAULT_ETA = 2.0
DEFAULT_ALPHA = 0.08
ALPHA_GRID = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.4]


def _day_expert_losses(preds: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean per-sample log loss of each expert on this day."""
    return np.array([per_sample_log_loss(y, preds[:, i]).mean() for i in range(preds.shape[1])])


def _fixed_share_update(w: np.ndarray, loss: np.ndarray, eta: float, alpha: float) -> np.ndarray:
    """One Fixed-Share step: exponential-weights loss update, then share
    `alpha` of the mass back to uniform (Herbster-Warmuth)."""
    w = w * np.exp(-eta * (loss - loss.min()))
    w = w / w.sum()
    N = len(w)
    return (1.0 - alpha) * w + alpha / N


def _rows_scaffold(bank, names, t, y, p):
    return {"day": t, "y_true": y, "y_pred": p,
            "n_train": int(np.mean([bank[n][t]["n_train"] for n in names])),
            "fit_time": float(sum(bank[n][t]["fit_time"] for n in names))}


def run_fixed_share(bank: dict, eligible_days, eta: float = DEFAULT_ETA, alpha: float = DEFAULT_ALPHA):
    names = list(WINDOW_FAMILY)
    N = len(names)
    w = np.full(N, 1.0 / N)
    rows = []
    for t in sorted(eligible_days):
        if not all(t in bank[n] for n in names):
            continue
        preds = np.stack([bank[n][t]["y_pred"] for n in names], axis=1)
        y = bank[names[0]][t]["y_true"]
        p = preds @ w
        rec = _rows_scaffold(bank, names, t, y, p)
        rec["weights"] = dict(zip(names, w.tolist()))
        rows.append(rec)
        w = _fixed_share_update(w, _day_expert_losses(preds, y), eta, alpha)
    return rows


def run_learn_alpha(bank: dict, eligible_days, eta: float = DEFAULT_ETA, alpha_grid=None):
    """Learn-alpha: one Fixed-Share sub-algorithm per switching rate in
    `alpha_grid`, with a top-level exponential-weights update over the
    sub-algorithms driven by each one's own realised prediction loss."""
    names = list(WINDOW_FAMILY)
    N = len(names)
    grid = list(ALPHA_GRID if alpha_grid is None else alpha_grid)
    M = len(grid)
    W = np.full((M, N), 1.0 / N)          # per-sub-algorithm expert weights
    v = np.full(M, 1.0 / M)               # top-level weights over switching rates
    rows = []
    for t in sorted(eligible_days):
        if not all(t in bank[n] for n in names):
            continue
        preds = np.stack([bank[n][t]["y_pred"] for n in names], axis=1)   # (n, N)
        y = bank[names[0]][t]["y_true"]
        sub_pred = preds @ W.T                                            # (n, M)
        p = sub_pred @ v
        rec = _rows_scaffold(bank, names, t, y, p)
        rec["weights"] = dict(zip(names, (v @ W).tolist()))
        rec["mean_alpha"] = float(v @ np.array(grid))
        rows.append(rec)
        # top-level: each sub-algorithm's realised loss this day
        sub_loss = np.array([per_sample_log_loss(y, sub_pred[:, j]).mean() for j in range(M)])
        v = v * np.exp(-eta * (sub_loss - sub_loss.min()))
        v = v / v.sum()
        # each sub-algorithm advances by its own Fixed-Share rate
        eloss = _day_expert_losses(preds, y)
        for j, a in enumerate(grid):
            W[j] = _fixed_share_update(W[j], eloss, eta, a)
    return rows
