"""P0 baselines: rules for which historical rows a fixed logistic-regression
model gets trained on, before predicting the next day's clicks.

Every rule has the same shape: given the array of row-days and the
prediction day `t`, return a boolean training mask and (optionally) a
per-row sample weight. The base model (SGDClassifier with log loss, i.e.
regularized logistic regression) is identical across all of them.
"""
import time

import numpy as np
from sklearn.linear_model import SGDClassifier

# Fixed windows shared by the rolling baselines and the validation-selection /
# hindsight-diagnostic candidate set (PDF section 3.2, 3.4, 5).
ROLLING_WINDOWS = [1, 3, 7, 14]
DECAY_HALF_LIVES = [1, 3, 7]


def expanding_rule(day: np.ndarray, t: int):
    return day < t, None


def rolling_rule(day: np.ndarray, t: int, h: int):
    return (day < t) & (day >= t - h), None


def decay_rule(day: np.ndarray, t: int, half_life: float):
    mask = day < t
    weights = np.zeros(len(day))
    weights[mask] = np.exp(-np.log(2) / half_life * (t - day[mask]))
    return mask, weights


def build_candidates() -> dict:
    """All P0 history rules, keyed by name."""
    candidates = {"expanding": expanding_rule}
    for h in ROLLING_WINDOWS:
        candidates[f"rolling_{h}"] = lambda day, t, h=h: rolling_rule(day, t, h)
    for hl in DECAY_HALF_LIVES:
        candidates[f"decay_hl{hl}"] = lambda day, t, hl=hl: decay_rule(day, t, hl)
    return candidates


# The window family used for "validation-selected window" (3.4) and the
# hindsight-best-window diagnostic (5) — fixed windows plus expanding only,
# not the decay baselines.
WINDOW_FAMILY = ["expanding"] + [f"rolling_{h}" for h in ROLLING_WINDOWS]


def fit_predict(X, y, day, t, rule, alpha=1e-4, seed=0):
    """Fit one regularized logistic regression per prediction day `t` and
    score it on that day. Returns None if there is no train or test data.
    """
    mask, weights = rule(day, t)
    test_mask = day == t
    if mask.sum() == 0 or test_mask.sum() == 0:
        return None
    if len(np.unique(y[mask])) < 2:
        return None  # can't fit logistic regression on a single class

    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=alpha, random_state=seed)
    start = time.time()
    clf.fit(X[mask], y[mask], sample_weight=weights[mask] if weights is not None else None)
    fit_time = time.time() - start

    y_true = y[test_mask]
    y_pred = clf.predict_proba(X[test_mask])[:, 1]
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "n_train": int(mask.sum()),
        "fit_time": fit_time,
    }
