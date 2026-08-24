"""Bennett & Clarkson, "Differentiable Forgetting" -- learned temporal
weighting baseline (PDF 3.6).

Reproduces their bilevel structure: an inner model fit on age-decayed
weights alpha(tau; eta) = exp(-eta * tau) (their "GradExp" variant), an
outer objective evaluated on later chronological data (never a random
split) that picks eta, then a final refit on all permissible history using
the learned eta.

The paper optimizes eta by differentiating through the inner solve via the
implicit function theorem (Gould et al. 2016), which needs an inner solver
exposing exact gradients/Hessians -- not available through
sklearn's SGDClassifier. Since eta is a single scalar here (GradExp), we
instead solve the same bilevel objective with bounded derivative-free
search (scipy Brent). The paper's own "GridSearchExp" ablation is exactly
this substitute for the one-parameter case, so this preserves the method's
statistical content without requiring custom autodiff.

One simplification from the published two-stage boundary scheme: rather
than weighting relative to a fixed inner/outer split point and then
re-weighting validation rows uniformly for the final refit (a scheme built
for their single-forecast-step regression setting), we use the actual
current prediction day t as the single age reference throughout -- both
for eta search and the final refit. This keeps Differentiable Forgetting
directly comparable to the P0 decay_hl baselines (baselines.decay_rule),
which use the same age(t) = t - day convention with a fixed half-life
instead of a learned one.

Reference implementation: https://github.com/jase-clarkson/pods_2022_icml_ts
"""
import math
import time

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

ETA_BOUNDS = (0.0, 3.0)  # half-life range ~ [0.23, inf) days


def _weights(day: np.ndarray, t_ref: int, eta: float) -> np.ndarray:
    return np.exp(-eta * (t_ref - day))


def _fit_weighted(X, y, day, mask, t_ref, eta, alpha, seed):
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=alpha, random_state=seed)
    clf.fit(X[mask], y[mask], sample_weight=_weights(day[mask], t_ref, eta))
    return clf


def _outer_loss(eta, X, y, day, inner_mask, outer_mask, t_ref, alpha, seed):
    clf = _fit_weighted(X, y, day, inner_mask, t_ref, eta, alpha, seed)
    p = clf.predict_proba(X[outer_mask])[:, 1]
    return log_loss(y[outer_mask], p, labels=[0, 1])


def fit_predict(X, y, day, t, val_window=3, min_inner_days=3, alpha=1e-4, seed=0, maxiter=12):
    """Bilevel-fit eta on chronologically-later held-out days, then refit on
    all permissible history and score day t. Returns None if there isn't
    enough history yet (mirrors baselines.fit_predict's contract)."""
    test_mask = day == t
    if test_mask.sum() == 0:
        return None

    hist_days = sorted(np.unique(day[day < t]))
    if len(hist_days) < min_inner_days + val_window:
        return None
    outer_val_days = set(hist_days[-val_window:])
    inner_train_days = set(hist_days[:-val_window])

    inner_mask = np.isin(day, list(inner_train_days))
    outer_mask = np.isin(day, list(outer_val_days))
    if len(np.unique(y[inner_mask])) < 2 or len(np.unique(y[outer_mask])) < 2:
        return None

    start = time.time()
    res = minimize_scalar(
        lambda eta: _outer_loss(eta, X, y, day, inner_mask, outer_mask, t, alpha, seed),
        bounds=ETA_BOUNDS, method="bounded", options={"maxiter": maxiter, "xatol": 1e-2},
    )
    eta_hat = float(res.x)

    all_mask = day < t
    if len(np.unique(y[all_mask])) < 2:
        return None
    clf = _fit_weighted(X, y, day, all_mask, t, eta_hat, alpha, seed)
    fit_time = time.time() - start

    y_pred = clf.predict_proba(X[test_mask])[:, 1]
    half_life = math.log(2) / eta_hat if eta_hat > 1e-9 else float("inf")
    return {
        "y_true": y[test_mask],
        "y_pred": y_pred,
        "n_train": int(all_mask.sum()),
        "fit_time": fit_time,
        "eta": eta_hat,
        "half_life": half_life,
    }
