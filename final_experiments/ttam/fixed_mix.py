"""Constant validation-fitted simplex mixture -- the "no AMG-TP" historical
control (plan sections 4-5).

For arms without AMG-TP a *single* convex weight vector ``w`` over the
distinct effective experts is fitted once per origin by minimising the
pooled inner-validation log loss

    w* = argmin_{w in simplex}  mean_i  logloss( sum_k w_k p_{i,k}, y_i )

and then **frozen for day d**. The identical ``w*`` is used by the
fixed-mixture ("without both modules"), AP-OPS-only, and OPS arms, so the
only thing that differs between those three arms and the AMG-TP arms is
the historical module (plan section 5's control).

The objective is convex in ``w``; it is solved by entropic mirror descent
(exponentiated gradient) with a fixed deterministic schedule -- no
external solver, so the fit is reproducible bit-for-bit given the same
inner-validation predictions.

The weight vector lives on the *distinct effective* experts of each day
(``bank.distinct_columns``); when it is applied on a day whose effective
set is smaller than the fitting day's, the entries for the missing
(coincident) experts are folded into the expert they coincide with, so
the mixture is well defined on every day of the origin's prefix.
"""
from __future__ import annotations

import numpy as np

from .bank import HORIZONS5, distinct_columns

EPS_LL = 1e-12
EPS_LOGIT = 1e-6


def _pooled_val_matrix(bank, inner_days):
    """(P, y) pooled over the inner-validation days, columns = the five
    nominal horizons (duplicates kept; the mixture is invariant to them)."""
    P, y = [], []
    for d in inner_days:
        if d not in bank:
            continue
        db = bank[d]
        P.append(np.stack([np.asarray(db.preds[h], float) for h in HORIZONS5], axis=1))
        y.append(np.asarray(db.y, float))
    if not P:
        return np.zeros((0, len(HORIZONS5))), np.zeros(0)
    return np.concatenate(P, axis=0), np.concatenate(y, axis=0)


def fit_simplex_weights(bank, inner_days, n_iter: int = 300, lr: float = 4.0,
                        tol: float = 1e-9) -> np.ndarray:
    """Fit ``w`` on the five nominal-horizon columns pooled over
    ``inner_days`` by exponentiated gradient. Returns a length-5 simplex
    vector (weight on each nominal horizon; coincident horizons simply
    share mass, which does not change any mixture prediction)."""
    P, y = _pooled_val_matrix(bank, inner_days)
    K = P.shape[1]
    if len(y) == 0:
        return np.full(K, 1.0 / K)
    w = np.full(K, 1.0 / K)
    prev = np.inf
    for t in range(1, n_iter + 1):
        p_mix = np.clip(P @ w, EPS_LL, 1 - EPS_LL)
        # gradient of mean logloss wrt w_k :  mean_i (p_mix - y) / (p_mix (1-p_mix)) * p_{i,k}
        # -> use the logistic-mixture surrogate gradient  mean_i (p_mix - y) * p_{i,k}
        # (same stationary point on the simplex; standard for convex mixture weights)
        resid = (p_mix - y)
        g = (P * resid[:, None]).mean(axis=0)
        step = lr / np.sqrt(t)
        w = w * np.exp(-step * (g - g.min()))
        s = w.sum()
        w = w / s if s > 0 else np.full(K, 1.0 / K)
        loss = float(-(y * np.log(p_mix) + (1 - y) * np.log(1 - p_mix)).mean())
        if abs(prev - loss) < tol:
            break
        prev = loss
    return w


def fixed_mixture_q(bank, days, w: np.ndarray) -> dict:
    """``q_{d,i} = sum_h w_h p^(h)_d(x_i)`` with the frozen ``w`` (length 5,
    one entry per nominal horizon), for every day in ``days``."""
    w = np.asarray(w, float)
    out = {}
    for d in sorted(int(x) for x in days if x in bank):
        db = bank[d]
        P = np.stack([np.asarray(db.preds[h], float) for h in HORIZONS5], axis=1)
        out[d] = P @ w
    return out


def effective_weight_report(bank, day: int, w: np.ndarray) -> dict:
    """Weight mass on each *distinct* effective expert of ``day`` (for the
    manifest / appendix)."""
    groups, _ = distinct_columns(bank[day])
    idx = {h: i for i, h in enumerate(HORIZONS5)}
    return {"+".join(g): float(sum(w[idx[h]] for h in g)) for g in groups}
