"""Long-term cross-day CTR prediction (plan section 2.1).

At the start of prediction day ``d`` we fit a small family of base predictors
on **days < d only**:

    f^(3)   rolling 3-day history
    f^(7)   rolling 7-day history
    f^(exp) expanding (all previous days)

Each is a fresh L2-regularised logistic regression (``SGDClassifier``, log
loss) on the shared hashed feature matrix -- identical model, differing only
in which rows it trains on, so any difference is the temporal mechanism.

The **adaptive multi-timescale** predictor combines them with exponential
weights (Hedge) driven by each candidate's own recent held-out day loss,
using only matured information:

    w_{d,h} proportional to exp(-eta * Lbar_{d,h}),
    Lbar_{d,h} = discounted mean per-day log loss of f^(h) over days e < d
                 (each f^(h) scored on day e was itself fitted on days < e).

    q_{d,i} = sum_h w_{d,h} f^(h)_d(x_i).

All weights are frozen for the whole of day ``d`` (clean decomposition,
plan section 2.1). This module has no dependency on the repo's other
adaptive-training code.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import SGDClassifier

from .data import Dataset

HORIZONS = ("roll3", "roll7", "expanding")
_WINDOW = {"roll3": 3, "roll7": 7, "expanding": None}


def _train_mask(day: np.ndarray, d: int, window):
    if window is None:
        return day < d
    return (day < d) & (day >= d - window)


@dataclass
class DayBank:
    """Per-candidate frozen predictions on one day plus that day's labels and
    within-day arrival times, all row-aligned and in arrival order."""
    d: int
    y: np.ndarray
    sec_in_day: np.ndarray
    preds: dict                     # horizon name -> np.ndarray of pCTR on this day
    n_train: dict = field(default_factory=dict)
    fit_time: float = 0.0


def _fit_one(X, day_arr, y, sl, d, h, alpha, seed):
    mask = _train_mask(day_arr, d, _WINDOW[h])
    ntr = int(mask.sum())
    n_test = sl.stop - sl.start
    if ntr == 0 or len(np.unique(y[mask])) < 2:
        return d, h, np.full(n_test, float(y[sl].mean() if n_test else 0.0)), ntr
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=alpha, random_state=seed)
    clf.fit(X[mask], y[mask])
    return d, h, clf.predict_proba(X[sl])[:, 1], ntr


def build_bank(ds: Dataset, eval_days, alpha: float = 1e-4, seed: int = 0,
               n_jobs: int = 4, verbose: bool = True) -> dict:
    """Fit every HORIZON candidate for every day in ``eval_days`` (train mask
    = days < d, restricted to the horizon window). Returns {d: DayBank}."""
    eval_days = sorted(int(d) for d in eval_days if (ds.day_slice(d).stop > ds.day_slice(d).start))
    slices = {d: ds.day_slice(d) for d in eval_days}
    jobs = [(d, h) for d in eval_days for h in HORIZONS]
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_fit_one)(ds.X, ds.day, ds.y, slices[d], d, h, alpha, seed) for d, h in jobs)

    bank = {}
    for d in eval_days:
        sl = slices[d]
        bank[d] = DayBank(d=d, y=ds.y[sl].astype(np.int8), sec_in_day=ds.sec_in_day[sl], preds={})
    for d, h, pred, ntr in results:
        bank[d].preds[h] = pred
        bank[d].n_train[h] = ntr
    for d in eval_days:
        bank[d].fit_time = (time.time() - t0) / len(eval_days)
        if verbose:
            sl = slices[d]
            print(f"  day {d:3d}: n={sl.stop - sl.start:>8d}  "
                  + "  ".join(f"{h}={bank[d].n_train[h]}" for h in HORIZONS), flush=True)
    print(f"  bank: {len(jobs)} fits in {time.time() - t0:.1f}s", flush=True)
    return bank


def _perday_logloss(y, p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def adaptive_weights(bank: dict, days, eta: float = 60.0, halflife: float = 5.0):
    """Exponential-weights mixture weight vector for each day in ``days``,
    computed causally from candidate day-losses on strictly earlier days:
    ``w_h proportional to exp(-eta * Lbar_h)`` with ``Lbar_h`` the
    half-life-discounted mean per-day log loss of candidate ``h`` over days
    ``e < d``. ``eta`` is large by default because log-loss gaps between the
    window candidates are O(0.01); it is also a tuned hyperparameter (dev
    days only). ``eta -> inf`` is follow-the-leader. Returns
    ``{d: {horizon: weight}}``."""
    days = sorted(int(d) for d in days)
    hist_days = sorted(bank)
    out = {}
    for d in days:
        past = [e for e in hist_days if e < d]
        if not past:
            out[d] = {h: 1.0 / len(HORIZONS) for h in HORIZONS}
            continue
        ages = np.array([d - e for e in past], dtype=float)
        disc = np.exp(-np.log(2) / halflife * ages)
        lbar = {}
        for h in HORIZONS:
            losses = np.array([_perday_logloss(bank[e].y, bank[e].preds[h]) for e in past])
            lbar[h] = float(np.average(losses, weights=disc))
        base = min(lbar.values())
        raw = {h: np.exp(-eta * (lbar[h] - base)) for h in HORIZONS}
        z = sum(raw.values())
        out[d] = {h: raw[h] / z for h in HORIZONS}
    return out


def long_term_predictions(bank: dict, days, mode: str, weights: dict = None):
    """q_{d,i} for each day. ``mode`` is a HORIZON name (fixed window),
    ``"equal"`` (uniform average -- the plan's 'equal ensemble'), or
    ``"adaptive"`` (needs ``weights`` from :func:`adaptive_weights`)."""
    days = sorted(int(d) for d in days if d in bank)
    out = {}
    for d in days:
        db = bank[d]
        if mode in HORIZONS:
            q = db.preds[mode]
        elif mode == "equal":
            q = np.mean([db.preds[h] for h in HORIZONS], axis=0)
        elif mode == "adaptive":
            w = weights[d]
            q = sum(w[h] * db.preds[h] for h in HORIZONS)
        else:
            raise ValueError(f"unknown long-term mode {mode!r}")
        out[d] = np.asarray(q, dtype=float)
    return out
