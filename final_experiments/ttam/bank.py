"""Shared five-horizon expert bank (plan section 4).

Nominal horizons ``{1, 3, 7, 14, expanding}`` (days). For prediction day
``d`` every expert is a fresh L2-regularised logistic regression
(``SGDClassifier``, log loss) fitted on **days < d only**, restricted to
its own trailing window; identical model, differing only in which rows it
sees, so any difference between experts is the temporal mechanism.

The backbone regularisation ``alpha`` is *not* tuned on any evaluation
data anywhere in this project -- it is fixed a priori at ``1e-4`` (the
value every earlier line in the repo used), which satisfies the plan's
"any backbone setting previously tuned on later data must be reselected
or fixed independently of those data" (section 6): it was fixed
independently of the outer days from the start.

Effective horizons and duplicates (plan section 4). When ``d`` is smaller
than a nominal window the window is truncated to ``d`` days, so e.g.
``roll14`` and ``expanding`` coincide whenever there are at most 14 days
of history (always, on Avazu's ten-day stream). Coinciding windows are
**fitted once** and the resulting predictions are aliased to every
nominal slot they cover, so:

  * every method still sees a full five-key ``preds`` dict every day
    (cross-day per-expert loss histories stay well defined), and
  * a mixture that splits weight across two aliased slots produces
    exactly the prediction it would from the single distinct expert
    (``sum_k w_k p_k`` is unchanged by duplicating a column and
    splitting its weight) -- so no method double-counts.

``DayBank5.effective`` records, per day, which distinct fitted expert each
nominal slot resolves to, and ``DayBank5.n_effective`` the distinct count
(reported in the run manifest / appendix).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import SGDClassifier

from twoscale.data import Dataset

HORIZONS5 = ("roll1", "roll3", "roll7", "roll14", "expanding")
_WINDOW = {"roll1": 1, "roll3": 3, "roll7": 7, "roll14": 14, "expanding": None}
BACKBONE_ALPHA = 1e-4


def _window_bounds(d: int, window):
    """Inclusive [lo, hi) day range of the trailing window for day ``d``
    (history is days ``0 .. d-1``). ``window is None`` -> expanding."""
    hi = d
    lo = 0 if window is None else max(0, d - window)
    return lo, hi


def effective_key(d: int, h: str) -> tuple:
    """A hashable key identifying the *distinct* training-row set a nominal
    horizon resolves to on day ``d``. Two nominal horizons with the same
    key are the same fitted model."""
    return _window_bounds(d, _WINDOW[h])


@dataclass
class DayBank5:
    d: int
    y: np.ndarray
    sec_in_day: np.ndarray
    preds: dict                         # nominal horizon -> pCTR array (aliased on ties)
    effective: dict = field(default_factory=dict)   # nominal -> effective-expert id (int)
    n_train: dict = field(default_factory=dict)
    n_effective: int = 0
    fit_time: float = 0.0


def _fit_one(X, day_arr, y, sl, lo, hi, alpha, seed):
    """Fit the logistic regression whose training rows are days [lo, hi)."""
    mask = (day_arr >= lo) & (day_arr < hi)
    ntr = int(mask.sum())
    n_test = sl.stop - sl.start
    if ntr == 0 or len(np.unique(y[mask])) < 2:
        base = float(y[sl].mean()) if n_test else 0.0
        return (lo, hi), np.full(n_test, base), ntr
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=alpha, random_state=seed)
    clf.fit(X[mask], y[mask])
    return (lo, hi), clf.predict_proba(X[sl])[:, 1], ntr


def build_bank5(ds: Dataset, eval_days, alpha: float = BACKBONE_ALPHA, seed: int = 0,
                n_jobs: int = 4, verbose: bool = False) -> dict:
    """Fit every distinct effective expert for every day in ``eval_days``
    (train rows = days < d within the horizon window). Returns
    ``{d: DayBank5}``. Distinct ``(lo, hi)`` training ranges are fitted
    once and shared across the days / nominal slots that need them."""
    eval_days = sorted(int(d) for d in eval_days
                       if ds.day_slice(d).stop > ds.day_slice(d).start)
    slices = {d: ds.day_slice(d) for d in eval_days}

    # one job per (day, distinct (lo,hi)) -- dedups roll14==expanding etc.
    jobs = []
    seen = set()
    for d in eval_days:
        for h in HORIZONS5:
            lo, hi = _window_bounds(d, _WINDOW[h])
            key = (d, lo, hi)
            if key not in seen:
                seen.add(key)
                jobs.append((d, lo, hi))

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_fit_one)(ds.X, ds.day, ds.y, slices[d], lo, hi, alpha, seed)
        for d, lo, hi in jobs)
    fitted = {(d, lo, hi): (pred, ntr)
              for (d, lo, hi), ((lo2, hi2), pred, ntr) in zip(jobs, results)}

    bank = {}
    for d in eval_days:
        sl = slices[d]
        eff_ids, eff_of = {}, {}
        preds, n_train = {}, {}
        for h in HORIZONS5:
            lo, hi = _window_bounds(d, _WINDOW[h])
            if (lo, hi) not in eff_ids:
                eff_ids[(lo, hi)] = len(eff_ids)
            eff_of[h] = eff_ids[(lo, hi)]
            pred, ntr = fitted[(d, lo, hi)]
            preds[h] = pred
            n_train[h] = ntr
        bank[d] = DayBank5(d=d, y=ds.y[sl].astype(np.int8), sec_in_day=ds.sec_in_day[sl],
                           preds=preds, effective=eff_of, n_train=n_train,
                           n_effective=len(eff_ids),
                           fit_time=(time.time() - t0) / max(len(eval_days), 1))
        if verbose:
            print(f"  day {d:3d}: n={sl.stop - sl.start:>9d}  "
                  f"n_eff={len(eff_ids)}  "
                  + "  ".join(f"{h}={n_train[h]}" for h in HORIZONS5), flush=True)
    print(f"  bank5: {len(jobs)} distinct fits ({len(eval_days)} days) "
          f"in {time.time() - t0:.1f}s", flush=True)
    return bank


def distinct_columns(db: DayBank5):
    """(names, matrix) of the distinct effective expert predictions for one
    day -- one column per distinct fitted model, plus the list of nominal
    names that map to each. Used by the mixing / selection methods so
    weights live on the distinct set."""
    by_eff = {}
    for h in HORIZONS5:
        by_eff.setdefault(db.effective[h], []).append(h)
    order = sorted(by_eff)
    cols = np.stack([db.preds[by_eff[e][0]] for e in order], axis=1)
    groups = [by_eff[e] for e in order]
    return groups, cols
