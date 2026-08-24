"""Fit the P0 window-family candidates (expanding + rolling 1/3/7/14) once
for every prediction day, keeping per-sample predictions.

Han ARW (methods/han_arw.py) and AdaMoE (methods/adamoe.py) both select
among / aggregate these same candidates rather than retraining their own
base model, so the fits are computed once here and shared instead of
duplicating ~2000s of SGDClassifier fitting per method.
"""
from joblib import Parallel, delayed

from baselines import WINDOW_FAMILY, build_candidates, fit_predict


def build_candidate_bank(X, y, day, days, alpha=1e-4, seed=0, n_jobs=1):
    """Returns bank[method][t] = fit_predict(...) result dict, restricted to
    the WINDOW_FAMILY methods (the only candidates P1's Han ARW and P2's
    AdaMoE select among / aggregate over, per the PDF's baseline ladder).
    """
    all_candidates = build_candidates()
    candidates = {name: all_candidates[name] for name in WINDOW_FAMILY}

    jobs = [(name, t) for t in days for name in candidates]
    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_predict)(X, y, day, t, candidates[name], alpha=alpha, seed=seed)
        for name, t in jobs
    )

    bank = {name: {} for name in candidates}
    for (name, t), result in zip(jobs, results):
        if result is not None:
            bank[name][t] = result
    return bank
