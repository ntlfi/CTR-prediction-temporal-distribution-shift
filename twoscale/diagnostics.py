"""Feasibility diagnostics (plan section 5) -- run before any full search to
quantify how much a scalar within-day calibration can possibly help.

All three are computed on the frozen long-term ``q`` for a set of days
(dev days in the protocol) and are analysis-only.
"""
from __future__ import annotations

import numpy as np

from .calib import SECONDS_PER_DAY, oracle_intercept
from .metrics import day_logloss, _ll


def daily_oracle_improvement(q_by_day, bank, days, B: float = 1.0, eps: float = 1e-5):
    """Section 5.1: per-day uncalibrated loss vs best hindsight fixed intercept."""
    rows = []
    for d in sorted(days):
        if d not in q_by_day:
            continue
        y = bank[d].y
        l_uncal = day_logloss(y, q_by_day[d])
        b_star, l_oracle = oracle_intercept(q_by_day[d], y, B=B, eps=eps)
        rows.append({"day": d, "l_uncal": l_uncal, "l_oracle": l_oracle,
                     "b_star": b_star, "improvement": l_uncal - l_oracle,
                     "rel_improvement": (l_uncal - l_oracle) / l_uncal})
    return rows


def intraday_residual_structure(q_by_day, bank, days, block_sec: int = 1800):
    """Section 5.2: mean residual (y - q) per within-day block, per day, plus
    a run-length summary (are there persistent positive/negative stretches?)."""
    nb = int(np.ceil(SECONDS_PER_DAY / block_sec))
    per_day, runs = [], []
    for d in sorted(days):
        if d not in q_by_day:
            continue
        y = np.asarray(bank[d].y, float)
        q = np.asarray(q_by_day[d], float)
        sec = bank[d].sec_in_day
        blk = np.minimum((sec // block_sec).astype(int), nb - 1)
        series = []
        for k in range(nb):
            m = blk == k
            if m.any():
                r = float((y[m] - q[m]).mean())
                per_day.append({"day": d, "block": k, "n": int(m.sum()), "mean_residual": r})
                series.append(r)
        series = np.array(series)
        if len(series) > 1:
            sign = np.sign(series)
            longest = maxrun = 1
            for i in range(1, len(sign)):
                maxrun = maxrun + 1 if sign[i] == sign[i - 1] and sign[i] != 0 else 1
                longest = max(longest, maxrun)
            runs.append({"day": d, "longest_same_sign_run": int(longest),
                         "n_blocks": int(len(series)),
                         "residual_autocorr_lag1": float(np.corrcoef(series[:-1], series[1:])[0, 1])
                         if len(series) > 2 else np.nan})
    return per_day, runs


def early_late_gain(q_by_day, calibrated_records, bank, days):
    """Section 5.3: does the calibrated method match the long-term model early
    in the day and only diverge later? Large early gains => leakage / artifact.
    Returns per-third pooled log loss for both."""
    from .metrics import early_mid_late
    long_recs = [{"day": d, "y": bank[d].y, "p": q_by_day[d], "sec_in_day": bank[d].sec_in_day}
                 for d in sorted(days) if d in q_by_day]
    cal_recs = [r for r in calibrated_records if r["day"] in set(days)]
    return {"long_term": early_mid_late(long_recs), "calibrated": early_mid_late(cal_recs)}
