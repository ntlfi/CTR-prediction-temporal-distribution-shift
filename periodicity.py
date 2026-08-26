"""Causal periodicity detection and phase features.

Every method in this project so far (P1/P2's han_arw/diff_forgetting/adamoe,
and this project's own M1/M2/M5b) shares one blind spot under `recurring`
drift (results/synthetic_analysis.md, results/m5_analysis.md): none of them
carry any notion of *where in a cycle* the process currently is, only *how
much* recent history to trust. This module estimates a dominant period
causally -- using only days < t, the same rule every gate in this project
already follows -- from a per-day loss signal already available in the
WINDOW_FAMILY candidate bank, and turns it into sin/cos phase features a
gate can condition on.

Signal choice: `synthetic_data.py`'s recurring schedule mixes two fixed
regimes with weight alpha(t) = 0.5*(1+sin(2*pi*t/period)) -- a smooth
sinusoid -- so *any* fixed-window candidate's per-day mean log loss
inherits that same periodicity (it's worse when the current true regime is
further from what the candidate was fit on). `expanding` is used here
because it's the one WINDOW_FAMILY candidate with a prediction for every
eligible day from the start, giving the longest available signal.
"""
import numpy as np


def autocorrelation_period(signal: np.ndarray, min_period: int = 3, max_period: int = 25,
                            min_history: int = None, min_acf: float = 0.3):
    """Estimate the dominant period of `signal` (one value per day, in day
    order -- callers must only ever pass days < t) via normalized
    autocorrelation over lags in [min_period, max_period]. Returns None if
    there isn't enough history yet, or the best lag's autocorrelation
    doesn't clear `min_acf`. `min_history` defaults to 2x max_period: want
    to see the longest candidate period repeat at least twice before
    trusting it (this project's synthetic horizon is only 120 days, so
    requiring 3+ repeats of a 25-day max search window would leave almost
    no test days with a detected period at all).

    Calibration note: an earlier version of this significance test required
    only "positive and >= 2x the median autocorrelation across the searched
    lags" -- on 200 trials of pure white noise (n=80, same lag range) that
    rule fired on effectively 100% of trials, because the *median*
    autocorrelation across lags is typically slightly negative, making
    "2x the median" a trivial bar for any positive value to clear. A flat
    absolute floor on the peak autocorrelation itself (0.3, well below a
    clean synthetic cycle's ~0.8 peak but far above white noise's peak
    lag-to-lag autocorrelation) brought the false-positive rate on the same
    noise trials down to ~2%."""
    if min_history is None:
        min_history = 2 * max_period
    n = len(signal)
    if n < max(min_history, 2 * min_period):
        return None
    s = np.asarray(signal, dtype=float)
    s = s - s.mean()
    if np.allclose(s, 0):
        return None
    denom = (s ** 2).sum()
    if denom == 0:
        return None
    lags = list(range(min_period, min(max_period, n - 1) + 1))
    if not lags:
        return None
    acf = np.array([np.dot(s[:-lag], s[lag:]) / denom for lag in lags])
    best_i = int(acf.argmax())
    if acf[best_i] < min_acf:
        return None
    return lags[best_i]


def phase_features(t: int, period) -> tuple:
    """sin/cos of day t's phase within `period`. Returns (0.0, 0.0) --
    a neutral "no periodicity signal" input, not a misleading arbitrary
    phase -- if `period` is None (nothing confidently detected yet)."""
    if period is None or period <= 0:
        return 0.0, 0.0
    phase = 2 * np.pi * (t / period)
    return float(np.sin(phase)), float(np.cos(phase))


def causal_period_series(loss_by_day: dict, day_list: list, min_period: int = 3, max_period: int = 40,
                          min_history: int = None) -> dict:
    """Precompute the causal period estimate available at each prediction
    day t in `day_list`, using only strictly-earlier days' losses from
    `loss_by_day` (day -> scalar). Returns {t: period_or_None}. O(T) calls
    to autocorrelation_period, each O(T*max_period) -- fine at this
    project's scale (<=120 days)."""
    ordered = sorted(day_list)
    result = {}
    for t in ordered:
        past_losses = [loss_by_day[d] for d in ordered if d < t and d in loss_by_day]
        result[t] = autocorrelation_period(np.array(past_losses), min_period=min_period,
                                            max_period=max_period, min_history=min_history)
    return result
