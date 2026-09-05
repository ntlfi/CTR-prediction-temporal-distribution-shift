"""Day-level statistical treatment (rolling-protocol section 5). The
calendar day is the unit of inference throughout this module -- never a
raw impression count, never a random-init/data-sampling seed. Every
function here takes (or returns) one number per *day*.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from twoscale.metrics import bootstrap_paired_ci


def leave_one_day_out(deltas) -> np.ndarray:
    deltas = np.asarray(deltas, float)
    n = len(deltas)
    if n < 2:
        return np.array([])
    return np.array([np.mean(np.delete(deltas, i)) for i in range(n)])


def moving_block_bootstrap_ci(deltas, block: int = 2, n_boot: int = 5000, seed: int = 0):
    """Moving-block bootstrap over *consecutive* days, respecting possible
    serial dependence between neighbors. ``None`` if there are fewer than
    ``2 * block`` days -- report as "skipped: too few days" rather than a
    number that overstates precision."""
    deltas = np.asarray(deltas, float)
    n = len(deltas)
    if n < 2 * block:
        return None
    rng = np.random.default_rng(seed)
    n_blocks_needed = int(np.ceil(n / block))
    starts = np.arange(0, n - block + 1)
    means = []
    for _ in range(n_boot):
        idx = rng.choice(starts, size=n_blocks_needed, replace=True)
        sample = np.concatenate([deltas[s:s + block] for s in idx])[:n]
        means.append(sample.mean())
    means = np.asarray(means)
    return {"mean": float(deltas.mean()), "ci95_lo": float(np.percentile(means, 2.5)),
           "ci95_hi": float(np.percentile(means, 97.5)), "block": block, "n_boot": n_boot}


def day_summary(deltas, seed: int = 0) -> dict:
    """``deltas[i] = L_d(v5) - L_d(baseline)`` for day i (negative favors
    V5). The single object every other summary in this module is built
    from."""
    deltas = np.asarray(deltas, float)
    n = len(deltas)
    mean, lo, hi = bootstrap_paired_ci(deltas, seed=seed) if n else (float("nan"),) * 3
    n_win = int(np.sum(deltas < 0))
    sign_p = float(stats.binomtest(n_win, n, 0.5, alternative="two-sided").pvalue) if n else float("nan")
    loo = leave_one_day_out(deltas)
    mbb = moving_block_bootstrap_ci(deltas, seed=seed)
    return {
        "n_days": n,
        "mean_delta": mean,
        "median_delta": float(np.median(deltas)) if n else float("nan"),
        "ci95_lo": lo, "ci95_hi": hi,
        "n_days_won": n_win,
        "frac_days_won": n_win / n if n else float("nan"),
        "sign_test_p": sign_p,
        "worst_day_delta": float(np.max(deltas)) if n else float("nan"),
        "loo_mean_min": float(np.min(loo)) if len(loo) else float("nan"),
        "loo_mean_max": float(np.max(loo)) if len(loo) else float("nan"),
        "loo_reverses_sign": bool(n > 1 and np.any(np.sign(loo) != np.sign(mean))),
        "moving_block_bootstrap": mbb,
    }


def impression_weighted_effect(n_impressions, ll_v5, ll_baseline) -> float:
    """Pools days by their own impression count -- the "impression-weighted
    aggregate difference," distinct from (and reported alongside) the
    equal-day-weighted mean in ``day_summary``."""
    n_impressions = np.asarray(n_impressions, float)
    ll_v5 = np.asarray(ll_v5, float)
    ll_baseline = np.asarray(ll_baseline, float)
    total = n_impressions.sum()
    if total == 0:
        return float("nan")
    return float(np.sum((ll_v5 - ll_baseline) * n_impressions) / total)
