"""Metrics and statistical analysis (plan section 7 and 8).

Everything operates on per-day prediction records
``{"day": d, "y": array, "p": array, "sec_in_day": array}`` so a method is
just a list of those.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def _ll(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def day_logloss(y, p):
    return float(_ll(y, p).mean())


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def ece(y, p, n_bins: int = 15):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def per_day_frame(records):
    """DataFrame-ready rows: one per day with log loss / brier / ece / n."""
    rows = []
    for r in records:
        rows.append({"day": int(r["day"]), "n": len(r["y"]),
                     "log_loss": day_logloss(r["y"], r["p"]),
                     "brier": brier(r["y"], r["p"]), "ece": ece(r["y"], r["p"]),
                     "clicks": int(np.sum(r["y"]))})
    return rows


def impression_weighted_logloss(records):
    tot = np.concatenate([_ll(r["y"], r["p"]) for r in records]) if records else np.array([])
    return float(tot.mean()) if len(tot) else float("nan")


def unweighted_daily_logloss(records):
    return float(np.mean([day_logloss(r["y"], r["p"]) for r in records])) if records else float("nan")


def paired_day_diffs(a_records, b_records):
    """Delta_d = L_d(a) - L_d(b), impression-weighted within the day, aligned
    on day index. Returns (days, deltas)."""
    bd = {r["day"]: r for r in b_records}
    days, deltas = [], []
    for r in a_records:
        if r["day"] not in bd:
            continue
        days.append(r["day"])
        deltas.append(day_logloss(r["y"], r["p"]) - day_logloss(bd[r["day"]]["y"], bd[r["day"]]["p"]))
    return np.array(days), np.array(deltas)


def bootstrap_paired_ci(deltas, n_boot: int = 5000, seed: int = 0):
    d = np.asarray(deltas, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def days_won(a_records, b_records):
    """Fraction of days where method a has strictly lower day log loss than b."""
    _, deltas = paired_day_diffs(a_records, b_records)
    return float(np.mean(deltas < 0)) if len(deltas) else float("nan"), int(np.sum(deltas < 0)), len(deltas)


def intraday_block_frame(records, block_sec: int = 3600):
    """Mean residual (y - p) and log loss per within-day block, pooled over
    days (plan section 7.2)."""
    from .calib import SECONDS_PER_DAY
    nb = int(np.ceil(SECONDS_PER_DAY / block_sec))
    acc = {k: {"res": [], "ll": [], "n": 0} for k in range(nb)}
    for r in records:
        blk = np.minimum((np.asarray(r["sec_in_day"]) // block_sec).astype(int), nb - 1)
        ll = _ll(r["y"], r["p"])
        res = np.asarray(r["y"], float) - np.asarray(r["p"], float)
        for k in range(nb):
            m = blk == k
            if m.any():
                acc[k]["res"].append(res[m].sum())
                acc[k]["ll"].append(ll[m].sum())
                acc[k]["n"] += int(m.sum())
    rows = []
    for k in range(nb):
        if acc[k]["n"] == 0:
            continue
        rows.append({"block": k, "n": acc[k]["n"],
                     "mean_residual": float(np.sum(acc[k]["res"]) / acc[k]["n"]),
                     "log_loss": float(np.sum(acc[k]["ll"]) / acc[k]["n"])})
    return rows


def early_mid_late(records, thirds=(1 / 3, 2 / 3)):
    """Log loss over the first / middle / last third of each day, pooled
    (plan section 5.3 / 7.2). A causal within-day calibrator should help most
    in the late third."""
    from .calib import SECONDS_PER_DAY
    seg = {"early": [], "mid": [], "late": []}
    for r in records:
        frac = np.asarray(r["sec_in_day"], float) / SECONDS_PER_DAY
        ll = _ll(r["y"], r["p"])
        for name, m in (("early", frac < thirds[0]),
                        ("mid", (frac >= thirds[0]) & (frac < thirds[1])),
                        ("late", frac >= thirds[1])):
            if m.any():
                seg[name].append((ll[m].sum(), int(m.sum())))
    out = {}
    for name, parts in seg.items():
        if parts:
            s = np.sum([p[0] for p in parts]); c = np.sum([p[1] for p in parts])
            out[name] = float(s / c)
    return out


def regret_and_captured_gain(method_records, long_records, q_by_day, B: float, eps: float,
                             denom_floor: float = 1e-4):
    """R_d = L_d(method) - L_d(oracle b*)  (plan eq 13) and
    CapturedGain_d = (L_long - L_method) / (L_long - L_oracle)  (eq 14),
    per day, using the frozen long-term q the method was built on
    (``q_by_day`` maps day -> q array). Days whose denominator
    ``L_long - L_oracle`` is below ``denom_floor`` are marked negligible and
    excluded from the captured-gain average (plan section 8). CapturedGain
    can exceed 1 or go negative on kept days when the method's within-day
    trajectory beats a single fixed intercept -- that is reported, not clipped."""
    from .calib import oracle_intercept
    lr = {r["day"]: r for r in long_records}
    rows = []
    for r in method_records:
        d = r["day"]
        if d not in lr or d not in q_by_day:
            continue
        _, l_or = oracle_intercept(q_by_day[d], lr[d]["y"], B=B, eps=eps)
        l_long = day_logloss(lr[d]["y"], lr[d]["p"])
        l_m = day_logloss(r["y"], r["p"])
        denom = l_long - l_or
        negligible = denom < denom_floor
        rows.append({"day": d, "regret": l_m - l_or,
                     "l_long": l_long, "l_oracle": l_or, "l_method": l_m,
                     "captured_gain": (l_long - l_m) / denom if not negligible else np.nan,
                     "denom_negligible": negligible})
    return rows
