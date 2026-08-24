"""Han, Huang & Wang (ICML 2024), "Model Assessment and Selection under
Temporal Distribution Shift" -- adaptive rolling window baseline (PDF 3.5).

Reproduces their Algorithm 1 (windowed mean estimation with a
Goldenshluger-Lepski bias/variance-adaptive window), Algorithm 2 (pairwise
model comparison via Algorithm 1 applied to the per-sample loss
difference), and Algorithm 3 (single-elimination tournament over m models
via repeated Algorithm 2). Candidate models here are the existing P0
window-family fits (expanding + rolling 1/3/7/14); the tournament picks
among their already-computed predictions for the current day, using only
the loss trajectories of earlier days -- i.e. it is a *selection* rule
over pretrained candidates, run fresh for every prediction day, not a
new base model. That is what "data-dependent global history length"
means here: the effective window is re-decided every day rather than
frozen once like the P0 validation-selected baseline.

Reference implementation: https://github.com/eliselyhan/ARW
"""
import math

import numpy as np

from baselines import WINDOW_FAMILY

EPS = 1e-12


def per_sample_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    p = np.clip(y_pred, EPS, 1 - EPS)
    return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def _period_stats(u: np.ndarray):
    """Sufficient stats for one period: (B, mean, mean-of-squares)."""
    return len(u), float(u.mean()), float((u ** 2).mean())


def _windowed_mean_var(stats: list, k: int):
    """Pool the most recent `k` periods (paper eq. 3.7-3.8): returns
    (B_{t,k}, mu_hat_{t,k}, v_hat_{t,k})."""
    sub = stats[-k:]
    B = sum(s[0] for s in sub)
    mu = sum(s[0] * s[1] for s in sub) / B
    sum_sq = sum(s[0] * s[2] for s in sub)
    if B > 1:
        # sum_i (u_i - mu)^2 = sum_j B_j*sqmean_j - 2*mu*sum_j B_j*mean_j + B*mu^2
        ss = sum_sq - 2 * mu * sum(s[0] * s[1] for s in sub) + B * mu ** 2
        var = max(ss, 0.0) / (B - 1)
    else:
        var = 0.0
    return B, mu, math.sqrt(var)


def _psi(B: int, v: float, delta: float, M: float) -> float:
    """Stochastic-error proxy (paper eq. between 3.7-3.8)."""
    if B < 2:
        return M
    return v * math.sqrt(2 * math.log(2 / delta) / B) + 8 * M * math.log(2 / delta) / (3 * (B - 1))


def select_k(stats: list, delta: float, M: float):
    """Algorithm 1: Goldenshluger-Lepski adaptive window selection.
    `stats` is a list of per-period (B, mean, sqmean), oldest first.
    Returns (k_hat, mu_hat_{t,k_hat})."""
    T = len(stats)
    # Precompute (B,mu,v,psi) for every candidate k once -- O(T) windowed means.
    cache = {}
    for k in range(1, T + 1):
        B, mu, v = _windowed_mean_var(stats, k)
        cache[k] = (B, mu, v, _psi(B, v, delta, M))

    best_k, best_score = 1, float("inf")
    for k in range(1, T + 1):
        _, muk, _, psik = cache[k]
        bias = max((abs(muk - cache[i][1]) - (psik + cache[i][3]) for i in range(1, k + 1)), default=0.0)
        bias = max(bias, 0.0)
        score = bias + psik
        if score < best_score:
            best_score, best_k = score, k
    return best_k, cache[best_k][1]


def compare_pair(stats_a: list, stats_b: list, delta: float, M: float) -> str:
    """Algorithm 2: paired per-sample loss-difference series (a - b, oldest
    first), Algorithm-1-selected window, sign of the estimate decides."""
    diff_stats = [
        (Ba, mu_a - mu_b, sq_a - 2 * mu_a * mu_b + sq_b)
        for (Ba, mu_a, sq_a), (Bb, mu_b, sq_b) in zip(stats_a, stats_b)
    ]
    _, delta_hat = select_k(diff_stats, delta, M)
    return "a" if delta_hat <= 0 else "b"


def tournament(candidate_names: list, per_day_stats: dict, delta: float, M: float) -> str:
    """Algorithm 3: single-elimination tournament (m-1 pairwise comparisons
    via Algorithm 2), reigning-champion bracket over `candidate_names`."""
    champ = candidate_names[0]
    for challenger in candidate_names[1:]:
        winner = compare_pair(per_day_stats[champ], per_day_stats[challenger], delta, M)
        champ = champ if winner == "a" else challenger
    return champ


def run_han_arw(bank: dict, eligible_days, dev_days=None, min_history: int = 3, delta: float = 0.1):
    """For each prediction day t (once >= min_history earlier days have
    candidate-bank results), run the ARW tournament over days < t and serve
    the winning candidate's already-computed prediction for day t.

    Returns rows: {day, y_true, y_pred, n_train, fit_time, selected_window}.
    """
    names = [n for n in WINDOW_FAMILY]
    # Empirical loss bound M (PDF/paper leave this a free implementation
    # choice; the authors themselves just fix M=0 in their own experiments).
    # Restricted to dev-period days only, per the "hyperparameters tuned only
    # on development periods" leakage rule (PDF section 4, 8) -- locked test
    # days never inform M.
    m_days = set(dev_days) if dev_days is not None else set(bank[names[0]])
    all_losses = [
        per_sample_log_loss(r["y_true"], r["y_pred"])
        for name in names for d, r in bank[name].items() if d in m_days
    ]
    M = float(max(losses.max() for losses in all_losses)) + 1e-6 if all_losses else 1.0

    rows = []
    for t in eligible_days:
        history_days = sorted(d for d in bank[names[0]] if d < t and all(d in bank[n] for n in names))
        if len(history_days) < min_history:
            continue
        if not all(t in bank[n] for n in names):
            continue

        per_day_stats = {
            name: [_period_stats(per_sample_log_loss(bank[name][d]["y_true"], bank[name][d]["y_pred"]))
                   for d in history_days]
            for name in names
        }
        selected = tournament(names, per_day_stats, delta, M)
        result = bank[selected][t]
        rows.append({
            "day": t,
            "y_true": result["y_true"],
            "y_pred": result["y_pred"],
            "n_train": result["n_train"],
            "fit_time": result["fit_time"],
            "selected_window": selected,
        })
    return rows
