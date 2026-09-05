"""Adaptive Rolling Window (ARW), reconstructed from Han, Huang & Wang
(2024), *"Model Assessment and Selection under Temporal Distribution
Shift"* (ICML 2024, arXiv:2402.08672).

**Provenance note**: this is built from the paper's algorithm description
(bias-proxy + Bernstein-variance-proxy window selection, applied to a
pairwise loss-difference sequence, decided by a single-elimination
tournament for >2 candidates) fetched from the paper's text -- it has
*not* been checked against the authors' own source code or a full
line-by-line reading of the paper, so treat the exact constants here as a
faithful-effort reconstruction, not a verified reproduction.

Applied here to select, causally, one of the three shared experts
(``roll3`` / ``roll7`` / ``expanding``) for each prediction day -- the
adaptive-window analogue of "Best Fixed Window" (which instead commits to
one window, once, via dev loss).
"""
from __future__ import annotations

import math

import numpy as np

EXPERTS = ("roll3", "roll7", "expanding")


def _bernstein_halfwidth(sample_var: float, k: int, M: float, delta: float) -> float:
    """Empirical-Bernstein confidence half-width (Maurer & Pontil, 2009):
    for a mean of ``k`` iid-ish observations bounded to range ``M``,
    ``|mu_hat - mu| <= this`` w.p. >= 1 - delta."""
    if k <= 0:
        return float("inf")
    log_term = math.log(2.0 / delta)
    variance_term = math.sqrt(2.0 * sample_var * log_term / k)
    range_term = 7.0 * M * log_term / (3.0 * max(k - 1, 1))
    return variance_term + range_term


def _select_window(u: np.ndarray, delta_prime: float, M: float):
    """Algorithm 1: adaptive window length for the running mean of ``u``
    (chronological, ``u[-1]`` = most recent day). Returns ``(k_hat,
    mu_hat)``. Every quantity here only ever looks at ``u``, which the
    caller must have already restricted to days strictly before the day
    being decided."""
    t = len(u)
    mu = np.empty(t + 1)   # mu[k] = mean of the last k observations, mu[0] unused
    psi = np.empty(t + 1)
    for k in range(1, t + 1):
        window = u[t - k:t]
        mu[k] = window.mean()
        var_k = float(window.var(ddof=1)) if k > 1 else 0.0
        psi[k] = _bernstein_halfwidth(var_k, k, M, delta_prime)

    best_k, best_score = 1, float("inf")
    for k in range(1, t + 1):
        bias_proxy = max((abs(mu[k] - mu[i]) - (psi[k] + psi[i])) for i in range(1, k + 1))
        bias_proxy = max(bias_proxy, 0.0)
        score = bias_proxy + psi[k]
        if score < best_score:
            best_k, best_score = k, score
    return best_k, float(mu[best_k])


def _estimate_M(u: np.ndarray, floor: float = 1e-6) -> float:
    """Empirical range bound, from past data only (the protocol's
    instruction -- no theoretical worst-case log-loss bound is assumed)."""
    if len(u) == 0:
        return floor
    return max(float(np.max(u) - np.min(u)), floor)


def pairwise_prefers_first(losses_a: np.ndarray, losses_b: np.ndarray, delta: float,
                           n_candidates: int = len(EXPERTS)) -> bool:
    """Algorithm 2: compares two experts using only their past per-day
    losses (both arrays already restricted to days < d, same length,
    same day order). ``True`` if the adaptive-window estimate of
    ``L(a) - L(b)`` is <= 0 (prefer ``a``)."""
    u = np.asarray(losses_a, float) - np.asarray(losses_b, float)
    delta_prime = delta / (3.0 * n_candidates ** 2 * max(len(u), 1))
    M = _estimate_M(u)
    _, mu_hat = _select_window(u, delta_prime, M)
    return mu_hat <= 0.0


def select_expert(loss_history: dict, delta: float, min_history: int = 3,
                  fallback: str = "expanding") -> str:
    """Algorithm 3: single-elimination tournament over ``EXPERTS``, using
    only ``loss_history[name]`` = that expert's past per-day losses
    (chronological, days < d, identical length/order across experts).
    Falls back to ``fallback`` (a benign default) before ``min_history``
    days of history exist."""
    lengths = {len(loss_history[name]) for name in EXPERTS}
    if min(lengths) < min_history:
        return fallback

    order = list(EXPERTS)
    current = order[0]
    for challenger in order[1:]:
        prefers_current = pairwise_prefers_first(
            np.asarray(loss_history[current]), np.asarray(loss_history[challenger]), delta)
        current = current if prefers_current else challenger
    return current
