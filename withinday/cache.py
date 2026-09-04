"""Stage A: one reusable causal replay cache (plan section 5.1).

The frozen ``q``, ``y`` and ``sec_in_day`` stored here are byte-identical to
what the twoscale baselines (``long_only`` / ``combined`` / ``online_platt``
/ ``time_of_day``) are scored on -- they come straight from
``twoscale.longterm.DayBank`` and ``twoscale.longterm.long_term_predictions``.
That makes "reproduce the twoscale baselines off this cache" (execution-order
step 1) a real end-to-end identity check on day-slicing and row alignment,
not just a unit test: see ``withinday_run.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from twoscale.data import Dataset

from .blocks import build_block_tokens, deterministic_summary, last_available_block
from .contextsketch import build_projection, context_sketch

SECONDS_PER_DAY = 86_400


def _logit(p, eps=1e-5):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


@dataclass
class DayCache:
    d: int
    y: np.ndarray                # (n,)
    q: np.ndarray                # (n,)  frozen long-term prediction
    sec_in_day: np.ndarray       # (n,)
    csketch: np.ndarray          # (n, m)          c(x_{d,i})
    a: np.ndarray                # (n, m + 2)      current-impression input, eq 6
    block_tokens: np.ndarray     # (n_blocks, token_dim)   u_k, eq 5
    block_summary: np.ndarray    # (n_blocks, summary_dim) s_k, eq 11
    k_avail: np.ndarray          # (n,) int, last matured block index (-1 = none yet)


def build_day_cache(d: int, q, y, sec_in_day, X_day, block_sec: int, delay_sec: int,
                    m: int, R, eps: float = 1e-5) -> DayCache:
    q = np.asarray(q, float)
    y = np.asarray(y, float)
    sec_in_day = np.asarray(sec_in_day, float)
    csketch = context_sketch(X_day, m=m, R=R)
    tokens = build_block_tokens(q, y, sec_in_day, csketch, block_sec, eps=eps)
    summary = deterministic_summary(tokens)
    k_avail = last_available_block(sec_in_day, block_sec, delay_sec)
    time_of_day = sec_in_day / SECONDS_PER_DAY
    a = np.concatenate([csketch, _logit(q, eps)[:, None], time_of_day[:, None]], axis=1)
    return DayCache(d=d, y=y, q=q, sec_in_day=sec_in_day, csketch=csketch, a=a,
                    block_tokens=tokens, block_summary=summary, k_avail=k_avail)


def build_cache(ds: Dataset, bank: dict, q_by_day: dict, days,
                block_sec: int = 900, delay_sec: int = 1800, m: int = 32,
                seed: int = 0) -> dict:
    """One :class:`DayCache` per day in ``days``, all sharing the same fixed
    context projection ``R`` (so ``c(x)`` means the same thing every day)
    and reading ``q``/``y``/``sec_in_day`` straight from the twoscale bank
    -- the "same long-term backbone and base predictor across all variants"
    rule (plan section 2.3)."""
    R = build_projection(ds.X.shape[1], m, seed=seed)
    out = {}
    for d in sorted(int(dd) for dd in days):
        if d not in bank or d not in q_by_day:
            continue
        sl = ds.day_slice(d)
        out[d] = build_day_cache(d, q_by_day[d], bank[d].y, bank[d].sec_in_day,
                                 ds.X[sl], block_sec, delay_sec, m, R)
    return out
