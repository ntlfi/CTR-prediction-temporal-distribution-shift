"""Causal per-block history representation (plan section 2.2, eq 3-6, 11).

A day is partitioned into fixed-width blocks. A block enters the causal
history of a later impression only after the block is complete *and* its
feedback-maturation delay has elapsed (eq 3) -- ``last_available_block``
is the one clock all capacity-ladder variants share for "how much history
can impression i see." Everything below it (the per-block token, the
deterministic level/trend summary) is a pure function of already-elapsed
blocks, so it can be precomputed once per day and then just *looked up* per
impression by every variant (V1's attention window, V2's GRU state, V3-V5's
summary vector) -- see ``withinday/cache.py``.
"""
from __future__ import annotations

import numpy as np

SECONDS_PER_DAY = 86_400

# plan eq 5, before the context-sketch halves: count, mean label, mean pred,
# mean logit(pred), mean residual, mean |residual|, mean per-impression log
# loss. The two sketch halves (mean c(x), mean r*c(x)) are appended after.
TOKEN_SCALAR_FIELDS = (
    "log1p_n", "mean_y", "mean_q", "mean_logit_q", "mean_r", "mean_absr", "mean_logloss",
)


def _logit(p, eps=1e-5):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def n_blocks_per_day(block_sec: int) -> int:
    return int(np.ceil(SECONDS_PER_DAY / block_sec))


def token_dim(m: int) -> int:
    return len(TOKEN_SCALAR_FIELDS) + 2 * m


def block_of(sec_in_day, block_sec: int) -> np.ndarray:
    nb = n_blocks_per_day(block_sec)
    return np.minimum((np.asarray(sec_in_day) // block_sec).astype(int), nb - 1)


def last_available_block(sec_in_day, block_sec: int, delay_sec: int) -> np.ndarray:
    """``k(i)``: index of the most recently *matured* block at prediction
    time ``sec_in_day[i]`` -- the largest ``k`` with
    ``(k + 1) * block_sec + delay_sec <= sec_in_day[i]``; ``-1`` if no block
    has matured yet (including the impossible-to-satisfy case
    ``block_sec + delay_sec > SECONDS_PER_DAY``, e.g. very long delays)."""
    sec = np.asarray(sec_in_day, dtype=float)
    k = np.floor((sec - delay_sec) / block_sec).astype(int) - 1
    nb = n_blocks_per_day(block_sec)
    return np.clip(k, -1, nb - 1)


def build_block_tokens(q, y, sec_in_day, csketch, block_sec: int, eps: float = 1e-5) -> np.ndarray:
    """Per-block token ``u_k`` (eq 5), one row per block of the day. Each
    token is built only from the impressions whose *arrival* falls in that
    block -- causality (which impressions are allowed to *see* a given
    block) is a separate, later step (:func:`last_available_block`); a
    block that has not matured yet still gets a well-defined token, it is
    simply not exposed to any impression until its maturation time."""
    q = np.clip(np.asarray(q, float), eps, 1 - eps)
    y = np.asarray(y, float)
    r = y - q
    ll = -(y * np.log(q) + (1 - y) * np.log(1 - q))
    nb = n_blocks_per_day(block_sec)
    blk = block_of(sec_in_day, block_sec)
    m = csketch.shape[1]
    tokens = np.zeros((nb, token_dim(m)), dtype=float)
    for k in range(nb):
        idx = np.where(blk == k)[0]
        if len(idx) == 0:
            continue
        base = np.array([
            np.log1p(len(idx)), y[idx].mean(), q[idx].mean(), _logit(q[idx], eps).mean(),
            r[idx].mean(), np.abs(r[idx]).mean(), ll[idx].mean(),
        ])
        c_mean = csketch[idx].mean(axis=0)
        rc_mean = (r[idx, None] * csketch[idx]).mean(axis=0)
        tokens[k] = np.concatenate([base, c_mean, rc_mean])
    return tokens


def _ewma_causal(tokens: np.ndarray, halflife_blocks: float) -> np.ndarray:
    """Causal (left-to-right) EWMA over the block axis: row ``k`` depends
    only on blocks ``<= k``."""
    alpha = 1.0 - 0.5 ** (1.0 / halflife_blocks)
    out = np.empty_like(tokens)
    acc = tokens[0].copy()
    out[0] = acc
    for k in range(1, tokens.shape[0]):
        acc = alpha * tokens[k] + (1.0 - alpha) * acc
        out[k] = acc
    return out


def deterministic_summary(tokens: np.ndarray, halflives=(1.0, 4.0, 16.0)) -> np.ndarray:
    """``s_k`` (eq 11): ``[u_k, u_{k-1}, EWMA_h1(u), EWMA_h2(u), EWMA_h3(u),
    u_k - u_{k-1}]``, one row per block, each depending only on blocks
    ``<= k`` (checked by ``withinday_tests.py``)."""
    prev = np.vstack([np.zeros((1, tokens.shape[1])), tokens[:-1]])
    ewmas = [_ewma_causal(tokens, h) for h in halflives]
    return np.concatenate([tokens, prev, *ewmas, tokens - prev], axis=1)


def summary_dim(m: int, n_halflives: int = 3) -> int:
    return token_dim(m) * (3 + n_halflives)


def shuffle_block_order(tokens: np.ndarray, seed: int) -> np.ndarray:
    """Chronology-placebo control (ablation 2 / H3): a fixed random
    permutation of the block axis, same composition of blocks, scrambled
    order -- destroys any signal that depends on *when* a drift happened
    while preserving daily composition."""
    perm = np.random.default_rng(seed).permutation(tokens.shape[0])
    return tokens[perm]
