"""DualTime-CTR's within-day module: a projected online-gradient learner
over the frozen phi features (protocol section 11), reset to ``w = 0``
every day. Structurally this generalizes ``twoscale.calib.replay_day``
(same block-boundary update cadence, same causal maturation logic) from a
scalar intercept/slope to a vector ``w`` over a richer feature ``phi`` --
the "frozen-encoder online-regret" variant the original within-day plan
sketched (eq 16-17) but never built, now implemented for real using the
plan's own frozen feature construction (``withinday.blocks``,
``withinday.contextsketch``).

IMPORTANT distinction from the capacity-ladder V5 adapter
(``withinday.adapters.V5Linear``, ``withinday_experiments/``): V5's ``w``
is trained OFFLINE on historical days by HPO/logistic regression, then
FROZEN for the whole test period -- only the history features ``h`` (via
``phi(x, h)``) change causally within a day, not the weights. DualTime-CTR
as defined in the final-experiment plan is the online version:
``w`` itself updates within the day,
``w_{d,i+1} = Pi_W(w_{d,i} - eta_i * grad(l_i(w_{d,i})))`` -- exactly
what ``replay_day`` below does (``w`` starts at 0 each day and is updated
by projected gradient descent at every matured block, ``eta_k =
B_w/sqrt(k)``, discretized at block cadence rather than continuously per
impression). V5's offline-frozen result is the experiment that motivated
this module's ``phi`` architecture (the hashed context x history bilinear
feature), not an implementation of DualTime-CTR itself -- the two must
never be conflated when writing up results.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from withinday.blocks import (block_of, build_block_tokens, deterministic_summary,
                              last_available_block, n_blocks_per_day, summary_dim, token_dim)
from withinday.contextsketch import context_sketch

SECONDS_PER_DAY = 86_400


def _logit(p, eps):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def build_hash_projection(a_dim: int, s_dim: int, cross_dim: int = 32, seed: int = 0):
    """Fixed (never updated) signed projections for the hashed context x
    history interaction ``H(a (x) s)`` -- the same compact-bilinear-pooling
    trick as ``withinday.adapters.V5Linear``, in plain numpy since
    DualTime-CTR's update is closed-form projected gradient descent, not
    autograd."""
    rng = np.random.default_rng(seed)
    Ra = (rng.integers(0, 2, size=(a_dim, cross_dim)).astype(float) * 2 - 1)
    Rs = (rng.integers(0, 2, size=(s_dim, cross_dim)).astype(float) * 2 - 1)
    return Ra, Rs


def build_phi(a: np.ndarray, s: np.ndarray, Ra: np.ndarray, Rs: np.ndarray) -> np.ndarray:
    """``phi = [1, a, s, H(a (x) s)]``, then norm-bounded to ``||phi||_2
    <= 1`` (protocol section 11) -- matches the bounded-feature assumption
    the online-regret bound needs."""
    cross = (a @ Ra) * (s @ Rs)
    n = a.shape[0]
    phi = np.concatenate([np.ones((n, 1)), a, s, cross], axis=1)
    norm = np.linalg.norm(phi, axis=1, keepdims=True)
    return phi / np.maximum(1.0, norm)


def phi_dim(a_dim: int, s_dim: int, cross_dim: int = 32) -> int:
    return 1 + a_dim + s_dim + cross_dim


@dataclass
class DualTimeConfig:
    block_sec: int = 900
    delay_sec: int = 1800
    m: int = 32
    cross_dim: int = 32
    B_w: float = 1.0
    eps: float = 1e-5


def replay_day(q: np.ndarray, y: np.ndarray, sec_in_day: np.ndarray, X_day, R_sketch, Ra, Rs,
               cfg: DualTimeConfig):
    """Replay one day of DualTime-CTR (protocol section 11): ``w`` resets
    to 0 at the start of the day (so ``p_hat == q`` until the first block
    matures -- the required "no-history identity"), updated once per
    matured block using that block's own mean gradient, projected onto
    ``||w||_2 <= B_w``, with ``eta_k = B_w / sqrt(k)``.

    Returns ``p_hat`` (aligned to the *input* row order) and a trace of
    ``w`` norms per update step (for sanity-checking the projection)."""
    q = np.clip(np.asarray(q, float), cfg.eps, 1 - cfg.eps)
    y = np.asarray(y, float)
    sec = np.asarray(sec_in_day, float)
    n = len(q)
    if n == 0:
        return {"p_hat": np.array([]), "w_end": None, "trace": []}

    csketch = context_sketch(X_day, m=cfg.m, R=R_sketch)
    tokens = build_block_tokens(q, y, sec, csketch, cfg.block_sec, eps=cfg.eps)
    summary = deterministic_summary(tokens)
    k_avail = last_available_block(sec, cfg.block_sec, cfg.delay_sec)
    time_of_day = sec / SECONDS_PER_DAY
    a = np.concatenate([csketch, _logit(q, cfg.eps)[:, None], time_of_day[:, None]], axis=1)

    nb, s_dim = summary.shape
    pad_summary = np.vstack([summary, np.zeros((1, s_dim))])
    s = pad_summary[np.where(k_avail < 0, nb, k_avail)]
    phi = build_phi(a, s, Ra, Rs)

    order = np.argsort(sec, kind="stable")
    inv = np.empty(n, dtype=int)
    inv[order] = np.arange(n)
    z = _logit(q[order], cfg.eps)
    phi_s = phi[order]
    ys = y[order]
    ts = sec[order]

    mature = ts + cfg.delay_sec
    mat_order = np.argsort(mature, kind="stable")
    mature_sorted = mature[mat_order]

    w = np.zeros(phi.shape[1])
    p_hat = np.full(n, np.nan)
    trace = []
    step = 0
    mp = 0

    n_blocks = n_blocks_per_day(cfg.block_sec)
    blk = np.minimum((ts // cfg.block_sec).astype(int), n_blocks - 1)
    for k in range(n_blocks):
        in_blk = np.where(blk == k)[0]
        if len(in_blk):
            p_hat[in_blk] = _sigmoid(z[in_blk] + phi_s[in_blk] @ w)
        block_end = (k + 1) * cfg.block_sec
        hi = int(np.searchsorted(mature_sorted, block_end, side="right"))
        newly = mat_order[mp:hi]
        mp = hi
        if len(newly):
            step += 1
            p_m = _sigmoid(z[newly] + phi_s[newly] @ w)
            resid = p_m - ys[newly]
            grad = (phi_s[newly] * resid[:, None]).mean(axis=0)
            eta = cfg.B_w / np.sqrt(step)
            w = w - eta * grad
            norm = np.linalg.norm(w)
            if norm > cfg.B_w:
                w = w * (cfg.B_w / norm)
            trace.append({"block": k, "step": step, "n_matured": int(len(newly)), "w_norm": float(np.linalg.norm(w))})

    miss = np.where(np.isnan(p_hat))[0]
    if len(miss):
        p_hat[miss] = _sigmoid(z[miss] + phi_s[miss] @ w)

    return {"p_hat": p_hat[inv], "w_end": w, "trace": trace}
