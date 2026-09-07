"""Full-Platt calibration experts updated by discounted Online Newton Step
(plan section 2, "Delayed second-order expert update").

Chronological single pass over the whole ``days`` stream with an
**absolute-time feedback queue** (implementation correction, 2026-09-06):
a block's labels mature at ``d*86400 + (k+1)*block_sec + delay_sec`` in
absolute time, so a block near a day boundary is scored on the *following*
day rather than dropped. Each block ``r`` contributes exactly one delayed
ONS step, from its own impressions' mean gradient, once every one of its
labels has matured::

    gamma = 2 ** (-block_sec / half_life)                 # per block
    g_r   = mean_{i in block r} (p_i - y_i) * (z_i, 1)    # grad of mean block log loss
    A_r   = gamma A_{r-1} + g_r g_rᵀ + (1-gamma) lambda I
    theta_{r+1} = Proj_Theta( theta_r - eta_ons A_r⁻¹ g_r ),  Theta = a∈a_bounds, b∈b_bounds

``reset_daily=False`` (default) carries ``theta`` and ``A`` continuously
across days -- the *persistent* expert. ``reset_daily=True`` re-initialises
``theta=(1,0)``, ``A=lambda I`` at every day boundary (Reset-ONS): a
label that matured after midnight still updates that new day's fresh
state, so Reset-ONS vs the persistent expert is a clean single-axis
comparison (reset vs carry, identical maturation).

The projection is the coordinate-wise clip onto the box (same choice
``dualtime.online`` makes for its norm ball); every clip is counted.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

SECONDS_PER_DAY = 86_400


def _logit(p, eps):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


@dataclass
class OnsConfig:
    half_life_h: float = 4.0          # physical forgetting half-life (hours)
    lam: float = 1e-2                 # initial curvature / ridge (A_0 = lam I)
    eta_ons: float = 1.0              # ONS step scale
    block_sec: int = 900
    delay_sec: int = 1800
    eps: float = 1e-5
    a_bounds: tuple = (0.2, 5.0)      # plan section 2: a in [0.2, 5]
    b_bounds: tuple = (-0.25, 0.25)   # plan section 2: b in [-0.25, 0.25]
    learn_slope: bool = True          # False => no-slope ablation (a fixed at 1)
    reset_daily: bool = False         # False = persistent; True = Reset-ONS


def replay_ons_stream(q_by_day: dict, bank: dict, days, cfg: OnsConfig):
    """Replay one ONS expert over the whole chronological ``days`` stream.
    Returns ``(records, trace)`` -- ``records`` the familiar
    ``{"day","y","p","sec_in_day"}`` list (``p`` row-aligned to each day's
    input order), ``trace`` the per-day end state + projection count."""
    days = sorted(int(d) for d in days if d in q_by_day and d in bank)
    gamma = 2.0 ** (-cfg.block_sec / (cfg.half_life_h * 3600.0))
    n_blocks = int(np.ceil(SECONDS_PER_DAY / cfg.block_sec))
    lo_a, hi_a = cfg.a_bounds
    lo_b, hi_b = cfg.b_bounds
    lam_I = cfg.lam * np.eye(2)

    theta = np.array([1.0, 0.0])
    A = cfg.lam * np.eye(2)
    step = 0
    proj_events = 0

    pending = deque()          # (mature_abs, z_blk, y_blk)
    p_hat = {}
    trace = []

    def apply_block(z_blk, y_blk):
        nonlocal theta, A, step, proj_events
        step += 1
        p_m = _sigmoid(theta[0] * z_blk + theta[1])
        resid = p_m - y_blk
        if cfg.learn_slope:
            g = np.array([float((resid * z_blk).mean()), float(resid.mean())])
        else:
            g = np.array([0.0, float(resid.mean())])
        A = gamma * A + np.outer(g, g) + (1.0 - gamma) * lam_I
        try:
            A_inv = np.linalg.inv(A + 1e-12 * np.eye(2))
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A)
        theta_new = theta - cfg.eta_ons * (A_inv @ g)
        clipped = np.array([
            min(max(theta_new[0], lo_a), hi_a) if cfg.learn_slope else 1.0,
            min(max(theta_new[1], lo_b), hi_b),
        ])
        if not np.array_equal(clipped, theta_new):
            proj_events += 1
        theta = clipped

    for di, d in enumerate(days):
        q = np.asarray(q_by_day[d], float)
        y = np.asarray(bank[d].y, float)
        sec = np.asarray(bank[d].sec_in_day, float)
        n = len(q)
        ph = np.full(n, np.nan)
        if n == 0:
            p_hat[d] = ph
            continue
        z = _logit(q, cfg.eps)
        blk = np.minimum((sec // cfg.block_sec).astype(int), n_blocks - 1)
        day_proj0 = proj_events

        for k in range(n_blocks):
            block_start_abs = d * SECONDS_PER_DAY + k * cfg.block_sec
            while pending and pending[0][0] <= block_start_abs:
                _, z_blk, y_blk = pending.popleft()
                apply_block(z_blk, y_blk)
            if cfg.reset_daily and di != 0 and k == 0:
                theta = np.array([1.0, 0.0])
                A = cfg.lam * np.eye(2)
                step = 0
            in_blk = np.where(blk == k)[0]
            if len(in_blk) == 0:
                continue
            ph[in_blk] = _sigmoid(theta[0] * z[in_blk] + theta[1])
            mature_abs = d * SECONDS_PER_DAY + (k + 1) * cfg.block_sec + cfg.delay_sec
            pending.append((mature_abs, z[in_blk].copy(), y[in_blk].copy()))

        miss = np.where(np.isnan(ph))[0]
        if len(miss):
            ph[miss] = _sigmoid(theta[0] * z[miss] + theta[1])
        p_hat[d] = ph
        trace.append({"day": d, "a_end": float(theta[0]), "b_end": float(theta[1]),
                      "proj_events_day": proj_events - day_proj0, "cum_steps": step,
                      "cond_A": float(np.linalg.cond(A))})

    # drain the tail for trace completeness (does not touch any emitted pred)
    while pending:
        _, z_blk, y_blk = pending.popleft()
        apply_block(z_blk, y_blk)

    records = [{"day": d, "y": bank[d].y, "p": p_hat[d], "sec_in_day": bank[d].sec_in_day}
               for d in days]
    return records, trace
