"""Persistent full-Platt calibration experts updated by discounted Online
Newton Step (plan section 2, "Delayed second-order expert update").

State ``theta = (a, b)`` and the 2x2 curvature ``A`` are carried
*continuously across days* -- unlike the reset anchor R (which is just
``twoscale.calib.replay_day`` with a daily reset and is built elsewhere
via ``methods.ops_method``).  Each persistent expert forgets old
curvature at a fixed physical rate::

    gamma = 2 ** (-block_sec / half_life)                 # per block
    g_r   = grad_theta L_r(theta_r)                       # mean block log-loss grad
    A_r   = gamma A_{r-1} + g_r g_r^T + (1 - gamma) lambda I
    theta_{r+1} = Proj_Theta( theta_r - eta_ons A_r^{-1} g_r )

with ``L_r`` the mean log loss over block ``r``'s impressions, computed
only once every label in the block has matured under the feedback delay.
The projection ``Proj_Theta`` is the cheap coordinate-wise clip onto the
box ``a in a_bounds``, ``b in b_bounds`` (same choice ``dualtime.online``
makes for its norm ball -- documented deviation from the exact
``A``-metric projection; every clip is counted in ``proj_events``).

Causality mirrors ``twoscale.calib.replay_day`` exactly: block ``k`` is
predicted with the state produced by strictly-earlier matured feedback,
and a label at within-day time ``tau`` only enters a gradient at
``tau + delay_sec``.
"""
from __future__ import annotations

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


def replay_ons_stream(q_by_day: dict, bank: dict, days, cfg: OnsConfig):
    """Replay one persistent ONS expert over the whole chronological
    ``days`` stream.  Returns ``(records, trace)`` where ``records`` is the
    familiar ``{"day","y","p","sec_in_day"}`` list (``p`` row-aligned to
    each day's input order) and ``trace`` carries the per-day end state and
    projection-event count."""
    days = sorted(int(d) for d in days if d in q_by_day and d in bank)
    gamma = 2.0 ** (-cfg.block_sec / (cfg.half_life_h * 3600.0))
    n_blocks = int(np.ceil(SECONDS_PER_DAY / cfg.block_sec))
    lo_a, hi_a = cfg.a_bounds
    lo_b, hi_b = cfg.b_bounds

    theta = np.array([1.0, 0.0])                 # (a, b)
    A = cfg.lam * np.eye(2)
    step = 0
    proj_events = 0

    records, trace = [], []
    for d in days:
        q = np.asarray(q_by_day[d], float)
        y = np.asarray(bank[d].y, float)
        sec = np.asarray(bank[d].sec_in_day, float)
        n = len(q)
        if n == 0:
            records.append({"day": d, "y": bank[d].y, "p": np.array([]),
                            "sec_in_day": bank[d].sec_in_day})
            continue

        order = np.argsort(sec, kind="stable")
        inv = np.empty(n, dtype=int)
        inv[order] = np.arange(n)
        z = _logit(q[order], cfg.eps)
        ys = y[order]
        ts = sec[order]

        mature = ts + cfg.delay_sec
        mat_order = np.argsort(mature, kind="stable")
        mature_sorted = mature[mat_order]

        blk = np.minimum((ts // cfg.block_sec).astype(int), n_blocks - 1)
        p_hat = np.full(n, np.nan)
        mp = 0
        day_proj = 0
        for k in range(n_blocks):
            in_blk = np.where(blk == k)[0]
            if len(in_blk):
                p_hat[in_blk] = _sigmoid(theta[0] * z[in_blk] + theta[1])
            block_end = (k + 1) * cfg.block_sec
            hi = int(np.searchsorted(mature_sorted, block_end, side="right"))
            newly = mat_order[mp:hi]
            mp = hi
            if len(newly):
                step += 1
                p_m = _sigmoid(theta[0] * z[newly] + theta[1])
                resid = p_m - ys[newly]
                if cfg.learn_slope:
                    g = np.array([float((resid * z[newly]).mean()), float(resid.mean())])
                else:
                    g = np.array([0.0, float(resid.mean())])
                A = gamma * A + np.outer(g, g) + (1.0 - gamma) * cfg.lam * np.eye(2)
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
                    day_proj += 1
                theta = clipped

        miss = np.where(np.isnan(p_hat))[0]
        if len(miss):
            p_hat[miss] = _sigmoid(theta[0] * z[miss] + theta[1])

        records.append({"day": d, "y": bank[d].y, "p": p_hat[inv],
                        "sec_in_day": bank[d].sec_in_day})
        trace.append({"day": d, "a_end": float(theta[0]), "b_end": float(theta[1]),
                      "proj_events_day": day_proj, "cum_steps": step,
                      "cond_A": float(np.linalg.cond(A))})
    return records, trace
