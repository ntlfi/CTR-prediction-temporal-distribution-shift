"""Within-day online calibration (plan section 2.2) and its causal replay.

The long-term prediction ``q`` is frozen for the day. A scalar state ``b``
(optionally with a Platt slope ``a``) corrects it on the logit scale:

    p_hat = sigmoid(a * logit(clip(q, eps)) + b),         a == 1 for intercept-only

    b <- Proj_[-B,B](b - eta_k * g_k),   g_k = mean_{newly matured i}(p_hat_i - y_i)

Causality is enforced two ways:

* a label from an impression at within-day time ``tau`` only enters ``g`` at
  ``tau + delay`` (feedback maturation, plan section 3, ``Delta``);
* the update producing the ``b`` used for a block/impression has seen only
  labels that matured strictly earlier.

``update="block"``  -> one aggregate step per time block (eq 6);
``update="impression"`` -> one step per matured label (eq 5).
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
class CalibConfig:
    B: float = 1.0                 # projection radius for b
    eta0: float = 0.1              # base learning rate
    eta_schedule: str = "inv_sqrt" # "inv_sqrt" (eta0/sqrt(k)) or "const"
    update: str = "block"          # "block" | "impression"
    block_sec: int = 1800          # 30-minute update blocks
    delay_sec: int = 1800          # feedback maturation delay Delta
    eps: float = 1e-5              # logit clip
    platt: bool = False            # also learn a slope a
    a_bounds: tuple = (0.2, 5.0)
    init_b: float = 0.0            # b_{d,1}; carry-over overrides this per day
    carryover_rho: float = 0.0     # b_{d,1} = rho * b_{d-1,end} (0 => daily reset)


def replay_day(q: np.ndarray, y: np.ndarray, sec_in_day: np.ndarray,
               cfg: CalibConfig, init_b: float | None = None,
               shuffle_seed: int | None = None):
    """Replay one day. Returns ``p_hat`` (aligned to the *input* row order),
    ``b_end``, ``a_end`` and a per-step trace."""
    q = np.asarray(q, float)
    y = np.asarray(y, float)
    sec = np.asarray(sec_in_day, float)
    n = len(q)
    if n == 0:
        return {"p_hat": np.array([]), "b_end": 0.0, "a_end": 1.0, "trace": []}

    order = np.argsort(sec, kind="stable")
    inv = np.empty(n, dtype=int)
    inv[order] = np.arange(n)

    z = _logit(q[order], cfg.eps)                # frozen logit of the long-term pred
    ys = y[order]
    ts = sec[order]
    if shuffle_seed is not None:
        # chronology placebo (section 9.5): keep every impression's arrival time,
        # block membership and maturation schedule, but scramble which (q, y)
        # pair occupies each slot -- so the calibrator can no longer exploit
        # within-day ordering. Position t now carries time-order index perm[t]
        # (input index order[perm[t]]).
        perm = np.random.default_rng(shuffle_seed).permutation(n)
        z, ys = z[perm], ys[perm]
        inv = np.argsort(order[perm])
    mature = ts + cfg.delay_sec
    mat_order = np.argsort(mature, kind="stable")   # labels in maturation order
    mature_sorted = mature[mat_order]

    b = float(np.clip(cfg.init_b if init_b is None else init_b, -cfg.B, cfg.B))
    a = 1.0
    p_hat = np.full(n, np.nan)
    trace = []
    step = 0
    mp = 0                                        # pointer into mat_order

    def eta():
        return cfg.eta0 / np.sqrt(step) if cfg.eta_schedule == "inv_sqrt" else cfg.eta0

    def apply_update(idx):
        nonlocal b, a, step
        if len(idx) == 0:
            return
        step += 1
        p_m = _sigmoid(a * z[idx] + b)
        resid = p_m - ys[idx]
        b = float(np.clip(b - eta() * float(resid.mean()), -cfg.B, cfg.B))
        if cfg.platt:
            a = float(np.clip(a - eta() * float((resid * z[idx]).mean()), *cfg.a_bounds))

    if cfg.update == "block":
        n_blocks = int(np.ceil(SECONDS_PER_DAY / cfg.block_sec))
        blk = np.minimum((ts // cfg.block_sec).astype(int), n_blocks - 1)
        for k in range(n_blocks):
            in_blk = np.where(blk == k)[0]
            if len(in_blk) == 0:
                continue
            p_hat[in_blk] = _sigmoid(a * z[in_blk] + b)     # predict with current state
            block_end = (k + 1) * cfg.block_sec
            hi = int(np.searchsorted(mature_sorted, block_end, side="right"))
            newly = mat_order[mp:hi]
            mp = hi
            apply_update(newly)
            trace.append({"pos": k, "b": b, "a": a,
                          "n_block": int(len(in_blk)), "n_matured": int(len(newly))})
    else:  # per-impression (eq 5)
        for i in range(n):
            while mp < n and mature[mat_order[mp]] <= ts[i]:
                apply_update(np.array([mat_order[mp]], dtype=int))
                mp += 1
            p_hat[i] = _sigmoid(a * z[i] + b)
        stride = max(1, n // 48)
        for k in range(0, n, stride):
            trace.append({"pos": int(k), "b": b, "a": a})

    miss = np.where(np.isnan(p_hat))[0]
    if len(miss):
        p_hat[miss] = _sigmoid(a * z[miss] + b)
    return {"p_hat": p_hat[inv], "b_end": b, "a_end": a, "trace": trace}


# --------------------------------------------------------------------------- #
#  non-deployable / seasonality references                                     #
# --------------------------------------------------------------------------- #
def oracle_intercept(q: np.ndarray, y: np.ndarray, B: float = 1.0, eps: float = 1e-5):
    """b*_d = argmin_b sum_i logloss(sigmoid(logit(q)+b), y), the best fixed
    within-day intercept in hindsight (plan section 5.1 / eq 10). 1-D convex;
    solved by Newton on [-B, B]. Returns (b_star, loss_at_b_star)."""
    z = _logit(np.asarray(q, float), eps)
    y = np.asarray(y, float)

    def loss(b):
        p = np.clip(_sigmoid(z + b), 1e-12, 1 - 1e-12)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()

    b = 0.0
    for _ in range(50):
        p = _sigmoid(z + b)
        g = float((p - y).mean())
        h = float((p * (1 - p)).mean()) + 1e-12
        b_new = float(np.clip(b - g / h, -B, B))
        if abs(b_new - b) < 1e-9:
            b = b_new
            break
        b = b_new
    return b, loss(b)


def time_of_day_intercepts(hist_q, hist_y, hist_slot, n_slots: int, B: float = 1.0,
                           eps: float = 1e-5):
    """One fixed intercept per within-day slot, fitted on prior-day data only
    (plan section 4 'Time-of-day' baseline). ``hist_slot`` is the slot index
    (e.g. hour) for each historical impression. Returns an array of length
    ``n_slots``."""
    hist_q = np.asarray(hist_q, float)
    hist_y = np.asarray(hist_y, float)
    hist_slot = np.asarray(hist_slot, int)
    out = np.zeros(n_slots)
    for s in range(n_slots):
        m = hist_slot == s
        if m.sum() < 50 or len(np.unique(hist_y[m])) < 2:
            continue
        out[s], _ = oracle_intercept(hist_q[m], hist_y[m], B=B, eps=eps)
    return out
