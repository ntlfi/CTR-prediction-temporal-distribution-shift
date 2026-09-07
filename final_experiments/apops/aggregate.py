"""Delayed fixed-share aggregation over the three memory experts
(plan section 2, "Delayed adaptive aggregation").

Given each expert's per-day probability vector (already row-aligned to the
day's input order), walk the whole stream one block at a time -- weights
persist across days -- and, for block ``r``:

  1. emit ``p^AP = sum_k w_k p^(k)`` using the *current* weights only;
  2. queue the block; no unmatured label may touch the weights;
  3. once every label in a queued block has matured, score each expert on
     its stored block predictions (pre-update losses) and update, with a
     **prior-centered** fixed share (implementation correction 2026-09-06):

         pi(lambda_AP) = (1 - lambda_AP, lambda_AP/2, lambda_AP/2)     # (R, S, L)
         w_1           = pi(lambda_AP)
         w_tilde_k     proportional to  w_k exp(-eta_m L_{r,k})
         w_{r+1}       = (1 - alpha) w_tilde_k + alpha * pi(lambda_AP)

The adaptive-mass parameter ``lambda_AP in [0, 1]`` is the single knob for
"how much of AP-OPS beyond OPS is used": ``lambda_AP = 0`` pins ``w`` to
``(1, 0, 0)`` forever, so AP-OPS reproduces OPS prediction by prediction.

Absolute time ``d * 86400 + sec`` is used for maturation so a block near a
day boundary is scored at the correct wall-clock moment (its labels
mature into the next day) rather than being dropped.

``fixed_weights`` forces a constant weight vector (bypasses the meta
update); ``(1, 0, 0)`` is equivalent to ``lambda_AP = 0``.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

SECONDS_PER_DAY = 86_400
EPS_LL = 1e-12


@dataclass
class MetaConfig:
    eta_m: float = 10.0               # meta learning rate on mean block log loss
    switch_half_life_h: float = 16.0  # fixed-share via alpha(tau) = 1 - 2 ** (-block_sec/tau)
    block_sec: int = 900
    delay_sec: int = 1800
    lambda_ap: float = 0.5            # adaptive mass; 0 => AP-OPS == OPS

    def alpha(self) -> float:
        tau = self.switch_half_life_h * 3600.0
        if not np.isfinite(tau):
            return 0.0
        return 1.0 - 2.0 ** (-self.block_sec / tau)

    def prior(self) -> np.ndarray:
        lam = float(np.clip(self.lambda_ap, 0.0, 1.0))
        return np.array([1.0 - lam, lam / 2.0, lam / 2.0])


def _block_logloss(y, p):
    p = np.clip(np.asarray(p, float), EPS_LL, 1 - EPS_LL)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def aggregate(expert_records: dict, bank: dict, days, cfg: MetaConfig,
              fixed_weights=None):
    """``expert_records`` maps expert name -> that expert's records list.
    The first three keys, in insertion order, are treated as (R, S, L).
    Returns ``(records, info)``."""
    names = list(expert_records)[:3]
    K = len(names)
    per_day = {name: {r["day"]: r for r in expert_records[name]} for name in names}
    days = sorted(int(d) for d in days if d in bank and all(d in per_day[n] for n in names))

    n_blocks = int(np.ceil(SECONDS_PER_DAY / cfg.block_sec))
    alpha = cfg.alpha()
    prior = cfg.prior()
    w = prior.copy()

    queue = deque()                       # (mature_abs, y_blk, [p_blk per expert])
    weight_path = []                      # one row per meta update
    dom_blocks = np.zeros(K, dtype=int)   # blocks emitted where expert k had max weight
    n_updates = 0
    min_w, max_w = 1.0, 0.0

    out = {}
    for d in days:
        y = np.asarray(bank[d].y, float)
        sec = np.asarray(bank[d].sec_in_day, float)
        n = len(y)
        p_ap = np.full(n, np.nan)
        blk = np.minimum((sec // cfg.block_sec).astype(int), n_blocks - 1)
        preds = {name: np.asarray(per_day[name][d]["p"], float) for name in names}

        for k in range(n_blocks):
            cur_start_abs = d * SECONDS_PER_DAY + k * cfg.block_sec
            # (2)->(3): fold in every block whose labels have all matured
            while queue and queue[0][0] <= cur_start_abs:
                _, y_blk, p_blks = queue.popleft()
                L = np.array([_block_logloss(y_blk, p_blks[j]) for j in range(K)])
                if fixed_weights is None:
                    w_tilde = w * np.exp(-cfg.eta_m * (L - L.min()))
                    s = w_tilde.sum()
                    w_tilde = w_tilde / s if s > 0 else prior.copy()
                    w = (1.0 - alpha) * w_tilde + alpha * prior
                    n_updates += 1
                    min_w, max_w = min(min_w, w.min()), max(max_w, w.max())
                    weight_path.append({"day": d, "block": k, **{names[j]: float(w[j]) for j in range(K)},
                                        "L_R": float(L[0]), "L_S": float(L[1]), "L_L": float(L[2])})

            in_blk = np.where(blk == k)[0]
            if len(in_blk) == 0:
                continue
            eff_w = np.asarray(fixed_weights, float) if fixed_weights is not None else w
            mix = np.zeros(len(in_blk))
            for j, name in enumerate(names):
                mix += eff_w[j] * preds[name][in_blk]
            p_ap[in_blk] = mix
            dom_blocks[int(np.argmax(eff_w))] += 1
            mature_abs = d * SECONDS_PER_DAY + (k + 1) * cfg.block_sec + cfg.delay_sec
            queue.append((mature_abs, y[in_blk].copy(),
                          [preds[name][in_blk].copy() for name in names]))

        miss = np.where(np.isnan(p_ap))[0]
        if len(miss):                    # blocks with no matured feedback yet -> current weights
            eff_w = np.asarray(fixed_weights, float) if fixed_weights is not None else w
            mix = np.zeros(len(miss))
            for j, name in enumerate(names):
                mix += eff_w[j] * preds[name][miss]
            p_ap[miss] = mix
        out[d] = p_ap

    records = [{"day": d, "y": bank[d].y, "p": out[d], "sec_in_day": bank[d].sec_in_day}
               for d in days]
    info = {
        "weight_path": weight_path,
        "final_weights": {names[j]: float(w[j]) for j in range(K)},
        "dominant_block_counts": {names[j]: int(dom_blocks[j]) for j in range(K)},
        "n_meta_updates": n_updates,
        "weight_min": float(min_w) if n_updates else float(w.min()),
        "weight_max": float(max_w) if n_updates else float(w.max()),
        "alpha": alpha, "lambda_ap": float(cfg.lambda_ap),
        "prior": {names[j]: float(prior[j]) for j in range(K)},
    }
    return records, info
