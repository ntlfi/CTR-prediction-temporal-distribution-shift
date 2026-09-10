"""Causal AMG-TP on the shared five-horizon bank (plan section 4).

    q_t(x)     = softmax( g_phi( feats_t(x), s_{t-1} ) )        sample-specific context gate
    beta_t     = sigmoid( r_psi( s_{t-1} ) ) in [0, 1]          global learned persistence (per period)
    pi_t(x)    = (1 - beta_t) q_t(x) + beta_t * m_{t-1}         deployed temporal mixture
    p_hat_t(x) = sum_k pi_{t,k}(x) p^(k)_t(x)
    m_t        = (1 - rho) m_{t-1} + rho * mean_x pi_t(x)       memory EMA of the DEPLOYED weights

Everything is causal at the period (day) level: day ``t``'s deployed
prediction uses ``q``, ``beta`` and ``m`` carried from days ``< t`` only.
The gate ``phi`` and persistence net ``psi`` are updated once per period,
*after* day ``t``'s labels are observed -- and only on labels that have
**matured** (plan section 3): a day-``t`` impression at ``sec_in_day = s``
matures at ``s + delay_sec``; if that is after midnight the impression is
held out of the period-``t`` update and folded into period ``t+1``'s
update instead (its feature row is preserved). The memory update uses the
weights **actually deployed** for day ``t`` (computed before the gate
step) -- the past weights are never recomputed after ``phi`` is retrained
(plan section 4).

The gate softmax and ``m`` both live on all five nominal horizon slots.
When two nominal horizons coincide (truncated windows, plan section 4)
their prediction columns are identical, so any split of weight between the
two aliased slots yields exactly the same ``sum_k pi_k p_k`` -- duplicates
are handled with no double counting and no special case.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from m5_multiscale_gate import MultiExpertGate, _entropy
from amgtp_method import PersistenceNet

from .bank import HORIZONS5
from .maturity import available_mask

SECONDS_PER_DAY = 86_400
CONTEXT_M = 32           # dimension of the cached per-example context sketch
K = len(HORIZONS5)
_STATE_NAMES = (["recent_loss_" + h for h in HORIZONS5]
                + ["recent_disagreement", "recent_ctr", "recent_gate_move",
                   "norm_time", "loss_jump", "q_vs_m_div"])
N_STATE = len(_STATE_NAMES)


@dataclass
class AMGTPConfig:
    lr: float = 0.05
    l2: float = 1e-3
    entropy_reg: float = 1e-3
    rho: float = 0.3                 # memory EMA rate
    init_bias: float = -1.0          # beta_0 = sigmoid(init_bias)
    epochs_per_day: int = 3
    persist_hidden: int = 0          # 0 => linear persistence net (Stage 2 architecture)
    adaptive_beta: bool = True
    fixed_beta: float = 0.0          # used iff adaptive_beta is False
    context_m: int = CONTEXT_M
    delay_sec: int = 1800
    eps: float = 1e-7


def _perday_logloss(y, p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _gate_features(db, C_day, norm_t, recent_loss):
    """Per-example gate feature matrix for one day: [p_h (K), spread,
    norm_time, recent per-expert loss (K), context (m)]."""
    preds = np.stack([np.asarray(db.preds[h], float) for h in HORIZONS5], axis=1)
    n = preds.shape[0]
    spread = preds.max(axis=1) - preds.min(axis=1)
    nt = np.full(n, norm_t)
    rl = np.tile(recent_loss, (n, 1))
    feats = np.concatenate([preds, spread[:, None], nt[:, None], rl, C_day], axis=1)
    return feats.astype(np.float32), preds


def _state_vector(bank, days, t, T, prev_gate_move, prev_mean_q, m_prev, delay_sec):
    """r_psi input s_{t-1}. Every summary is over labels available by the
    start of day ``t`` -- the last past day's tail that has not matured by
    then is excluded (plan: the 30-min delay applies to loss / CTR
    summaries too, not only the online calibrator queues)."""
    past = [d for d in days if d < t and d in bank]
    norm_t = t / max(T, 1)
    if not past:
        return np.zeros(N_STATE, dtype=np.float32)
    prev = past[-1]

    def _avail(d):
        return available_mask(np.full(len(bank[d].sec_in_day), d, np.int64),
                              bank[d].sec_in_day, t, delay_sec)

    a_prev = _avail(prev)
    yp = np.asarray(bank[prev].y, float)[a_prev]
    recent_loss = np.array([_perday_logloss(yp, np.asarray(bank[prev].preds[h], float)[a_prev])
                            for h in HORIZONS5]) if a_prev.any() else np.zeros(K)
    if len(past) >= 2:
        a2 = _avail(past[-2])
        y2 = np.asarray(bank[past[-2]].y, float)[a2]
        pl2 = np.array([_perday_logloss(y2, np.asarray(bank[past[-2]].preds[h], float)[a2])
                        for h in HORIZONS5]) if a2.any() else np.zeros(K)
        loss_jump = float(np.abs(recent_loss - pl2).max())
    else:
        loss_jump = 0.0
    p_short = np.asarray(bank[prev].preds["roll3"], float)[a_prev]
    p_long = np.asarray(bank[prev].preds["expanding"], float)[a_prev]
    disagreement = float(np.abs(p_short - p_long).mean()) if a_prev.any() else 0.0
    recent_ctr = float(yp.mean()) if a_prev.any() else 0.0
    q_vs_m = float(np.abs(np.asarray(prev_mean_q) - np.asarray(m_prev)).sum()) if prev_mean_q is not None else 0.0
    return np.concatenate([recent_loss,
                           [disagreement, recent_ctr, prev_gate_move, norm_t,
                            loss_jump, q_vs_m]]).astype(np.float32)


def run_amgtp5(bank: dict, context_by_day: dict, days, cfg: AMGTPConfig, seed: int = 0):
    """Replay AMG-TP over the chronological ``days`` prefix. Returns
    ``(q_by_day, trace)`` where ``q_by_day[d]`` is the deployed AMG-TP
    mixture prediction for day ``d`` (row-aligned to that day's arrival
    order, i.e. ``bank[d]`` order), and ``trace`` is a per-day list of
    ``{day, beta, mean_pi (dict), m_state (dict)}``.

    ``context_by_day[d]`` is the fixed signed-hash context sketch for day
    ``d`` (shape ``(n_d, context_m)``, L2-normalised), precomputed and
    cached from a seeded random projection of the hashed feature matrix
    (``final_experiments/ttam/build_banks.py``)."""
    torch.manual_seed(seed)
    days = sorted(int(d) for d in days if d in bank and d in context_by_day)
    if not days:
        return {}, []
    T = max(days)

    n_gate_features = K + 2 + K + cfg.context_m
    gate = MultiExpertGate(n_gate_features, K)
    persist = PersistenceNet(N_STATE, init_bias=cfg.init_bias, hidden=cfg.persist_hidden)
    params = list(gate.parameters())
    if cfg.adaptive_beta:
        params += list(persist.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr) if params else None

    m_state = torch.full((K,), 1.0 / K)
    prev_gate_move = 0.0
    prev_mean_pi = None
    prev_mean_q = None
    prev_day = None
    carry = None                       # (feats, y) of the previous day's not-yet-matured tail

    q_by_day, trace = {}, []
    for t in days:
        db = bank[t]
        y_np = np.asarray(db.y, float)
        sec = np.asarray(db.sec_in_day, float)
        C_day = np.asarray(context_by_day[t], dtype=np.float64)
        s_prev = torch.tensor(
            _state_vector(bank, days, t, T, prev_gate_move, prev_mean_q, m_state.numpy(),
                          cfg.delay_sec),
            dtype=torch.float32)
        recent_loss = s_prev.numpy()[:K]
        feats_np, preds_np = _gate_features(db, C_day, t / max(T, 1), recent_loss)
        feats_t = torch.tensor(feats_np)

        # ---- deploy day t: q, beta, m all carried from days < t ----------
        with torch.no_grad():
            q = gate(feats_t)                                  # (n, K)
            if cfg.adaptive_beta:
                beta = torch.sigmoid(persist.logit(s_prev))
            else:
                beta = torch.tensor(float(cfg.fixed_beta))
            pi = (1.0 - beta) * q + beta * m_state.unsqueeze(0)
            pi_np = pi.numpy()
            q_np = q.numpy()
            beta_val = float(beta)
        q_by_day[t] = (preds_np * pi_np).sum(axis=1)
        mean_pi = pi_np.mean(axis=0)
        mean_q = q_np.mean(axis=0)
        gate_move = float(np.abs(mean_pi - prev_mean_pi).sum()) if prev_mean_pi is not None else 0.0
        trace.append({"day": t, "beta": beta_val,
                      "mean_pi": dict(zip(HORIZONS5, mean_pi.tolist())),
                      "m_state": dict(zip(HORIZONS5, m_state.tolist()))})

        # ---- memory update: EMA of the DEPLOYED weights (pre gate step) --
        m_state = (1.0 - cfg.rho) * m_state + cfg.rho * torch.tensor(mean_pi, dtype=torch.float32)

        # ---- matured-label set for the period-t update (plan section 3) --
        matured = sec <= (SECONDS_PER_DAY - cfg.delay_sec)
        upd_feats = [feats_np[matured]]
        upd_y = [y_np[matured]]
        if carry is not None and prev_day == t - 1:
            upd_feats.append(carry[0])
            upd_y.append(carry[1])
        upd_feats = np.concatenate(upd_feats, axis=0) if upd_feats else np.zeros((0, n_gate_features), np.float32)
        upd_y = np.concatenate(upd_y, axis=0) if upd_y else np.zeros(0)
        carry = (feats_np[~matured].copy(), y_np[~matured].copy()) if (~matured).any() else None
        prev_day = t

        # ---- gate / persistence update (period-level, matured labels) ----
        if opt is not None and len(upd_y) > 0:
            xf = torch.tensor(upd_feats)
            yf = torch.tensor(upd_y, dtype=torch.float32)
            preds_upd = torch.tensor(xf[:, :K].numpy())          # p_h columns are the first K features
            for _ in range(cfg.epochs_per_day):
                opt.zero_grad()
                q_tr = gate(xf)
                if cfg.adaptive_beta:
                    beta_tr = torch.sigmoid(persist.logit(s_prev))
                else:
                    beta_tr = torch.tensor(float(cfg.fixed_beta))
                pi_tr = (1.0 - beta_tr) * q_tr + beta_tr * m_state.detach().unsqueeze(0)
                p_mix = (preds_upd * pi_tr).sum(dim=-1).clamp(cfg.eps, 1 - cfg.eps)
                bce = -(yf * p_mix.log() + (1 - yf) * (1 - p_mix).log()).mean()
                l2 = sum((p ** 2).sum() for p in params)
                loss = bce + cfg.l2 * l2 + cfg.entropy_reg * (-_entropy(q_tr).mean())
                loss.backward()
                opt.step()

        prev_gate_move = gate_move
        prev_mean_pi = mean_pi
        prev_mean_q = mean_q

    return q_by_day, trace
