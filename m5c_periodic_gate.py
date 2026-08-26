"""M5c -- M5b (context-dependent gating over the full WINDOW_FAMILY, see
m5_multiscale_gate.py) plus explicit periodicity phase features, to test
whether the recurring-drift blind spot shared by every method in this
project (results/m5_analysis.md) is fixable with an explicit
"where in the cycle are we" signal rather than only "how much recent
history to trust".

Identical gate architecture and training rule to M5b -- same causal rule
(day t's prediction only ever uses a gate fit on days < t), same
regularization -- with 2 extra per-day features appended: sin/cos of day
t's phase within a period, from periodicity.py's phase_features. The
caller supplies `period_by_day: {day: period_or_None}` so this module
stays agnostic to how the period was obtained:

- **deployed**: `periodicity.causal_period_series(...)` over a per-day loss
  signal built only from days < t (see run_new_methods.py) -- realistic,
  works on real data too, no assumption a period even exists.
- **oracle** (diagnostic only, synthetic data only): a constant dict mapping
  every day to the generator's true `period_days` -- quantifies the upper
  bound on what a periodicity feature could buy if detection were perfect,
  the same role M1a's oracle alpha plays for M1.
"""
import numpy as np
import torch
import torch.nn as nn

from baselines import WINDOW_FAMILY
from han_arw import per_sample_log_loss
from m5_multiscale_gate import MultiExpertGate, _entropy, _recent_expert_losses, multi_days
from periodicity import phase_features

BASE_FEATURE_NAMES = ["abs_spread", "norm_time", "sin_phase", "cos_phase"]


def gate_feature_matrix(bank: dict, day_list: list, t: int, T: int, period_by_day: dict,
                         context: np.ndarray = None, day: np.ndarray = None):
    """Same per-example feature matrix as m5_multiscale_gate.gate_feature_matrix
    (per-expert predictions, spread, norm_time, recent per-expert loss, optional
    context), plus sin/cos phase features from `period_by_day[t]` -- (0.0, 0.0)
    if no period was available/detected for day t. None of these use day t's
    own true labels. Returns (features, preds (n, K))."""
    preds = np.stack([bank[name][t]["y_pred"] for name in WINDOW_FAMILY], axis=1)  # (n, K)
    n = preds.shape[0]
    spread = preds.max(axis=1) - preds.min(axis=1)
    norm_t = np.full(n, t / max(T, 1))
    recent_losses = np.tile(_recent_expert_losses(bank, day_list, t), (n, 1))  # (n, K)
    sin_p, cos_p = phase_features(t, period_by_day.get(t))
    feats = np.concatenate([preds, spread[:, None], norm_t[:, None], recent_losses,
                             np.full((n, 1), sin_p), np.full((n, 1), cos_p)], axis=1)
    if context is not None:
        feats = np.concatenate([feats, context[day == t]], axis=1)
    return feats, preds


def run_m5c(bank: dict, eligible_days, T: int, period_by_day: dict, lr: float = 0.05, l2: float = 1e-3,
            entropy_reg: float = 1e-3, smooth_reg: float = 1e-3, epochs_per_day: int = 3,
            seed: int = 0, context: np.ndarray = None, day: np.ndarray = None):
    """Returns rows: {day, y_true, y_pred, n_train, fit_time, weights,
    mean_weights, period}. Identical to m5_multiscale_gate.run_m5 except
    for the 2 extra phase features and the `period` diagnostic column
    (the period_by_day value actually used for day t, None if
    undetected/unavailable)."""
    torch.manual_seed(seed)
    days = multi_days(bank, eligible_days)
    if not days:
        return []

    n_experts = len(WINDOW_FAMILY)
    n_context = context.shape[1] if context is not None else 0
    n_features = len(WINDOW_FAMILY) + len(BASE_FEATURE_NAMES) + n_experts + n_context
    gate = MultiExpertGate(n_features, n_experts)
    opt = torch.optim.Adam(gate.parameters(), lr=lr)
    prev_mean_pi = torch.full((n_experts,), 1.0 / n_experts)

    rows = []
    for t in days:
        feats, preds = gate_feature_matrix(bank, days, t, T, period_by_day, context=context, day=day)
        y_true = bank[WINDOW_FAMILY[0]][t]["y_true"]
        feats_t = torch.tensor(feats, dtype=torch.float32)
        preds_t = torch.tensor(preds, dtype=torch.float32)
        y_t = torch.tensor(y_true, dtype=torch.float32)

        with torch.no_grad():
            pi = gate(feats_t).numpy()
        y_pred = (preds * pi).sum(axis=1)
        mean_pi = pi.mean(axis=0)

        rows.append({
            "day": t,
            "y_true": y_true,
            "y_pred": y_pred,
            "n_train": int(np.mean([bank[name][t]["n_train"] for name in WINDOW_FAMILY])),
            "fit_time": float(sum(bank[name][t]["fit_time"] for name in WINDOW_FAMILY)),
            "weights": pi,
            "mean_weights": dict(zip(WINDOW_FAMILY, mean_pi.tolist())),
            "period": period_by_day.get(t),
        })

        # Update the gate only now that day t's true labels are observed --
        # used for t+1 onward, never for day t itself.
        for _ in range(epochs_per_day):
            opt.zero_grad()
            pi_t = gate(feats_t)
            p_mix = (preds_t * pi_t).sum(dim=-1).clamp(1e-7, 1 - 1e-7)
            bce = -(y_t * p_mix.log() + (1 - y_t) * (1 - p_mix).log()).mean()
            l2_term = sum((p ** 2).sum() for p in gate.parameters())
            entropy_term = -_entropy(pi_t).mean()
            smooth_term = (pi_t.mean(dim=0) - prev_mean_pi).abs().sum()
            loss = bce + l2 * l2_term + entropy_reg * entropy_term + smooth_reg * smooth_term
            loss.backward()
            opt.step()
        with torch.no_grad():
            prev_mean_pi = gate(feats_t).mean(dim=0).detach()

    return rows
