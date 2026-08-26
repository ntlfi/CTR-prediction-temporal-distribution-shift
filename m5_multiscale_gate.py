"""M5b -- Context-Dependent Multi-Timescale Mixture of Temporal Experts
(adaptive-training-methods-implementation-plan.md section 10). Generalizes
M2's two-expert (short=rolling_3, long=expanding) gate to the full
WINDOW_FAMILY candidate bank (rolling_1/3/7/14 + expanding) already built
for han_arw.py/adamoe.py:

    p_hat_t(x) = sum_h pi_{t,h}(x) * p_t^(h)(x),   sum_h pi_{t,h}(x) = 1
    pi_t(x) = softmax(gate_phi(u_t(x)))

Built only after M2's two-expert gate was validated (plan section 19's
"implement M5 only after the two-expert gate is stable"): M2, once given
real per-example context (data.raw_numeric_features), beat every P0/P1/P2
baseline including han_arw under S3 (recurring, replicated across 5
seeds) and S4 (local) drift -- but stayed a notch behind han_arw under
S1/S2 (abrupt/gradual). The diagnosed reason: mixing only 2 of the 5
window candidates can't reach rolling_14, the actual best fixed window
there. This module removes that ceiling by gating over all 5 candidates
with the same online, context-carrying, regularized gate M2 uses --
same causal rule (day t's prediction only ever uses a gate fit on
days < t), same entropy regularization (here: toward the uniform
distribution over experts, the natural "no downside" default, achieved
by zero-initializing the gate so it starts as AdaMoE-style uniform
averaging), same day-level smoothness penalty.
"""
import numpy as np
import torch
import torch.nn as nn

from baselines import WINDOW_FAMILY
from han_arw import per_sample_log_loss

BASE_FEATURE_NAMES = ["abs_spread", "norm_time"]  # + one recent-loss feature per expert + per-expert p_h


def multi_days(bank: dict, eligible_days) -> list:
    """Prediction days for which every WINDOW_FAMILY candidate exists."""
    return sorted(t for t in eligible_days if all(t in bank[name] for name in WINDOW_FAMILY))


def _recent_expert_losses(bank: dict, day_list: list, t: int) -> np.ndarray:
    past = [d for d in day_list if d < t]
    if not past:
        return np.zeros(len(WINDOW_FAMILY))
    prev = past[-1]
    return np.array([per_sample_log_loss(bank[name][prev]["y_true"], bank[name][prev]["y_pred"]).mean()
                      for name in WINDOW_FAMILY])


def gate_feature_matrix(bank: dict, day_list: list, t: int, T: int,
                         context: np.ndarray = None, day: np.ndarray = None):
    """Per-example gate feature matrix for day t (plan section 10, M5b):
    [p_h for h in WINDOW_FAMILY, spread=max_h p_h - min_h p_h, norm_time,
    recent per-expert mean log loss] plus, if given, per-example context
    columns -- none of these use day t's own true labels. Returns
    (features, preds (n, K))."""
    preds = np.stack([bank[name][t]["y_pred"] for name in WINDOW_FAMILY], axis=1)  # (n, K)
    n = preds.shape[0]
    spread = preds.max(axis=1) - preds.min(axis=1)
    norm_t = np.full(n, t / max(T, 1))
    recent_losses = np.tile(_recent_expert_losses(bank, day_list, t), (n, 1))  # (n, K)
    feats = np.concatenate([preds, spread[:, None], norm_t[:, None], recent_losses], axis=1)
    if context is not None:
        feats = np.concatenate([feats, context[day == t]], axis=1)
    return feats, preds


class MultiExpertGate(nn.Module):
    def __init__(self, n_features: int, n_experts: int):
        super().__init__()
        self.linear = nn.Linear(n_features, n_experts)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.linear(feats), dim=-1)  # (n, K), uniform at init


def _entropy(pi: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    p = pi.clamp(eps, 1.0)
    return -(p * p.log()).sum(dim=-1)


def run_m5(bank: dict, eligible_days, T: int, lr: float = 0.05, l2: float = 1e-3,
           entropy_reg: float = 1e-3, smooth_reg: float = 1e-3, epochs_per_day: int = 3,
           seed: int = 0, context: np.ndarray = None, day: np.ndarray = None):
    """Returns rows: {day, y_true, y_pred, n_train, fit_time, weights,
    mean_weights (dict name -> mean pi across x)}. `y_pred` is the
    per-example softmax mixture actually used for day t's deployed
    prediction (gate trained on days < t only)."""
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
        feats, preds = gate_feature_matrix(bank, days, t, T, context=context, day=day)
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
        })

        # Update the gate only now that day t's true labels are observed --
        # used for t+1 onward, never for day t itself.
        for _ in range(epochs_per_day):
            opt.zero_grad()
            pi_t = gate(feats_t)
            p_mix = (preds_t * pi_t).sum(dim=-1).clamp(1e-7, 1 - 1e-7)
            bce = -(y_t * p_mix.log() + (1 - y_t) * (1 - p_mix).log()).mean()
            l2_term = sum((p ** 2).sum() for p in gate.parameters())
            entropy_term = -_entropy(pi_t).mean()  # minimized -> maximizes entropy -> pi toward uniform
            smooth_term = (pi_t.mean(dim=0) - prev_mean_pi).abs().sum()
            loss = bce + l2 * l2_term + entropy_reg * entropy_term + smooth_reg * smooth_term
            loss.backward()
            opt.step()
        with torch.no_grad():
            prev_mean_pi = gate(feats_t).mean(dim=0).detach()

    return rows
