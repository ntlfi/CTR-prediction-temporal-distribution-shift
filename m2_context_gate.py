"""M2 -- Context-Dependent Short/Long Gating (adaptive-training-methods-
implementation-plan.md section 7, the plan's primary candidate method).
Per-example alpha_t(x) mixes the short-memory (rolling_3) and long-memory
(expanding) candidate-bank predictions:

    p_hat_t(x) = (1-alpha_t(x)) p_L(x) + alpha_t(x) p_S(x)
    alpha_t(x) = sigmoid(gate_phi(u_t(x)))

Implements M2a (direct mixture-loss training, the plan's preferred
variant): the gate is a small logistic model over the compact feature set
in short_long.gate_feature_matrix (p_S, p_L, disagreement, signed
disagreement, normalized time, recent short/long loss, plus an optional
per-example context block -- see the `context` argument below), trained
online -- after each day t's true labels are observed, a few gradient
steps update the gate on that day's just-matured (features, label) pairs
before moving to t+1. This keeps the causal rule intact (day t's
prediction only ever uses a gate fit on days < t, same as AdaMoE's EMA
update) while staying O(T) instead of refitting on all pooled history
every single day.

The base 7 features alone carry no direct subpopulation signal -- an
earlier version of this file used only them and, on the S4 local-drift
synthetic benchmark, learned an almost-identical gate to M1's single
global alpha (no meaningful edge from being "context-dependent" at all).
`context` (typically data.raw_numeric_features) is what actually lets the
gate condition on which subpopulation an example belongs to.

Regularized per plan section 7: L2 weight decay (zero-initialized weights
so alpha(x)=0.5 everywhere before any training, i.e. no downside out of
the box) plus an entropy penalty that shrinks alpha(x) toward 0.5, plus a
day-level smoothness penalty on mean(alpha) between consecutive days (a
day-level approximation of the plan's per-example |alpha_t(x)-alpha_{t-1}
(x)| term, since example identity isn't consistent across days here).
"""
import numpy as np
import torch
import torch.nn as nn

from short_long import BASE_FEATURE_NAMES, LONG_NAME, SHORT_NAME, gate_feature_matrix, mix, short_long_days


class ContextGate(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(feats)).squeeze(-1)


def _entropy(alpha: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    a = alpha.clamp(eps, 1 - eps)
    return -(a * a.log() + (1 - a) * (1 - a).log())


def run_m2(bank: dict, eligible_days, T: int, lr: float = 0.05, l2: float = 1e-3,
           entropy_reg: float = 1e-3, smooth_reg: float = 1e-3, epochs_per_day: int = 3,
           seed: int = 0, context: np.ndarray = None, day: np.ndarray = None):
    """Returns rows: {day, y_true, y_pred, n_train, fit_time, alpha,
    mean_alpha, std_alpha}. `alpha` is the full per-example array actually
    used for day t's deployed prediction (gate trained on days < t only).
    `context`/`day` (e.g. data.raw_numeric_features and the full per-row
    day array) are optional per-example context columns appended to the
    gate's input alongside the base short/long features -- without them
    the gate can only condition on the short/long candidates' own
    (dis)agreement, which carries no direct subpopulation signal."""
    torch.manual_seed(seed)
    days = short_long_days(bank, eligible_days)
    if not days:
        return []

    n_context = context.shape[1] if context is not None else 0
    gate = ContextGate(len(BASE_FEATURE_NAMES) + n_context)
    opt = torch.optim.Adam(gate.parameters(), lr=lr)
    prev_mean_alpha = torch.tensor(0.5)

    rows = []
    for t in days:
        feats, p_s, p_l = gate_feature_matrix(bank, days, t, T, context=context, day=day)
        y_true = bank[SHORT_NAME][t]["y_true"]
        feats_t = torch.tensor(feats, dtype=torch.float32)
        p_s_t = torch.tensor(p_s, dtype=torch.float32)
        p_l_t = torch.tensor(p_l, dtype=torch.float32)
        y_t = torch.tensor(y_true, dtype=torch.float32)

        with torch.no_grad():
            alpha_t = gate(feats_t).numpy()
        y_pred = mix(p_s, p_l, alpha_t)

        rows.append({
            "day": t,
            "y_true": y_true,
            "y_pred": y_pred,
            "n_train": int(bank[SHORT_NAME][t]["n_train"] + bank[LONG_NAME][t]["n_train"]),
            "fit_time": float(bank[SHORT_NAME][t]["fit_time"] + bank[LONG_NAME][t]["fit_time"]),
            "alpha": alpha_t,
            "mean_alpha": float(alpha_t.mean()),
            "std_alpha": float(alpha_t.std()),
        })

        # Update the gate only now that day t's true labels are observed --
        # used for t+1 onward, never for day t itself.
        for _ in range(epochs_per_day):
            opt.zero_grad()
            alpha = gate(feats_t)
            p_mix = (1 - alpha) * p_l_t + alpha * p_s_t
            p_mix = p_mix.clamp(1e-7, 1 - 1e-7)
            bce = -(y_t * p_mix.log() + (1 - y_t) * (1 - p_mix).log()).mean()
            l2_term = sum((p ** 2).sum() for p in gate.parameters())
            entropy_term = -_entropy(alpha).mean()  # minimized -> maximizes entropy -> alpha toward 0.5
            smooth_term = (alpha.mean() - prev_mean_alpha).abs()
            loss = bce + l2 * l2_term + entropy_reg * entropy_term + smooth_reg * smooth_term
            loss.backward()
            opt.step()
        with torch.no_grad():
            prev_mean_alpha = gate(feats_t).mean().detach()

    return rows
