"""M2+M5b Ensemble -- a meta-gate over the two context-dependent gating
methods (m2_context_gate.py, m5_multiscale_gate.py).

Motivation (results/m5_analysis.md): on the synthetic drift suite, M2 (a
2-expert short=rolling_3/long=expanding gate) wins outright under recurring
(cyclical) and does fine elsewhere, but trails han_arw/rolling_14 under
abrupt/gradual because its 2-candidate mixture family can't reach
rolling_14. M5b (a 5-expert gate over the full WINDOW_FAMILY) fixes exactly
that -- it wins abrupt/gradual/local -- but is *worse* than M2 under
recurring, where a wider expert pool apparently dilutes the sharper 2-expert
blend. Since neither dominates the other, this learns a per-example blend
of their two final *predictions* instead of picking one upfront:

    p_hat_t(x) = (1-beta_t(x)) * p_M2,t(x) + beta_t(x) * p_M5,t(x)
    beta_t(x) = sigmoid(meta_gate(v_t(x)))

Same online causal-training rule as M2/M5b: the meta-gate used for day t's
prediction was trained only on days < t's already-matured (features, label)
pairs; it updates after day t's true labels are observed, for day t+1
onward. Zero-initialized so beta(x)=0.5 everywhere before any training --
a plain average of M2 and M5b, not worse than the average of the two
individually as a starting point -- plus the same L2/entropy/day-smoothness
regularization pattern used throughout this plan's gated methods.

Takes M2's and M5b's already-computed per-day rows (from run_m2/run_m5) as
input rather than the candidate bank directly -- no new base-model fitting,
this only ever mixes two already-deployed predictions.
"""
import numpy as np
import torch
import torch.nn as nn

from han_arw import per_sample_log_loss

BASE_FEATURE_NAMES = ["p_m2", "p_m5", "abs_disagreement", "signed_disagreement", "norm_time",
                       "recent_m2_loss", "recent_m5_loss"]


def ensemble_days(m2_by_day: dict, m5_by_day: dict) -> list:
    """Prediction days for which both M2 and M5b produced a prediction."""
    return sorted(set(m2_by_day) & set(m5_by_day))


def _recent_losses(m2_by_day: dict, m5_by_day: dict, day_list: list, t: int):
    past = [d for d in day_list if d < t]
    if not past:
        return 0.0, 0.0
    prev = past[-1]
    m2_loss = float(per_sample_log_loss(m2_by_day[prev]["y_true"], m2_by_day[prev]["y_pred"]).mean())
    m5_loss = float(per_sample_log_loss(m5_by_day[prev]["y_true"], m5_by_day[prev]["y_pred"]).mean())
    return m2_loss, m5_loss


def gate_feature_matrix(m2_by_day: dict, m5_by_day: dict, day_list: list, t: int, T: int,
                         context: np.ndarray = None, day: np.ndarray = None):
    """Per-example meta-gate feature matrix for day t: [p_M2, p_M5,
    |p_M2-p_M5|, p_M2-p_M5, norm_time, recent M2/M5 mean log loss] plus,
    if given, per-example context columns -- none use day t's own labels.
    Returns (features, p_m2, p_m5)."""
    p_m2 = m2_by_day[t]["y_pred"]
    p_m5 = m5_by_day[t]["y_pred"]
    n = p_m2.shape[0]
    abs_disagreement = np.abs(p_m2 - p_m5)
    signed_disagreement = p_m2 - p_m5
    norm_t = np.full(n, t / max(T, 1))
    recent_m2_loss, recent_m5_loss = _recent_losses(m2_by_day, m5_by_day, day_list, t)
    feats = np.stack([p_m2, p_m5, abs_disagreement, signed_disagreement, norm_t,
                       np.full(n, recent_m2_loss), np.full(n, recent_m5_loss)], axis=1)
    if context is not None:
        feats = np.concatenate([feats, context[day == t]], axis=1)
    return feats, p_m2, p_m5


class MetaGate(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(feats)).squeeze(-1)


def _entropy(beta: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    b = beta.clamp(eps, 1 - eps)
    return -(b * b.log() + (1 - b) * (1 - b).log())


def run_ensemble(m2_rows: list, m5_rows: list, T: int, lr: float = 0.05, l2: float = 1e-3,
                  entropy_reg: float = 1e-3, smooth_reg: float = 1e-3, epochs_per_day: int = 3,
                  seed: int = 0, context: np.ndarray = None, day: np.ndarray = None):
    """Returns rows: {day, y_true, y_pred, n_train, fit_time, beta,
    mean_beta, std_beta}. `beta` is the full per-example array actually
    used for day t's deployed prediction (meta-gate trained on days < t
    only). `beta` near 0 means "trust M2 here", near 1 means "trust M5b"."""
    torch.manual_seed(seed)
    m2_by_day = {r["day"]: r for r in m2_rows}
    m5_by_day = {r["day"]: r for r in m5_rows}
    days = ensemble_days(m2_by_day, m5_by_day)
    if not days:
        return []

    n_context = context.shape[1] if context is not None else 0
    gate = MetaGate(len(BASE_FEATURE_NAMES) + n_context)
    opt = torch.optim.Adam(gate.parameters(), lr=lr)
    prev_mean_beta = torch.tensor(0.5)

    rows = []
    for t in days:
        feats, p_m2, p_m5 = gate_feature_matrix(m2_by_day, m5_by_day, days, t, T, context=context, day=day)
        y_true = m2_by_day[t]["y_true"]
        feats_t = torch.tensor(feats, dtype=torch.float32)
        p_m2_t = torch.tensor(p_m2, dtype=torch.float32)
        p_m5_t = torch.tensor(p_m5, dtype=torch.float32)
        y_t = torch.tensor(y_true, dtype=torch.float32)

        with torch.no_grad():
            beta_t = gate(feats_t).numpy()
        y_pred = (1 - beta_t) * p_m2 + beta_t * p_m5

        rows.append({
            "day": t,
            "y_true": y_true,
            "y_pred": y_pred,
            "n_train": int(m2_by_day[t]["n_train"] + m5_by_day[t]["n_train"]),
            "fit_time": float(m2_by_day[t]["fit_time"] + m5_by_day[t]["fit_time"]),
            "beta": beta_t,
            "mean_beta": float(beta_t.mean()),
            "std_beta": float(beta_t.std()),
        })

        # Update the meta-gate only now that day t's true labels are observed --
        # used for t+1 onward, never for day t itself.
        for _ in range(epochs_per_day):
            opt.zero_grad()
            beta_train = gate(feats_t)
            p_mix = ((1 - beta_train) * p_m2_t + beta_train * p_m5_t).clamp(1e-7, 1 - 1e-7)
            bce = -(y_t * p_mix.log() + (1 - y_t) * (1 - p_mix).log()).mean()
            l2_term = sum((p ** 2).sum() for p in gate.parameters())
            entropy_term = -_entropy(beta_train).mean()
            smooth_term = (beta_train.mean() - prev_mean_beta).abs()
            loss = bce + l2 * l2_term + entropy_reg * entropy_term + smooth_reg * smooth_term
            loss.backward()
            opt.step()
        with torch.no_grad():
            prev_mean_beta = gate(feats_t).mean().detach()

    return rows
