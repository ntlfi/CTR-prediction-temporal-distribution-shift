"""3-way ensemble: M2 (2-expert short/long gate) / M5b-default
(smooth_reg=1e-3) / M5b-high-smooth (smooth_reg=0.1) -- generalizes
ensemble_m2_m5.py's 2-way meta-gate to 3 experts, reusing
m5_multiscale_gate.MultiExpertGate (already generic over expert count).

Motivation (results/m5_analysis.md's smooth_reg sweep): M5b-high-smooth
beats every method tried on recurring drift so far (0.4118 vs M2's 0.4180),
but at the cost of clear regressions on abrupt (+7.1%) and local (+3.7%)
drift relative to plain M5b-default -- raising smooth_reg trades M5b's
step-function-regime strength for recurring-regime strength rather than
improving it outright, since recurring drift is smooth (a continuous
sinusoid) while abrupt/local are literal step functions a heavily-smoothed
gate reacts to sluggishly. Rather than pick one smooth_reg globally, this
learns a per-example blend of all three specialists' predictions:

    p_hat_t(x) = sum_k pi_{t,k}(x) * p_t^(k)(x),  k in {m2, m5b, m5b_hs}
    pi_t(x) = softmax(gate_phi(v_t(x)))

Same online causal rule as every gate in this project: day t's prediction
only ever uses a gate fit on days < t.
"""
import numpy as np
import torch

from han_arw import per_sample_log_loss
from m5_multiscale_gate import MultiExpertGate, _entropy

EXPERTS = ["m2", "m5b", "m5b_hs"]
BASE_FEATURE_NAMES = ["spread", "norm_time", "recent_m2_loss", "recent_m5b_loss", "recent_m5b_hs_loss"]


def ensemble_days(*rows_by_day_dicts) -> list:
    """Prediction days for which every expert produced a prediction."""
    days = set(rows_by_day_dicts[0])
    for d in rows_by_day_dicts[1:]:
        days &= set(d)
    return sorted(days)


def _recent_loss(rows_by_day: dict, day_list: list, t: int) -> float:
    past = [d for d in day_list if d < t]
    if not past:
        return 0.0
    r = rows_by_day[past[-1]]
    return float(per_sample_log_loss(r["y_true"], r["y_pred"]).mean())


def gate_feature_matrix(m2_by_day: dict, m5b_by_day: dict, m5hs_by_day: dict, day_list: list, t: int, T: int,
                         context: np.ndarray = None, day: np.ndarray = None):
    """Per-example gate feature matrix for day t: [p_m2, p_m5b, p_m5b_hs,
    spread=max-min across the three, norm_time, each expert's recent mean
    log loss] plus, if given, per-example context -- none use day t's own
    labels. Returns (features, preds (n, 3))."""
    p_m2 = m2_by_day[t]["y_pred"]
    p_m5b = m5b_by_day[t]["y_pred"]
    p_m5hs = m5hs_by_day[t]["y_pred"]
    preds = np.stack([p_m2, p_m5b, p_m5hs], axis=1)  # (n, 3)
    n = preds.shape[0]
    spread = preds.max(axis=1) - preds.min(axis=1)
    norm_t = np.full(n, t / max(T, 1))
    recent = [_recent_loss(m2_by_day, day_list, t), _recent_loss(m5b_by_day, day_list, t),
              _recent_loss(m5hs_by_day, day_list, t)]
    recent_arr = np.tile(recent, (n, 1))  # (n, 3)
    feats = np.concatenate([preds, spread[:, None], norm_t[:, None], recent_arr], axis=1)
    if context is not None:
        feats = np.concatenate([feats, context[day == t]], axis=1)
    return feats, preds


def run_ensemble3(m2_rows: list, m5b_rows: list, m5hs_rows: list, T: int, lr: float = 0.05, l2: float = 1e-3,
                   entropy_reg: float = 1e-3, smooth_reg: float = 1e-3, epochs_per_day: int = 3,
                   seed: int = 0, context: np.ndarray = None, day: np.ndarray = None):
    """Returns rows: {day, y_true, y_pred, n_train, fit_time, weights,
    mean_weights (dict expert -> mean pi across x)}. `y_pred` is the
    per-example softmax mixture actually used for day t's deployed
    prediction (gate trained on days < t only)."""
    torch.manual_seed(seed)
    m2_by_day = {r["day"]: r for r in m2_rows}
    m5b_by_day = {r["day"]: r for r in m5b_rows}
    m5hs_by_day = {r["day"]: r for r in m5hs_rows}
    days = ensemble_days(m2_by_day, m5b_by_day, m5hs_by_day)
    if not days:
        return []

    n_experts = len(EXPERTS)
    n_context = context.shape[1] if context is not None else 0
    n_features = n_experts + len(BASE_FEATURE_NAMES) + n_context
    gate = MultiExpertGate(n_features, n_experts)
    opt = torch.optim.Adam(gate.parameters(), lr=lr)
    prev_mean_pi = torch.full((n_experts,), 1.0 / n_experts)

    rows = []
    for t in days:
        feats, preds = gate_feature_matrix(m2_by_day, m5b_by_day, m5hs_by_day, days, t, T,
                                            context=context, day=day)
        y_true = m2_by_day[t]["y_true"]
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
            "n_train": int(m2_by_day[t]["n_train"] + m5b_by_day[t]["n_train"] + m5hs_by_day[t]["n_train"]),
            "fit_time": float(m2_by_day[t]["fit_time"] + m5b_by_day[t]["fit_time"] + m5hs_by_day[t]["fit_time"]),
            "weights": pi,
            "mean_weights": dict(zip(EXPERTS, mean_pi.tolist())),
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
