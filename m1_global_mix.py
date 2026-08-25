"""M1 -- Global Adaptive Short/Long Mixing (adaptive-training-methods-
implementation-plan.md section 6). One global alpha_t at a time mixes the
short-memory (rolling_3) and long-memory (expanding) candidate-bank
predictions:

    p_hat_t(x) = (1-alpha_t) p_L(x) + alpha_t p_S(x)

M1b (deployed): alpha_t is chosen from a fixed grid by minimizing pooled
mean log loss of the mixture over the most recent `val_window` *matured*
days (their true labels are already known when choosing alpha_t) -- never
day t's own labels. This is the cheapest proof-of-concept method in the
plan (Stage 1) and the baseline M2's context-dependent gate is compared
against.

M1a (oracle) is computed alongside purely as a diagnostic: the
hindsight-best alpha using day t's own labels, quantifying how much
headroom a perfect global mixture would have over the deployed choice.
Per the plan, this must never be fed back into the deployed prediction.
"""
import numpy as np

from han_arw import per_sample_log_loss
from short_long import LONG_NAME, SHORT_NAME, mix, oracle_alpha, short_long_days

ALPHA_GRID = np.round(np.linspace(0.0, 1.0, 11), 2)


def _select_alpha(bank: dict, history_days: list, grid: np.ndarray) -> float:
    """Grid-search alpha minimizing pooled mixture log loss across the
    given (already-matured) history days."""
    losses = np.zeros(len(grid))
    for d in history_days:
        y_true = bank[SHORT_NAME][d]["y_true"]
        p_s = bank[SHORT_NAME][d]["y_pred"]
        p_l = bank[LONG_NAME][d]["y_pred"]
        for i, a in enumerate(grid):
            losses[i] += per_sample_log_loss(y_true, mix(p_s, p_l, a)).sum()
    return float(grid[int(losses.argmin())])


def run_m1(bank: dict, eligible_days, val_window: int = 3, grid: np.ndarray = ALPHA_GRID,
           min_history: int = 1):
    """Returns rows: {day, y_true, y_pred, n_train, fit_time, alpha,
    alpha_oracle, oracle_headroom}. `alpha` is the deployed M1b choice;
    `alpha_oracle`/`oracle_headroom` are the M1a diagnostic (never fed
    back into a prediction). Skips days with fewer than `min_history`
    matured prior days available to select alpha from."""
    days = short_long_days(bank, eligible_days)
    rows = []
    for t in days:
        history = [d for d in days if d < t][-val_window:]
        if len(history) < min_history:
            continue
        alpha_t = _select_alpha(bank, history, grid)

        p_s = bank[SHORT_NAME][t]["y_pred"]
        p_l = bank[LONG_NAME][t]["y_pred"]
        y_true = bank[SHORT_NAME][t]["y_true"]
        y_pred = mix(p_s, p_l, alpha_t)

        alpha_oracle, oracle_loss = oracle_alpha(y_true, p_s, p_l, grid)
        deployed_loss = float(per_sample_log_loss(y_true, y_pred).mean())

        rows.append({
            "day": t,
            "y_true": y_true,
            "y_pred": y_pred,
            "n_train": int(bank[SHORT_NAME][t]["n_train"] + bank[LONG_NAME][t]["n_train"]),
            "fit_time": float(bank[SHORT_NAME][t]["fit_time"] + bank[LONG_NAME][t]["fit_time"]),
            "alpha": alpha_t,
            "alpha_oracle": alpha_oracle,
            "oracle_headroom": deployed_loss - oracle_loss,
        })
    return rows
