"""Shared short/long-memory utilities for M1 (global adaptive mixing) and
M2 (context-dependent gating) -- adaptive-training-methods-implementation-
plan.md sections 5-7.

Both methods mix a short-memory predictor p_S(x) and a long-memory
predictor p_L(x) rather than fitting a new base model: p_S/p_L are read
straight from the shared WINDOW_FAMILY candidate bank (candidate_bank.py),
the same one han_arw.py and adamoe.py select over / aggregate, so the
short/long fits themselves are already leakage-safe (each trained only on
data < t) and shared across methods instead of duplicated.

Short = rolling_3, long = expanding, per the plan's "recommended initial
definitions" (section 5).
"""
import numpy as np

from han_arw import per_sample_log_loss

SHORT_NAME = "rolling_3"
LONG_NAME = "expanding"

BASE_FEATURE_NAMES = ["p_short", "p_long", "abs_disagreement", "signed_disagreement",
                       "norm_time", "recent_short_loss", "recent_long_loss"]


def short_long_days(bank: dict, eligible_days) -> list:
    """Prediction days for which both the short and long candidate exist."""
    return sorted(t for t in eligible_days if t in bank[SHORT_NAME] and t in bank[LONG_NAME])


def mix(p_s: np.ndarray, p_l: np.ndarray, alpha) -> np.ndarray:
    """(1-alpha)*p_L + alpha*p_S, alpha either a scalar (M1) or a
    per-example array (M2)."""
    return (1 - alpha) * p_l + alpha * p_s


def oracle_alpha(y_true: np.ndarray, p_s: np.ndarray, p_l: np.ndarray, grid: np.ndarray):
    """M1a: hindsight-best global alpha using day t's own labels.
    Diagnostic only -- must never be used to produce a deployed prediction
    (plan section 6, M1a). Returns (alpha*, achieved mean log loss)."""
    losses = np.array([per_sample_log_loss(y_true, mix(p_s, p_l, a)).mean() for a in grid])
    best = int(losses.argmin())
    return float(grid[best]), float(losses[best])


def state_features(bank: dict, day_list: list, t: int, n: int, T: int):
    """Per-day scalar temporal-state features available at prediction time
    for day t, using only days < t: most recent matured day's short/long
    mean log loss, and normalized time index. Broadcast to n rows since
    day t's own labels aren't known yet."""
    past = [d for d in day_list if d < t]
    if not past:
        recent_s_loss = recent_l_loss = 0.0
    else:
        prev = past[-1]
        recent_s_loss = float(per_sample_log_loss(
            bank[SHORT_NAME][prev]["y_true"], bank[SHORT_NAME][prev]["y_pred"]).mean())
        recent_l_loss = float(per_sample_log_loss(
            bank[LONG_NAME][prev]["y_true"], bank[LONG_NAME][prev]["y_pred"]).mean())
    norm_t = t / max(T, 1)
    return np.full(n, recent_s_loss), np.full(n, recent_l_loss), np.full(n, norm_t)


def gate_feature_matrix(bank: dict, day_list: list, t: int, T: int,
                         context: np.ndarray = None, day: np.ndarray = None):
    """Compact per-example gate feature matrix for day t (plan section 7):
    [p_S, p_L, |p_S-p_L|, p_S-p_L, norm_time, recent_short_loss,
    recent_long_loss] plus, if given, the per-example context columns for
    day t (context[day == t], row-aligned with bank's y_true/y_pred via
    the same day==t mask fit_predict uses) -- e.g. data.raw_numeric_features,
    which is what lets the gate condition on more than just the short/long
    candidates' own agreement (needed for it to tell subpopulations apart,
    plan section 7's "selected original/context features"). None of these
    use day t's own true labels. Returns (features, p_s, p_l)."""
    p_s = bank[SHORT_NAME][t]["y_pred"]
    p_l = bank[LONG_NAME][t]["y_pred"]
    n = len(p_s)
    recent_s, recent_l, norm_t = state_features(bank, day_list, t, n, T)
    feats = np.stack([p_s, p_l, np.abs(p_s - p_l), p_s - p_l, norm_t, recent_s, recent_l], axis=1)
    if context is not None:
        feats = np.concatenate([feats, context[day == t]], axis=1)
    return feats, p_s, p_l
