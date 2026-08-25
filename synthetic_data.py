"""Synthetic CTR data with a controllable ground-truth drift schedule.

Motivation: the Criteo Attribution dataset (31 days) shows only shallow
distribution shift -- every P1/P2 adaptive method converges to roughly what
a static recency window already gives. Neither the papers being reproduced
here (Han et al., Differentiable Forgetting) nor the general CTR-benchmark
literature (AdaMoE's own Avazu benchmark, Ali-CCP, Criteo 1TB) offer a
public, CTR-native dataset with a genuinely long time horizon and
documented drift -- the two papers that needed real multi-month/multi-year
drift to demonstrate their own methods had to leave the CTR domain entirely
(text-topic trends, real-estate prices, equities).

This module follows their same workaround: a synthetic generator over a
much longer horizon, with a known ground-truth logistic CTR model whose
weights evolve over time according to a chosen schedule. Because we control
the true generating process, we can directly check whether each method
tracks it, rather than inferring drift indirectly from held-out loss on
data whose true process is unknown.

Interface matches data.load_dataset: returns (X, y, day), so every existing
P0/P1/P2 method works against it unchanged.
"""
import numpy as np
import pandas as pd

from data import hash_features

DRIFT_MODES = ["none", "gradual", "abrupt", "recurring", "local"]


def _drift_alpha(t: int, n_days: int, drift_mode: str, shift_day: int, period_days: int) -> float:
    """Mixing weight toward the drifted regime w1, in [0, 1]. Used directly
    by every global mode; "local" reuses the "abrupt" schedule but only
    applies it to group A rows (see group_membership / row_alpha below)."""
    if drift_mode == "none":
        return 0.0
    if drift_mode == "gradual":
        return t / max(n_days - 1, 1)
    if drift_mode in ("abrupt", "local"):
        return 0.0 if t < shift_day else 1.0
    if drift_mode == "recurring":
        return 0.5 * (1 + np.sin(2 * np.pi * t / period_days))
    raise ValueError(f"Unknown drift_mode: {drift_mode!r} (choices: {DRIFT_MODES})")


def group_membership(cat_vals: np.ndarray, cardinality: int) -> np.ndarray:
    """S4 local/subpopulation-drift group split (adaptive-training-methods
    plan section 12): group A is cat0 < cardinality // 2, group B is the
    rest -- a structural ~50/50 split defined independently of drift_mode,
    so downstream analysis can always break results down by group even
    when the mode doesn't drift them differently."""
    return cat_vals[:, 0] < (cardinality // 2)


def generate_synthetic_raw(n_days: int = 180, rows_per_day: int = 5000, n_cat_features: int = 10,
                            cardinality: int = 200, drift_mode: str = "gradual",
                            drift_magnitude: float = 1.0, shift_day: int = None, period_days: int = 14,
                            intercept: float = -2.0, seed: int = 0):
    """Same generation as generate_synthetic_ctr, but returns the raw
    (categorical columns, click, day) DataFrame before hashing -- used by
    sftl.py, which needs per-column integer indices for embedding lookups
    rather than the sparse hashed-vector representation the rest of the
    pipeline uses. Returns (df, columns)."""
    """Generate (X, y, day) with a ground-truth logistic CTR model
    w(t) = (1 - alpha(t)) * w0 + alpha(t) * w1, alpha(t) set by `drift_mode`:
      - "none": alpha=0 always -- stationary sanity check, no method should
        beat expanding-history ERM here.
      - "gradual": alpha ramps 0->drift_magnitude linearly over the horizon.
      - "abrupt": alpha jumps 0->drift_magnitude at `shift_day` (default the
        midpoint) -- tests recovery speed after a sharp regime change
        (PDF section 10's "recovers slowly after abrupt shifts").
      - "recurring": alpha oscillates with period `period_days` -- tests
        whether adaptive methods track a cyclical (e.g. seasonal) regime.
      - "local": the "abrupt" schedule applied only to group A rows
        (cat0 < cardinality // 2); group B's ground truth never moves.
        Tests whether context-dependent methods can shrink memory for the
        drifted subpopulation without discarding history that's still
        valid for the stable one (plan section 12, S4).

    w0 and w1 are independent random coefficient vectors over
    (categorical column, value) pairs, so "abrupt"/"local" are a full
    regime swap (for the affected rows), not a subtle perturbation. The
    returned df always includes a "group" column (bool, group A = True)
    so downstream analysis can break results down by group regardless of
    drift_mode.
    """
    if drift_mode not in DRIFT_MODES:
        raise ValueError(f"Unknown drift_mode: {drift_mode!r} (choices: {DRIFT_MODES})")
    if shift_day is None:
        shift_day = n_days // 2

    rng = np.random.default_rng(seed)
    columns = [f"cat{i}" for i in range(n_cat_features)]
    n_true_features = n_cat_features * cardinality
    w0 = rng.normal(0, 1.0, size=n_true_features)
    w1 = rng.normal(0, 1.0, size=n_true_features)
    col_offsets = np.arange(n_cat_features) * cardinality

    frames = []
    for t in range(n_days):
        cat_vals = rng.integers(0, cardinality, size=(rows_per_day, n_cat_features))
        true_idx = cat_vals + col_offsets[None, :]
        logits0 = w0[true_idx].sum(axis=1)
        logits1 = w1[true_idx].sum(axis=1)
        is_group_a = group_membership(cat_vals, cardinality)

        if drift_mode == "local":
            row_alpha = np.where(is_group_a, _drift_alpha(t, n_days, "local", shift_day, period_days), 0.0)
        else:
            row_alpha = np.full(rows_per_day, _drift_alpha(t, n_days, drift_mode, shift_day, period_days))
        row_alpha = row_alpha * drift_magnitude

        logits = (1 - row_alpha) * logits0 + row_alpha * logits1 + intercept
        p = 1 / (1 + np.exp(-logits))
        y_t = rng.binomial(1, p)

        df_t = pd.DataFrame(cat_vals, columns=columns)
        df_t["click"] = y_t
        df_t["day"] = t
        df_t["group"] = is_group_a
        frames.append(df_t)

    df = pd.concat(frames, ignore_index=True)
    return df, columns


def generate_synthetic_ctr(n_days: int = 180, rows_per_day: int = 5000, n_cat_features: int = 10,
                            cardinality: int = 200, drift_mode: str = "gradual",
                            drift_magnitude: float = 1.0, shift_day: int = None, period_days: int = 14,
                            intercept: float = -2.0, n_features: int = 2**18, seed: int = 0,
                            return_group: bool = False):
    """Same drift-schedule generator as generate_synthetic_raw, hashed into
    (X, y, day) ready for the SGDClassifier-based P0/P1/P2 methods. With
    return_group=True also returns the S4 group-A/B split (bool array),
    for evaluating local-drift methods' shifted-vs-stable subgroup loss
    (plan section 15's "local adaptation gap") -- backward compatible,
    existing callers get the same (X, y, day) 3-tuple as before."""
    df, columns = generate_synthetic_raw(
        n_days=n_days, rows_per_day=rows_per_day, n_cat_features=n_cat_features, cardinality=cardinality,
        drift_mode=drift_mode, drift_magnitude=drift_magnitude, shift_day=shift_day, period_days=period_days,
        intercept=intercept, seed=seed)
    X = hash_features(df, columns=columns, n_features=n_features)
    y = df["click"].to_numpy()
    day = df["day"].to_numpy()
    if return_group:
        return X, y, day, df["group"].to_numpy()
    return X, y, day
