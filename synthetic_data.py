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

DRIFT_MODES = ["none", "gradual", "abrupt", "recurring", "local", "opposing_local",
               "mixed", "opposing_recurring"]

# AMG-TP plan (AMG-TP_Academic_LaTeX.pdf, Table 2) adds regimes beyond the
# original five:
#   S5 "opposing_local"  -- two subpopulations drift at *different times and in
#                           different directions*: group A swaps w0->w1 abruptly
#                           at n_days//3, group B swaps w0->w2 (an independent
#                           regime) abruptly at 2*n_days//3. Stress-tests the
#                           assumption behind a single global persistence/memory
#                           knob, since no one global schedule fits both groups.
#   S6 "mixed"           -- a per-seed random sequence of shift events (stationary
#                           stretches, abrupt jumps to freshly drawn regimes, and
#                           linear ramps), so a method must generalise across
#                           shift types within one run without any regime label.
#   S7 "opposing_recurring" -- both subpopulations oscillate on the w0<->w1 axis
#                           but a *quarter period* out of phase, so at every block
#                           one group sits near a turning point (slow local change
#                           -> high persistence optimal) while the other is at
#                           maximum slope (fast change -> low persistence optimal),
#                           and they swap roles over the cycle. Unlike
#                           opposing_local the differential is present at *every*
#                           block rather than being a one-off transient, so it
#                           survives into any fixed locked-test window. This is the
#                           test bed for per-example persistence beta_t(x): a single
#                           global beta_t is provably wrong for one group at all
#                           times, while the context gate q_t(x) alone cannot fix
#                           it (it routes experts, not persistence).
_MIXED_N_REGIMES = 4


def _mixed_schedule(n_days: int, n_regimes: int, rng: np.random.Generator) -> dict:
    """Random piecewise ground-truth schedule for drift_mode='mixed' (S6).

    The horizon is cut into 4-8 contiguous segments. The process starts in the
    base regime (index 0 = w0); at each segment boundary it transitions -- either
    abruptly (jump on the segment's first day) or as a linear ramp over the first
    third of the segment -- to a regime index drawn uniformly from
    [0, n_regimes] (0 = w0, 1..n_regimes = the extra regime bank). Returns
    per-day int arrays `from_idx`, `to_idx` and a float array `frac` in [0, 1]:
    day t's true weight vector is (1-frac[t]) * W[from_idx[t]] + frac[t] * W[to_idx[t]].
    """
    n_seg = int(rng.integers(4, 9))
    bounds = np.linspace(0, n_days, n_seg + 1).astype(int)
    regime_seq = [0]
    for _ in range(n_seg):
        regime_seq.append(int(rng.integers(0, n_regimes + 1)))

    from_idx = np.zeros(n_days, dtype=int)
    to_idx = np.zeros(n_days, dtype=int)
    frac = np.zeros(n_days, dtype=float)
    for s in range(n_seg):
        lo, hi = int(bounds[s]), int(bounds[s + 1])
        if hi <= lo:
            continue
        prev_regime, cur_regime = regime_seq[s], regime_seq[s + 1]
        gradual = bool(rng.integers(0, 2))
        ramp = max(1, (hi - lo) // 3) if gradual else 1
        for i, t in enumerate(range(lo, hi)):
            from_idx[t] = prev_regime
            to_idx[t] = cur_regime
            frac[t] = min(1.0, (i + 1) / ramp) if gradual else 1.0
    return {"from_idx": from_idx, "to_idx": to_idx, "frac": frac,
            "n_seg": n_seg, "regime_seq": regime_seq, "bounds": bounds.tolist()}


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
      - "opposing_recurring": both groups oscillate w0<->w1 with period
        `period_days`, group B a quarter period behind group A, so their
        optimal memory length is anti-correlated at every block (S7 -- the
        per-example persistence test bed; see module docstring).

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

    # Extra ground-truth regimes for the multi-regime modes. Drawn only when
    # needed and *after* w0/w1, so the RNG stream (hence the exact data) for the
    # original five modes is byte-for-byte unchanged.
    w2 = None
    mixed_sched = None
    W_mixed = None
    if drift_mode == "opposing_local":
        w2 = rng.normal(0, 1.0, size=n_true_features)
        shift_day_a = n_days // 3
        shift_day_b = (2 * n_days) // 3
    elif drift_mode == "opposing_recurring":
        # Both groups share the w0<->w1 axis (no w2 drawn -> RNG stream for the
        # other modes is untouched); the two subpopulations differ only in the
        # phase of their oscillation.
        omega = 2 * np.pi / max(period_days, 1)
        phase_b = np.pi / 2  # quarter period: A slow <-> B fast, and vice versa
    elif drift_mode == "mixed":
        W_mixed = [rng.normal(0, 1.0, size=n_true_features) for _ in range(_MIXED_N_REGIMES)]
        mixed_sched = _mixed_schedule(n_days, _MIXED_N_REGIMES, rng)

    frames = []
    for t in range(n_days):
        cat_vals = rng.integers(0, cardinality, size=(rows_per_day, n_cat_features))
        true_idx = cat_vals + col_offsets[None, :]
        logits0 = w0[true_idx].sum(axis=1)
        logits1 = w1[true_idx].sum(axis=1)
        is_group_a = group_membership(cat_vals, cardinality)

        if drift_mode == "opposing_local":
            logits2 = w2[true_idx].sum(axis=1)
            a_a = _drift_alpha(t, n_days, "abrupt", shift_day_a, period_days) * drift_magnitude
            a_b = _drift_alpha(t, n_days, "abrupt", shift_day_b, period_days) * drift_magnitude
            logits = np.where(is_group_a,
                              (1 - a_a) * logits0 + a_a * logits1,
                              (1 - a_b) * logits0 + a_b * logits2) + intercept
            row_alpha = np.where(is_group_a, a_a, a_b)  # bookkeeping only
        elif drift_mode == "opposing_recurring":
            a_a = 0.5 * (1 + np.sin(omega * t)) * drift_magnitude
            a_b = 0.5 * (1 + np.sin(omega * t + phase_b)) * drift_magnitude
            row_alpha = np.where(is_group_a, a_a, a_b)
            logits = (1 - row_alpha) * logits0 + row_alpha * logits1 + intercept
        elif drift_mode == "mixed":
            j_from = int(mixed_sched["from_idx"][t])
            j_to = int(mixed_sched["to_idx"][t])
            f = float(mixed_sched["frac"][t]) * drift_magnitude
            lf = logits0 if j_from == 0 else W_mixed[j_from - 1][true_idx].sum(axis=1)
            lt = logits0 if j_to == 0 else W_mixed[j_to - 1][true_idx].sum(axis=1)
            logits = (1 - f) * lf + f * lt + intercept
            row_alpha = np.full(rows_per_day, f)  # bookkeeping only
        else:
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
        df_t["p_true"] = p          # ground-truth click probability (for oracle / cost models)
        frames.append(df_t)

    df = pd.concat(frames, ignore_index=True)
    return df, columns


def generate_synthetic_ctr(n_days: int = 180, rows_per_day: int = 5000, n_cat_features: int = 10,
                            cardinality: int = 200, drift_mode: str = "gradual",
                            drift_magnitude: float = 1.0, shift_day: int = None, period_days: int = 14,
                            intercept: float = -2.0, n_features: int = 2**18, seed: int = 0,
                            return_group: bool = False, return_p: bool = False):
    """Same drift-schedule generator as generate_synthetic_raw, hashed into
    (X, y, day) ready for the SGDClassifier-based P0/P1/P2 methods. With
    return_group=True also returns the S4 group-A/B split (bool array),
    for evaluating local-drift methods' shifted-vs-stable subgroup loss
    (plan section 15's "local adaptation gap"); with return_p=True also
    returns the ground-truth per-row click probability (used by the
    autobidding eval's oracle bidder and synthetic cost landscape) --
    backward compatible, existing callers get the same (X, y, day) 3-tuple.
    When both flags are set the order is (X, y, day, group, p)."""
    df, columns = generate_synthetic_raw(
        n_days=n_days, rows_per_day=rows_per_day, n_cat_features=n_cat_features, cardinality=cardinality,
        drift_mode=drift_mode, drift_magnitude=drift_magnitude, shift_day=shift_day, period_days=period_days,
        intercept=intercept, seed=seed)
    X = hash_features(df, columns=columns, n_features=n_features)
    y = df["click"].to_numpy()
    day = df["day"].to_numpy()
    out = [X, y, day]
    if return_group:
        out.append(df["group"].to_numpy())
    if return_p:
        out.append(df["p_true"].to_numpy())
    return tuple(out) if len(out) > 3 else (X, y, day)
