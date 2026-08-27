"""Leakage and edge-case tests for the AMG-TP experimental plan
(adaptive-training-methods-implementation-plan.md section 21, PDF section 7).

Run:  .venv/bin/python amgtp_tests.py
Exits non-zero if any check fails. Fast (small synthetic data, ~30s).
"""
import sys

import numpy as np

FAILS = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def _small_bank(drift="abrupt", n_days=40, seed=0, rows=600):
    from splits import compute_splits
    from candidate_bank import build_candidate_bank
    from synthetic_data import generate_synthetic_raw
    from data import hash_features, raw_numeric_features
    df, cols = generate_synthetic_raw(n_days=n_days, rows_per_day=rows, drift_mode=drift,
                                      shift_day=n_days // 2, seed=seed)
    X = hash_features(df, columns=cols, n_features=2**16)
    context = raw_numeric_features(df, columns=cols)
    y = df["click"].to_numpy()
    day = df["day"].to_numpy()
    group = df["group"].to_numpy()
    eligible, dev, test = compute_splits(day, 3, 0.3)
    bank = build_candidate_bank(X, y, day, list(eligible), n_jobs=4)
    return dict(X=X, y=y, day=day, group=group, context=context,
                eligible=list(eligible), dev=dev, test=test, T=int(day.max()))


def test_no_future_leakage(d):
    """Perturb all labels for days strictly after a cutoff, rebuild everything,
    and confirm every prediction on or before the cutoff is bit-identical
    (plan 21.1)."""
    from candidate_bank import build_candidate_bank
    from m5_multiscale_gate import run_m5

    cutoff = sorted(d["test"])[len(d["test"]) // 2]
    base = run_m5(build_candidate_bank(d["X"], d["y"], d["day"], d["eligible"], n_jobs=4),
                  d["eligible"], T=d["T"], smooth_reg=0.1, context=d["context"], day=d["day"], seed=0)

    y2 = d["y"].copy()
    future = d["day"] > cutoff
    y2[future] = 1 - y2[future]
    pert = run_m5(build_candidate_bank(d["X"], y2, d["day"], d["eligible"], n_jobs=4),
                  d["eligible"], T=d["T"], smooth_reg=0.1, context=d["context"], day=d["day"], seed=0)

    b = {r["day"]: r["y_pred"] for r in base}
    p = {r["day"]: r["y_pred"] for r in pert}
    max_diff = 0.0
    for t in b:
        if t <= cutoff and t in p:
            max_diff = max(max_diff, float(np.abs(b[t] - p[t]).max()))
    check("no future leakage (M5b-high-smooth predictions <= cutoff unchanged)",
          max_diff == 0.0, f"max |Δpred| on/before day {cutoff} = {max_diff:.2e}")


def test_gate_range(d):
    """Every deployed gate weight in [0,1] and each example's expert
    distribution sums to 1 (plan 21.2)."""
    from m5_multiscale_gate import run_m5
    from candidate_bank import build_candidate_bank
    bank = build_candidate_bank(d["X"], d["y"], d["day"], d["eligible"], n_jobs=4)
    rows = run_m5(bank, d["eligible"], T=d["T"], smooth_reg=0.1, context=d["context"], day=d["day"], seed=0)
    lo = min(float(r["weights"].min()) for r in rows)
    hi = max(float(r["weights"].max()) for r in rows)
    sums = np.concatenate([r["weights"].sum(axis=1) for r in rows])
    check("gate weights in [0,1]", lo >= -1e-9 and hi <= 1 + 1e-9, f"range [{lo:.3e}, {hi:.3e}]")
    check("per-example expert weights sum to 1", np.allclose(sums, 1.0, atol=1e-5),
          f"max |sum-1| = {float(np.abs(sums - 1).max()):.2e}")


def test_mixture_identity(d):
    """short_long.mix: alpha=0 reproduces the long model, alpha=1 the short
    model (plan 21.3)."""
    from short_long import mix
    p_s = np.array([0.1, 0.4, 0.9])
    p_l = np.array([0.2, 0.5, 0.7])
    check("mix(alpha=0) == p_long", np.allclose(mix(p_s, p_l, 0.0), p_l))
    check("mix(alpha=1) == p_short", np.allclose(mix(p_s, p_l, 1.0), p_s))


def test_local_shift_isolation():
    """Under S4 local drift, only group A's conditional label distribution
    moves; group B's stays put (plan 21.5)."""
    from synthetic_data import generate_synthetic_raw
    df, _ = generate_synthetic_raw(n_days=60, rows_per_day=4000, drift_mode="local",
                                   shift_day=30, seed=0)
    pre = df[df["day"] < 30]
    post = df[df["day"] >= 30]
    a_pre = pre[pre["group"]]["click"].mean()
    a_post = post[post["group"]]["click"].mean()
    b_pre = pre[~pre["group"]]["click"].mean()
    b_post = post[~post["group"]]["click"].mean()
    check("S4 local drift: group A CTR moves at the shift", abs(a_post - a_pre) > 0.02,
          f"A: {a_pre:.3f} -> {a_post:.3f}")
    check("S4 local drift: group B CTR stays put", abs(b_post - b_pre) < 0.015,
          f"B: {b_pre:.3f} -> {b_post:.3f}")


def test_opposing_local_isolation():
    """Under S5, the two groups change at different times."""
    from synthetic_data import generate_synthetic_raw
    n = 90
    df, _ = generate_synthetic_raw(n_days=n, rows_per_day=4000, drift_mode="opposing_local", seed=0)
    def ctr(lo, hi, grp):
        s = df[(df["day"] >= lo) & (df["day"] < hi)]
        return s[s["group"] == grp]["click"].mean()
    a_early, a_mid, a_late = ctr(0, n // 3, True), ctr(n // 3, 2 * n // 3, True), ctr(2 * n // 3, n, True)
    b_early, b_mid, b_late = ctr(0, n // 3, False), ctr(n // 3, 2 * n // 3, False), ctr(2 * n // 3, n, False)
    check("S5: group A shifts at ~n/3", abs(a_mid - a_early) > 0.02, f"A {a_early:.3f}->{a_mid:.3f}")
    check("S5: group B still stable across n/3", abs(b_mid - b_early) < 0.02, f"B {b_early:.3f}->{b_mid:.3f}")
    check("S5: group B shifts at ~2n/3", abs(b_late - b_mid) > 0.02, f"B {b_mid:.3f}->{b_late:.3f}")


def test_reproducibility(d):
    """Same seed -> identical predictions within numerical tolerance (plan 21.6)."""
    from m5_multiscale_gate import run_m5
    from candidate_bank import build_candidate_bank
    bank = build_candidate_bank(d["X"], d["y"], d["day"], d["eligible"], n_jobs=4)
    r1 = run_m5(bank, d["eligible"], T=d["T"], smooth_reg=0.1, context=d["context"], day=d["day"], seed=7)
    r2 = run_m5(bank, d["eligible"], T=d["T"], smooth_reg=0.1, context=d["context"], day=d["day"], seed=7)
    mx = max(float(np.abs(a["y_pred"] - b["y_pred"]).max()) for a, b in zip(r1, r2))
    check("reproducible under fixed seed", mx < 1e-6, f"max |Δpred| = {mx:.2e}")


def test_calibration_metric():
    """ECE is 0 for a perfectly calibrated constant predictor and > 0 for a
    miscalibrated one."""
    from metrics import expected_calibration_error
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)
    good = expected_calibration_error(y, p)
    bad = expected_calibration_error(y, np.clip(p + 0.2, 0, 1))
    check("ECE ~0 for calibrated predictor", good < 0.02, f"ECE={good:.3f}")
    check("ECE detects 0.2 over-confidence shift", bad > 0.1, f"ECE={bad:.3f}")


def main():
    print("Building small synthetic candidate bank for pipeline tests ...")
    d = _small_bank()
    test_no_future_leakage(d)
    test_gate_range(d)
    test_mixture_identity(d)
    test_reproducibility(d)
    test_local_shift_isolation()
    test_opposing_local_isolation()
    test_calibration_metric()
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILURE(S): {FAILS}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
