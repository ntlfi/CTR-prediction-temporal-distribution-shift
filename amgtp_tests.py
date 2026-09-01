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


def test_opposing_recurring_isolation():
    """Under S7, both subpopulations oscillate on the w0<->w1 axis but a
    quarter period out of phase, so their per-block CTR series are close to
    uncorrelated -- unlike plain `recurring`, where every row shares one
    schedule and the two group series are identical."""
    from synthetic_data import generate_synthetic_raw
    n, period = 120, 14
    df, _ = generate_synthetic_raw(n_days=n, rows_per_day=4000, drift_mode="opposing_recurring",
                                   period_days=period, seed=0)
    a = df[df["group"]].groupby("day")["click"].mean().to_numpy()
    b = df[~df["group"]].groupby("day")["click"].mean().to_numpy()
    check("S7: group A CTR oscillates", a.max() - a.min() > 0.03, f"A range {a.max() - a.min():.3f}")
    check("S7: group B CTR oscillates", b.max() - b.min() > 0.03, f"B range {b.max() - b.min():.3f}")
    corr = float(np.corrcoef(a, b)[0, 1])
    # quarter-period phase offset -> the two group CTR series are decorrelated
    # (here mildly anti-correlated); plain `recurring` gives corr ~ +0.8.
    check("S7: group A/B schedules are out of phase (corr < 0.4)", corr < 0.4,
          f"corr(A CTR, B CTR) = {corr:.2f}")
    peak_a, peak_b = int(a[:period].argmax()), int(b[:period].argmax())
    check("S7: group A and B peak on different blocks within a period",
          abs(peak_a - peak_b) >= 2, f"A peak block {peak_a}, B peak block {peak_b}")

    df_rec, _ = generate_synthetic_raw(n_days=n, rows_per_day=4000, drift_mode="recurring",
                                       period_days=period, seed=0)
    ar = df_rec[df_rec["group"]].groupby("day")["click"].mean().to_numpy()
    br = df_rec[~df_rec["group"]].groupby("day")["click"].mean().to_numpy()
    check("S7 sanity: plain recurring keeps the two groups in phase",
          np.corrcoef(ar, br)[0, 1] > 0.7, f"corr = {np.corrcoef(ar, br)[0, 1]:.2f}")


def test_opposing_recurring_rng_untouched():
    """Adding S7 must not perturb the RNG stream of the original modes: the
    first-block clicks for `recurring` are byte-for-byte what they were."""
    from synthetic_data import generate_synthetic_raw
    df, _ = generate_synthetic_raw(n_days=8, rows_per_day=500, drift_mode="recurring",
                                   period_days=14, seed=3)
    # regression fingerprint: total clicks on block 0 and block 7 at this config
    b0 = int(df[df["day"] == 0]["click"].sum())
    b7 = int(df[df["day"] == 7]["click"].sum())
    check("recurring RNG stream unchanged (block-0 / block-7 click totals)",
          (b0, b7) == (125, 124), f"got ({b0}, {b7}), expected (125, 124)")


def test_amgtp_hidden_persistence(d):
    """Extension A -- the hidden-layer PersistenceNet: hidden=0 is
    deterministic and matches the default path; hidden>0 still starts at
    sigmoid(init_bias) (no imposed inertia at deploy) and keeps beta in
    [0,1] with finite predictions."""
    import math
    from amgtp_method import run_amgtp
    from candidate_bank import build_candidate_bank
    bank = build_candidate_bank(d["X"], d["y"], d["day"], d["eligible"], n_jobs=4)

    r_def = run_amgtp(bank, d["eligible"], T=d["T"], context=d["context"], day=d["day"], seed=0)
    r_h0 = run_amgtp(bank, d["eligible"], T=d["T"], context=d["context"], day=d["day"], seed=0,
                     persist_hidden=0)
    mx = max(float(np.abs(a["y_pred"] - b["y_pred"]).max()) for a, b in zip(r_def, r_h0))
    check("PersistenceNet: persist_hidden=0 == default linear path", mx == 0.0,
          f"max |Δpred| = {mx:.2e}")

    r_h8 = run_amgtp(bank, d["eligible"], T=d["T"], context=d["context"], day=d["day"], seed=0,
                     persist_hidden=8)
    b0 = r_h8[0]["beta"]
    check("PersistenceNet(hidden=8): day-0 beta = sigmoid(init_bias)",
          abs(b0 - 1.0 / (1.0 + math.exp(1.0))) < 1e-6, f"day-0 beta = {b0:.5f}")
    betas = np.array([r["beta"] for r in r_h8])
    check("PersistenceNet(hidden=8): beta_t in [0,1]", betas.min() >= 0.0 and betas.max() <= 1.0,
          f"range [{betas.min():.3f}, {betas.max():.3f}]")
    finite = all(np.isfinite(r["y_pred"]).all() for r in r_h8)
    check("PersistenceNet(hidden=8): predictions finite", finite)

    r_h8b = run_amgtp(bank, d["eligible"], T=d["T"], context=d["context"], day=d["day"], seed=0,
                      persist_hidden=8)
    mx8 = max(float(np.abs(a["y_pred"] - b["y_pred"]).max()) for a, b in zip(r_h8, r_h8b))
    check("PersistenceNet(hidden=8): reproducible under fixed seed", mx8 < 1e-6,
          f"max |Δpred| = {mx8:.2e}")


def test_amgtp_beta_per_example(d):
    """Extension B -- per-example beta_t(x): g_xi is zero-init so day 0 is
    bit-identical to the global model; a huge Var_x[beta] penalty collapses
    it back to the global beta_t (the A12 identity check); beta_t(x) stays
    in [0,1]; per-group beta is recorded when `group` is passed."""
    from amgtp_method import run_amgtp
    from candidate_bank import build_candidate_bank
    bank = build_candidate_bank(d["X"], d["y"], d["day"], d["eligible"], n_jobs=4)

    g = run_amgtp(bank, d["eligible"], T=d["T"], context=d["context"], day=d["day"], seed=0)
    bx = run_amgtp(bank, d["eligible"], T=d["T"], context=d["context"], day=d["day"], seed=0,
                   beta_per_example=True, beta_var_reg=1e-3, group=d["group"])

    d0 = float(np.abs(g[0]["y_pred"] - bx[0]["y_pred"]).max())
    check("beta_t(x): g_xi zero-init -> day-0 == global AMG-TP", d0 == 0.0, f"max |Δpred| day 0 = {d0:.2e}")

    betas = np.array([r["beta"] for r in bx])
    check("beta_t(x): mean beta_t in [0,1]", betas.min() >= 0.0 and betas.max() <= 1.0,
          f"range [{betas.min():.3f}, {betas.max():.3f}]")
    check("beta_t(x): predictions finite", all(np.isfinite(r["y_pred"]).all() for r in bx))
    check("beta_t(x): per-group beta recorded", "beta_A" in bx[0] and "beta_B" in bx[0])

    bx_var0 = run_amgtp(bank, d["eligible"], T=d["T"], context=d["context"], day=d["day"], seed=0,
                        beta_per_example=True, beta_var_reg=0.0, group=d["group"])
    bx_hi = run_amgtp(bank, d["eligible"], T=d["T"], context=d["context"], day=d["day"], seed=0,
                      beta_per_example=True, beta_var_reg=1.0, group=d["group"])
    spread0 = float(np.mean([r["beta_std"] for r in bx_var0]))
    spread_hi = float(np.mean([r["beta_std"] for r in bx_hi]))
    check("beta_t(x): Var_x[beta] penalty shrinks the per-example spread (A11 vs A12)",
          spread_hi < 0.5 * spread0, f"mean beta_std: no penalty {spread0:.3f} -> lambda=1 {spread_hi:.3f}")

    bxb = run_amgtp(bank, d["eligible"], T=d["T"], context=d["context"], day=d["day"], seed=0,
                    beta_per_example=True, beta_var_reg=1e-3, group=d["group"])
    rep = max(float(np.abs(a["y_pred"] - b["y_pred"]).max()) for a, b in zip(bx, bxb))
    check("beta_t(x): reproducible under fixed seed", rep < 1e-6, f"max |Δpred| = {rep:.2e}")


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
    test_amgtp_hidden_persistence(d)
    test_amgtp_beta_per_example(d)
    test_local_shift_isolation()
    test_opposing_local_isolation()
    test_opposing_recurring_isolation()
    test_opposing_recurring_rng_untouched()
    test_calibration_metric()
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILURE(S): {FAILS}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
