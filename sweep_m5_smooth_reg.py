"""Ablate M5b's day-level smoothness regularizer (`smooth_reg`) on recurring
drift, to test the hypothesis (results/m5_analysis.md's M5c section) that
M5b's recurring-drift weakness relative to M2 comes from this term actively
resisting the fast expert-weight swings a ~14-day cycle demands across 5
experts -- something M2's scalar 2-expert gate has no analogous term
fighting. M5c (explicit periodicity features) already ruled out "missing
phase information" as the cause -- even the oracle period didn't help --
so this tests "gate dynamics fight the cycle" instead, before touching
architecture.

Builds the candidate bank once, then sweeps M5b's smooth_reg through several
values (M1/M2/han_arw etc. are unaffected by this and are not re-run --
compare against the frozen baselines already in --out from
run_new_methods.py, printed for reference if present).

Example:
    python sweep_m5_smooth_reg.py --source synthetic --synthetic-days 120 \\
        --synthetic-drift recurring --synthetic-period-days 14 --out results_synthetic_recurring
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from candidate_bank import build_candidate_bank
from data_source import add_data_source_args, load_data_with_context
from m5_multiscale_gate import run_m5
from run_advanced import rows_from_list
from run_baselines import aggregate_table
from splits import compute_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_source_args(parser)
    parser.add_argument("--n-features", type=int, default=2**18)
    parser.add_argument("--warmup-days", type=int, default=3)
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--m5-lr", type=float, default=0.05)
    parser.add_argument("--m5-l2", type=float, default=1e-3)
    parser.add_argument("--m5-entropy-reg", type=float, default=1e-3)
    parser.add_argument("--m5-epochs-per-day", type=int, default=3)
    parser.add_argument("--smooth-reg-grid", type=float, nargs="+",
                         default=[0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, day, group, context = load_data_with_context(args, args.n_features, args.seed)
    print(f"Loaded {X.shape[0]} rows, {day.max() + 1} days, click rate {y.mean():.3f}")

    eligible_days, dev_days, test_days = compute_splits(day, args.warmup_days, args.test_frac)

    print("Building window-family candidate bank (short=rolling_3, long=expanding) ...")
    t0 = time.time()
    bank = build_candidate_bank(X, y, day, eligible_days, alpha=args.alpha, seed=args.seed, n_jobs=args.n_jobs)
    print(f"  done in {time.time() - t0:.1f}s")

    rows_all = []
    for smooth_reg in args.smooth_reg_grid:
        t0 = time.time()
        m5_rows = run_m5(bank, eligible_days, T=int(day.max()), lr=args.m5_lr, l2=args.m5_l2,
                          entropy_reg=args.m5_entropy_reg, smooth_reg=smooth_reg,
                          epochs_per_day=args.m5_epochs_per_day, seed=args.seed,
                          context=context, day=day)
        method_name = f"m5b_smooth{smooth_reg:g}"
        rows_all += rows_from_list(m5_rows, method_name)
        print(f"  smooth_reg={smooth_reg:g} done in {time.time() - t0:.1f}s")

    per_day = pd.DataFrame(rows_all)
    per_day.to_csv(out_dir / "m5_smooth_reg_sweep_per_day_metrics.csv", index=False)
    test_per_day = per_day[per_day["day"].isin(test_days)]
    table = aggregate_table(test_per_day)
    table.to_csv(out_dir / "m5_smooth_reg_sweep_comparison_table.csv", index=False)
    print("\n=== M5b smooth_reg sweep: locked-test comparison ===")
    print(table.to_string(index=False))

    # For reference, print the frozen M2/M5b(default)/han_arw/ensemble baselines
    # already in this dir (from run_new_methods.py), if present.
    for per_day_name in ["all_methods_with_new_methods_per_day_metrics.csv",
                          "all_methods_with_sftl_per_day_metrics.csv",
                          "all_methods_per_day_metrics.csv"]:
        p = out_dir / per_day_name
        if p.exists():
            existing = pd.read_csv(p)
            existing_test = existing[existing["day"].isin(test_days)]
            baselines = existing_test[existing_test["method"].isin(
                ["m2_context_gate", "m5_multiscale_gate", "han_arw", "m2_m5_ensemble", "expanding"])]
            if len(baselines):
                baseline_table = aggregate_table(baselines)
                print("\n=== For reference: existing baselines already in this dir ===")
                print(baseline_table.to_string(index=False))
            break

    print(f"\nAll outputs written to {out_dir}/")


if __name__ == "__main__":
    main()
