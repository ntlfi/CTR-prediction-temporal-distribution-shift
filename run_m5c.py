"""Run M5c (M5b + periodicity phase features: `deployed`, a causally
-detected period, and -- on synthetic recurring drift only, where the true
period is known -- an `oracle` diagnostic upper bound) and merge into the
existing M1/M2/M5b/ensemble comparison. Run run_new_methods.py first in the
same --out dir for the merge to include the full P0/P1/P2/M1/M2/M5b/ensemble
ladder; this script still runs standalone otherwise.

Motivation (results/m5_analysis.md, "what the ensemble does not do"): the
M2+M5b ensemble is real insurance against not knowing the drift regime in
advance, but it inherits both specialists' shared blind spot -- neither M2
nor M5b carries any explicit periodicity signal, only "how much recent
history to trust". This tests whether an explicit "where in the cycle are
we" phase feature (periodicity.py) actually closes that gap, and how much
of the gap is closable at all (the oracle variant) versus how much a
realistic causal detector actually recovers (the deployed variant).

Example:
    python run_new_methods.py --source synthetic --synthetic-days 120 \\
        --synthetic-drift recurring --synthetic-period-days 14 --out results_synthetic_recurring
    python run_m5c.py --source synthetic --synthetic-days 120 \\
        --synthetic-drift recurring --synthetic-period-days 14 --out results_synthetic_recurring
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from candidate_bank import build_candidate_bank
from data_source import add_data_source_args, load_data_with_context
from han_arw import per_sample_log_loss
from m5c_periodic_gate import run_m5c
from periodicity import causal_period_series
from run_advanced import rows_from_list
from run_baselines import aggregate_table
from short_long import LONG_NAME
from splits import compute_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_source_args(parser)
    parser.add_argument("--n-features", type=int, default=2**18)
    parser.add_argument("--warmup-days", type=int, default=3)
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=1e-4, help="L2 reg for the candidate-bank base models.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--m5c-lr", type=float, default=0.05)
    parser.add_argument("--m5c-l2", type=float, default=1e-3)
    parser.add_argument("--m5c-entropy-reg", type=float, default=1e-3)
    parser.add_argument("--m5c-smooth-reg", type=float, default=1e-3)
    parser.add_argument("--m5c-epochs-per-day", type=int, default=3)
    parser.add_argument("--min-period", type=int, default=3)
    parser.add_argument("--max-period", type=int, default=25)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, day, group, context = load_data_with_context(args, args.n_features, args.seed)
    print(f"Loaded {X.shape[0]} rows, {day.max() + 1} days, click rate {y.mean():.3f}")

    eligible_days, dev_days, test_days = compute_splits(day, args.warmup_days, args.test_frac)
    print(f"Dev days: {sorted(dev_days)}")
    print(f"Test days (locked): {sorted(test_days)}")

    print("Building window-family candidate bank (short=rolling_3, long=expanding) ...")
    t0 = time.time()
    bank = build_candidate_bank(X, y, day, eligible_days, alpha=args.alpha, seed=args.seed, n_jobs=args.n_jobs)
    print(f"  done in {time.time() - t0:.1f}s")

    # Causal period detection: per-day mean log loss of the `expanding`
    # candidate (has a prediction for every eligible day, the longest
    # available signal) -- estimating day t's period only ever from days < t.
    expanding_loss_by_day = {
        t: float(per_sample_log_loss(bank[LONG_NAME][t]["y_true"], bank[LONG_NAME][t]["y_pred"]).mean())
        for t in bank[LONG_NAME]
    }
    period_by_day = causal_period_series(expanding_loss_by_day, list(bank[LONG_NAME].keys()),
                                          min_period=args.min_period, max_period=args.max_period)
    detected = [(t, p) for t, p in sorted(period_by_day.items()) if p is not None]
    print(f"Causal period detection: {len(detected)}/{len(period_by_day)} days had a detected period"
          + (f", most recent estimate {detected[-1][1]} days (day {detected[-1][0]})" if detected else ""))

    print("Running M5c (M5b + causally-detected periodicity features) ...")
    t0 = time.time()
    m5c_rows = run_m5c(bank, eligible_days, T=int(day.max()), period_by_day=period_by_day,
                        lr=args.m5c_lr, l2=args.m5c_l2, entropy_reg=args.m5c_entropy_reg,
                        smooth_reg=args.m5c_smooth_reg, epochs_per_day=args.m5c_epochs_per_day,
                        seed=args.seed, context=context, day=day)
    print(f"  done in {time.time() - t0:.1f}s ({len(m5c_rows)} prediction days)")

    all_new_rows = rows_from_list(m5c_rows, "m5c_periodic")
    oracle_rows = None
    if args.source == "synthetic" and args.synthetic_drift == "recurring":
        oracle_period_by_day = {t: args.synthetic_period_days for t in period_by_day}
        print(f"Running M5c-oracle (M5b + true period={args.synthetic_period_days}d phase features, "
              "diagnostic only, never a deployed prediction) ...")
        t0 = time.time()
        oracle_rows = run_m5c(bank, eligible_days, T=int(day.max()), period_by_day=oracle_period_by_day,
                               lr=args.m5c_lr, l2=args.m5c_l2, entropy_reg=args.m5c_entropy_reg,
                               smooth_reg=args.m5c_smooth_reg, epochs_per_day=args.m5c_epochs_per_day,
                               seed=args.seed, context=context, day=day)
        print(f"  done in {time.time() - t0:.1f}s ({len(oracle_rows)} prediction days)")
        all_new_rows += rows_from_list(oracle_rows, "m5c_periodic_oracle")

    new_per_day = pd.DataFrame(all_new_rows)
    new_per_day.to_csv(out_dir / "m5c_per_day_metrics.csv", index=False)
    pd.DataFrame([{"day": r["day"], "period": r["period"]} for r in m5c_rows]) \
        .to_csv(out_dir / "m5c_detected_period.csv", index=False)

    # Merge with whatever M1/M2/M5b/ensemble (+ P0/P1/P2/SFTL) comparison already exists.
    existing_candidates = [
        ("all_methods_with_new_methods_per_day_metrics.csv", "all_methods_with_new_methods_comparison_table.csv"),
        ("all_methods_with_sftl_per_day_metrics.csv", "all_methods_with_sftl_comparison_table.csv"),
        ("all_methods_per_day_metrics.csv", "all_methods_comparison_table.csv"),
        ("per_day_metrics.csv", "comparison_table.csv"),
    ]
    existing_per_day, existing_table = None, None
    for per_day_name, table_name in existing_candidates:
        if (out_dir / per_day_name).exists() and (out_dir / table_name).exists():
            existing_per_day = pd.read_csv(out_dir / per_day_name)
            existing_table = pd.read_csv(out_dir / table_name)
            print(f"Merging with existing {per_day_name} / {table_name}")
            break
    if existing_per_day is None:
        print("Warning: no existing results found in out dir -- run run_new_methods.py first for full context.")
        existing_per_day = pd.DataFrame(columns=new_per_day.columns)
        existing_table = pd.DataFrame(columns=["method", "log_loss", "brier", "pr_auc",
                                                "mean_train_rows", "mean_fit_time_s", "n_test_days"])

    all_per_day = pd.concat([existing_per_day, new_per_day], ignore_index=True)
    all_per_day.to_csv(out_dir / "all_methods_with_m5c_per_day_metrics.csv", index=False)

    new_test = new_per_day[new_per_day["day"].isin(test_days)]
    new_table = aggregate_table(new_test) if len(new_test) else pd.DataFrame(columns=existing_table.columns)
    table = pd.concat([existing_table, new_table], ignore_index=True).sort_values("log_loss").reset_index(drop=True)
    table.to_csv(out_dir / "all_methods_with_m5c_comparison_table.csv", index=False)
    print("\n=== Locked-test comparison table (all methods + M5c) ===")
    print(table.to_string(index=False))

    days_list = [r["day"] for r in m5c_rows]
    periods = [r["period"] if r["period"] is not None else np.nan for r in m5c_rows]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(days_list, periods, marker="o", markersize=3, label="causally detected period")
    if args.source == "synthetic" and args.synthetic_drift == "recurring":
        ax.axhline(args.synthetic_period_days, color="red", linestyle="--", linewidth=1, label="true period")
    ax.set_xlabel("prediction day")
    ax.set_ylabel("detected period (days)")
    ax.set_title("M5c: causally detected period over time")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "m5c_detected_period.png", dpi=150)
    plt.close(fig)

    findings = [
        "M5c is M5b (context-dependent gating over the full WINDOW_FAMILY, see m5_multiscale_gate.py) "
        "plus sin/cos periodicity phase features (periodicity.py), testing whether an explicit "
        "'where in the cycle are we' signal closes the recurring-drift gap shared by every other "
        "method in this project.",
    ]
    m5c_test = new_per_day[(new_per_day["method"] == "m5c_periodic") & (new_per_day["day"].isin(test_days))]
    if len(m5c_test):
        findings.append(f"M5c (deployed, causally-detected period) locked-test log loss "
                         f"{np.average(m5c_test['log_loss'], weights=m5c_test['n']):.4f}; "
                         f"{len(detected)}/{len(period_by_day)} eligible days had a detected period.")
    if oracle_rows is not None:
        oracle_test = new_per_day[(new_per_day["method"] == "m5c_periodic_oracle")
                                   & (new_per_day["day"].isin(test_days))]
        if len(oracle_test):
            findings.append(f"M5c-oracle (diagnostic, true period={args.synthetic_period_days}d, "
                             f"never a deployed prediction) locked-test log loss "
                             f"{np.average(oracle_test['log_loss'], weights=oracle_test['n']):.4f} -- "
                             "quantifies the ceiling if period detection were perfect.")
    best = table.iloc[0]
    findings.append(f"Best method overall on locked test (including M5c): "
                     f"{best['method']} (log loss {best['log_loss']:.4f}).")

    findings_text = "\n".join(f"- {line}" for line in findings)
    (out_dir / "m5c_findings.md").write_text("# M5c (periodicity-aware M5b) findings\n\n" + findings_text + "\n")
    print("\n=== Findings ===")
    print(findings_text)
    print(f"\nAll outputs written to {out_dir}/")


if __name__ == "__main__":
    main()
