"""Run the 3-way ensemble (M2 / M5b-default / M5b-high-smooth, ensemble3.py)
and merge into the existing comparison. Run run_new_methods.py first in the
same --out dir for the merge to include the full P0/P1/P2/M1/M2/M5b/ensemble
ladder; this script still runs standalone otherwise.

Motivation (results/m5_analysis.md's smooth_reg sweep section): raising
M5b's smooth_reg to 0.1 makes it beat every method tried on recurring drift
so far (0.4118 vs M2's 0.4180), but at a real cost on abrupt (+7.1%) and
local (+3.7%) drift relative to M5b-default -- a regime-dependent tradeoff,
not a free win. This tests whether a 3-way meta-gate over M2 / M5b-default
/ M5b-high-smooth's predictions can get M5b-high-smooth's recurring win
*and* keep M5b-default's abrupt/local wins, the same way the 2-way
M2+M5b ensemble already gets the best of M2 and M5b without knowing the
regime in advance.

Example:
    python run_new_methods.py --source synthetic --synthetic-days 120 \\
        --synthetic-drift recurring --synthetic-period-days 14 --out results_synthetic_recurring
    python run_ensemble3.py --source synthetic --synthetic-days 120 \\
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
from ensemble3 import EXPERTS, run_ensemble3
from m2_context_gate import run_m2
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
    parser.add_argument("--alpha", type=float, default=1e-4, help="L2 reg for the candidate-bank base models.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--m2-lr", type=float, default=0.05)
    parser.add_argument("--m2-l2", type=float, default=1e-3)
    parser.add_argument("--m2-entropy-reg", type=float, default=1e-3)
    parser.add_argument("--m2-smooth-reg", type=float, default=1e-3)
    parser.add_argument("--m2-epochs-per-day", type=int, default=3)
    parser.add_argument("--m5b-hs-smooth-reg", type=float, default=0.1,
                         help="smooth_reg for the M5b-high-smooth expert (best value found by sweep_m5_smooth_reg.py).")
    parser.add_argument("--ens3-lr", type=float, default=0.05)
    parser.add_argument("--ens3-l2", type=float, default=1e-3)
    parser.add_argument("--ens3-entropy-reg", type=float, default=1e-3)
    parser.add_argument("--ens3-smooth-reg", type=float, default=1e-3)
    parser.add_argument("--ens3-epochs-per-day", type=int, default=3)
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

    print("Running M2 (context-dependent gating) ...")
    t0 = time.time()
    m2_rows = run_m2(bank, eligible_days, T=int(day.max()), lr=args.m2_lr, l2=args.m2_l2,
                      entropy_reg=args.m2_entropy_reg, smooth_reg=args.m2_smooth_reg,
                      epochs_per_day=args.m2_epochs_per_day, seed=args.seed, context=context, day=day)
    print(f"  done in {time.time() - t0:.1f}s ({len(m2_rows)} prediction days)")

    print("Running M5b-default (smooth_reg=1e-3) ...")
    t0 = time.time()
    m5b_rows = run_m5(bank, eligible_days, T=int(day.max()), seed=args.seed, context=context, day=day)
    print(f"  done in {time.time() - t0:.1f}s ({len(m5b_rows)} prediction days)")

    print(f"Running M5b-high-smooth (smooth_reg={args.m5b_hs_smooth_reg:g}) ...")
    t0 = time.time()
    m5hs_rows = run_m5(bank, eligible_days, T=int(day.max()), smooth_reg=args.m5b_hs_smooth_reg,
                        seed=args.seed, context=context, day=day)
    print(f"  done in {time.time() - t0:.1f}s ({len(m5hs_rows)} prediction days)")

    print("Running 3-way ensemble (M2 / M5b-default / M5b-high-smooth) ...")
    t0 = time.time()
    ens3_rows = run_ensemble3(m2_rows, m5b_rows, m5hs_rows, T=int(day.max()), lr=args.ens3_lr, l2=args.ens3_l2,
                               entropy_reg=args.ens3_entropy_reg, smooth_reg=args.ens3_smooth_reg,
                               epochs_per_day=args.ens3_epochs_per_day, seed=args.seed, context=context, day=day)
    print(f"  done in {time.time() - t0:.1f}s ({len(ens3_rows)} prediction days)")

    new_per_day = pd.DataFrame(rows_from_list(m5hs_rows, "m5b_high_smooth")
                                + rows_from_list(ens3_rows, "ensemble3"))
    new_per_day.to_csv(out_dir / "ensemble3_per_day_metrics.csv", index=False)
    pd.DataFrame([{"day": r["day"], **r["mean_weights"]} for r in ens3_rows]) \
        .to_csv(out_dir / "ensemble3_weights.csv", index=False)

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
    all_per_day.to_csv(out_dir / "all_methods_with_ensemble3_per_day_metrics.csv", index=False)

    new_test = new_per_day[new_per_day["day"].isin(test_days)]
    new_table = aggregate_table(new_test) if len(new_test) else pd.DataFrame(columns=existing_table.columns)
    table = pd.concat([existing_table, new_table], ignore_index=True).sort_values("log_loss").reset_index(drop=True)
    table.to_csv(out_dir / "all_methods_with_ensemble3_comparison_table.csv", index=False)
    print("\n=== Locked-test comparison table (all methods + ensemble3) ===")
    print(table.to_string(index=False))

    days_list = [r["day"] for r in ens3_rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in EXPERTS:
        ax.plot(days_list, [r["mean_weights"][name] for r in ens3_rows], marker="o", markersize=3, label=name)
    ax.set_xlabel("prediction day")
    ax.set_ylabel("mean gate weight pi(x)")
    ax.set_title("3-way ensemble: mean per-expert gate weight over time (M2 / M5b / M5b-high-smooth)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "ensemble3_diagnostics.png", dpi=150)
    plt.close(fig)

    findings = [
        "ensemble3 is a 3-way meta-gate (ensemble3.py) blending M2, M5b-default (smooth_reg=1e-3), and "
        f"M5b-high-smooth (smooth_reg={args.m5b_hs_smooth_reg:g}) -- built after the smooth_reg sweep found "
        "M5b-high-smooth beats every method on recurring drift but regresses on abrupt/local.",
    ]
    ens3_test = new_per_day[(new_per_day["method"] == "ensemble3") & (new_per_day["day"].isin(test_days))]
    if len(ens3_test):
        final_mean_weights = ens3_rows[-1]["mean_weights"]
        top_expert = max(final_mean_weights, key=final_mean_weights.get)
        findings.append(f"ensemble3 locked-test log loss "
                         f"{np.average(ens3_test['log_loss'], weights=ens3_test['n']):.4f}; "
                         f"final-day mean weights {[(k, round(v, 2)) for k, v in final_mean_weights.items()]}, "
                         f"top expert {top_expert}.")
    m5hs_test = new_per_day[(new_per_day["method"] == "m5b_high_smooth") & (new_per_day["day"].isin(test_days))]
    if len(m5hs_test):
        findings.append(f"M5b-high-smooth (standalone) locked-test log loss "
                         f"{np.average(m5hs_test['log_loss'], weights=m5hs_test['n']):.4f}.")
    best = table.iloc[0]
    findings.append(f"Best method overall on locked test (including ensemble3): "
                     f"{best['method']} (log loss {best['log_loss']:.4f}).")

    findings_text = "\n".join(f"- {line}" for line in findings)
    (out_dir / "ensemble3_findings.md").write_text("# 3-way ensemble (M2/M5b/M5b-high-smooth) findings\n\n"
                                                     + findings_text + "\n")
    print("\n=== Findings ===")
    print(findings_text)
    print(f"\nAll outputs written to {out_dir}/")


if __name__ == "__main__":
    main()
