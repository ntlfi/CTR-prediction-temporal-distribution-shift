"""Run the P1 (Han ARW, Differentiable Forgetting) and P2 (AdaMoE) methods
on the Criteo dataset and merge them into the P0 comparison (PDF section
2, 5, 9). Run run_baselines.py first -- this script reuses its saved
results/per_day_metrics.csv and results/comparison_table.csv.

Example:
    python run_advanced.py --sample-frac 0.2   # quick preliminary pass
    python run_advanced.py                     # full dataset
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from adamoe import run_adamoe
from baselines import WINDOW_FAMILY
from candidate_bank import build_candidate_bank
import diff_forgetting
from data_source import add_data_source_args, load_data
from han_arw import run_han_arw
from metrics import day_metrics
from run_baselines import aggregate_table
from splits import compute_splits

P1_P2_METHODS = ["han_arw", "diff_forgetting", "adamoe"]


def run_diff_forgetting_all_days(X, y, day, days, val_window, maxiter, alpha, seed, n_jobs):
    def _one(t):
        return t, diff_forgetting.fit_predict(
            X, y, day, t, val_window=val_window, alpha=alpha, seed=seed, maxiter=maxiter)
    results = Parallel(n_jobs=n_jobs)(delayed(_one)(t) for t in days)
    return {t: r for t, r in results if r is not None}


def rows_from_list(results_list, method_name):
    rows = []
    for r in results_list:
        m = day_metrics(r["y_true"], r["y_pred"])
        rows.append({"method": method_name, "day": r["day"], **m,
                      "n_train": r["n_train"], "fit_time": r["fit_time"]})
    return rows


def rows_from_dict(results_dict, method_name):
    rows = []
    for t, r in sorted(results_dict.items()):
        m = day_metrics(r["y_true"], r["y_pred"])
        rows.append({"method": method_name, "day": t, **m,
                      "n_train": r["n_train"], "fit_time": r["fit_time"]})
    return rows


def plot_combined_per_day_logloss(per_day: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for method, g in per_day.groupby("method"):
        g = g.sort_values("day")
        lw = 2.5 if method in P1_P2_METHODS else 1.2
        ax.plot(g["day"], g["log_loss"], marker="o", markersize=3, linewidth=lw, label=method)
    ax.set_xlabel("prediction day")
    ax.set_ylabel("log loss")
    ax.set_title("Per-day log loss, all methods (P0 + P1 + P2)")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_memory_behavior(han_rows, df_results, out_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5))

    ax = axes[0]
    categories = WINDOW_FAMILY
    days = [r["day"] for r in han_rows]
    ys = [categories.index(r["selected_window"]) for r in han_rows]
    ax.scatter(days, ys, marker="s")
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xlabel("prediction day")
    ax.set_title("Han ARW: selected effective window per day")

    ax2 = axes[1]
    days2 = sorted(df_results.keys())
    half_lives = [min(df_results[t]["half_life"], 60.0) for t in days2]  # cap inf for plotting
    ax2.plot(days2, half_lives, marker="o")
    ax2.set_xlabel("prediction day")
    ax2.set_ylabel("learned half-life (days, capped at 60)")
    ax2.set_title("Differentiable Forgetting: learned half-life per day")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_source_args(parser)
    parser.add_argument("--n-features", type=int, default=2**18)
    parser.add_argument("--warmup-days", type=int, default=3)
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=1e-4, help="L2 regularization strength.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=4, help="Parallel workers for candidate-bank fits and Differentiable Forgetting.")
    parser.add_argument("--diff-forgetting-val-window", type=int, default=3)
    parser.add_argument("--diff-forgetting-maxiter", type=int, default=12, help="Max Brent iterations for the outer eta search.")
    parser.add_argument("--han-arw-delta", type=float, default=0.1, help="Significance parameter (paper's own experiments fix this at 0.1).")
    parser.add_argument("--han-arw-min-history", type=int, default=3)
    parser.add_argument("--adamoe-lambda", type=float, default=0.5, help="EMA momentum for AdaMoE's expert weights.")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, day = load_data(args, args.n_features, args.seed)
    print(f"Loaded {X.shape[0]} rows, {day.max() + 1} days, click rate {y.mean():.3f}")

    eligible_days, dev_days, test_days = compute_splits(day, args.warmup_days, args.test_frac)
    print(f"Dev days: {sorted(dev_days)}")
    print(f"Test days (locked): {sorted(test_days)}")

    print("Building window-family candidate bank (shared by Han ARW & AdaMoE) ...")
    t0 = time.time()
    bank = build_candidate_bank(X, y, day, eligible_days, alpha=args.alpha, seed=args.seed, n_jobs=args.n_jobs)
    print(f"  done in {time.time() - t0:.1f}s")

    print("Running Han ARW tournament ...")
    t0 = time.time()
    han_rows = run_han_arw(bank, eligible_days, dev_days=dev_days,
                            min_history=args.han_arw_min_history, delta=args.han_arw_delta)
    print(f"  done in {time.time() - t0:.1f}s ({len(han_rows)} prediction days)")

    print("Running AdaMoE ...")
    t0 = time.time()
    adamoe_rows = run_adamoe(bank, eligible_days, lam=args.adamoe_lambda)
    print(f"  done in {time.time() - t0:.1f}s ({len(adamoe_rows)} prediction days)")

    print(f"Running Differentiable Forgetting ({args.n_jobs} parallel workers) ...")
    t0 = time.time()
    df_results = run_diff_forgetting_all_days(
        X, y, day, eligible_days, args.diff_forgetting_val_window,
        args.diff_forgetting_maxiter, args.alpha, args.seed, args.n_jobs)
    print(f"  done in {time.time() - t0:.1f}s ({len(df_results)} prediction days)")

    p1_p2_per_day = pd.DataFrame(
        rows_from_list(han_rows, "han_arw")
        + rows_from_list(adamoe_rows, "adamoe")
        + rows_from_dict(df_results, "diff_forgetting")
    )
    p1_p2_per_day.to_csv(out_dir / "p1_p2_per_day_metrics.csv", index=False)

    pd.DataFrame([{"day": r["day"], "selected_window": r["selected_window"]} for r in han_rows]) \
        .to_csv(out_dir / "han_arw_selected_window.csv", index=False)
    pd.DataFrame([{"day": t, "eta": r["eta"], "half_life": r["half_life"]}
                  for t, r in sorted(df_results.items())]).to_csv(out_dir / "diff_forgetting_eta.csv", index=False)
    pd.DataFrame([{"day": r["day"], **r["weights"]} for r in adamoe_rows]) \
        .to_csv(out_dir / "adamoe_expert_weights.csv", index=False)

    # Merge with the P0 run (must be run first via run_baselines.py).
    p0_per_day_path, p0_table_path = out_dir / "per_day_metrics.csv", out_dir / "comparison_table.csv"
    if p0_per_day_path.exists() and p0_table_path.exists():
        p0_per_day = pd.read_csv(p0_per_day_path)
        p0_table = pd.read_csv(p0_table_path)
    else:
        print("Warning: P0 results not found in results/ -- run run_baselines.py first. "
              "Comparison table will only include P1/P2.")
        p0_per_day = pd.DataFrame(columns=p1_p2_per_day.columns)
        p0_table = pd.DataFrame(columns=["method", "log_loss", "brier", "pr_auc",
                                          "mean_train_rows", "mean_fit_time_s", "n_test_days"])

    all_per_day = pd.concat([p0_per_day, p1_p2_per_day], ignore_index=True)
    all_per_day.to_csv(out_dir / "all_methods_per_day_metrics.csv", index=False)

    p1_p2_test = p1_p2_per_day[p1_p2_per_day["day"].isin(test_days)]
    p1_p2_table = aggregate_table(p1_p2_test) if len(p1_p2_test) else pd.DataFrame(columns=p0_table.columns)
    table = pd.concat([p0_table, p1_p2_table], ignore_index=True).sort_values("log_loss").reset_index(drop=True)
    table.to_csv(out_dir / "all_methods_comparison_table.csv", index=False)
    print("\n=== Locked-test comparison table (all methods) ===")
    print(table.to_string(index=False))

    test_per_day = all_per_day[all_per_day["day"].isin(test_days)]
    plot_combined_per_day_logloss(test_per_day, out_dir / "all_methods_per_day_logloss.png")
    plot_memory_behavior(han_rows, df_results, out_dir / "advanced_memory_behavior.png")

    best = table.iloc[0]
    p1_p2_mask = table["method"].isin(P1_P2_METHODS)
    han_windows = sorted(set(r["selected_window"] for r in han_rows))
    findings = [
        "P1/P2 methods: han_arw (Han et al. adaptive rolling window, PDF 3.5), "
        "diff_forgetting (Differentiable Forgetting, PDF 3.6), "
        "adamoe (AdaMoE-style closed-form mixture-of-experts, PDF 3.7).",
        f"Best method overall on locked test by mean log loss: {best['method']} (log loss {best['log_loss']:.4f}).",
    ]
    if p1_p2_mask.any():
        best_new = table[p1_p2_mask].iloc[0]
        findings.append(f"Best P1/P2 method: {best_new['method']} (log loss {best_new['log_loss']:.4f}).")
    findings.append(f"Han ARW selected windows used across prediction days: {han_windows}.")
    half_lives = [r["half_life"] for r in df_results.values() if r["half_life"] != float("inf")]
    if half_lives:
        findings.append(f"Differentiable Forgetting learned half-life ranged "
                         f"{min(half_lives):.2f}-{max(half_lives):.2f} days (mean {np.mean(half_lives):.2f}).")
    findings_text = "\n".join(f"- {line}" for line in findings)
    (out_dir / "advanced_findings.md").write_text("# P1/P2 findings\n\n" + findings_text + "\n")
    print("\n=== Findings ===")
    print(findings_text)
    print(f"\nAll outputs written to {out_dir}/")


if __name__ == "__main__":
    main()
