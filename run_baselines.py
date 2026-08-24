"""Run the P0 baseline ladder on the Criteo dataset and produce a first
comparison table + diagnostic plots (PDF section 2, 5, 9).

Example:
    python run_baselines.py --sample-frac 0.2   # quick preliminary pass
    python run_baselines.py                     # full dataset
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baselines import build_candidates, fit_predict, WINDOW_FAMILY
from data import load_dataset
from metrics import day_metrics
from splits import compute_splits


def run_all_days(X, y, day, days, candidates, alpha, seed):
    """Fit+predict every candidate for every prediction day in `days`.
    Returns one row per (method, day) with metrics and bookkeeping info.
    """
    rows = []
    for t in days:
        for name, rule in candidates.items():
            result = fit_predict(X, y, day, t, rule, alpha=alpha, seed=seed)
            if result is None:
                continue
            m = day_metrics(result["y_true"], result["y_pred"])
            rows.append({
                "method": name, "day": t, **m,
                "n_train": result["n_train"], "fit_time": result["fit_time"],
            })
        print(f"  day {t}: done", flush=True)
    return pd.DataFrame(rows)


def weighted_mean(df, col):
    return np.average(df[col], weights=df["n"])


def aggregate_table(per_day: pd.DataFrame) -> pd.DataFrame:
    out = []
    for method, g in per_day.groupby("method"):
        out.append({
            "method": method,
            "log_loss": weighted_mean(g, "log_loss"),
            "brier": weighted_mean(g, "brier"),
            "pr_auc": g["pr_auc"].mean(skipna=True),
            "mean_train_rows": g["n_train"].mean(),
            "mean_fit_time_s": g["fit_time"].mean(),
            "n_test_days": len(g),
        })
    return pd.DataFrame(out).sort_values("log_loss").reset_index(drop=True)


def select_validation_window(per_day: pd.DataFrame, dev_days) -> str:
    """Pick the fixed window (from WINDOW_FAMILY) with the best mean dev-period
    log loss, freezing it before it ever sees test days (PDF 3.4)."""
    dev = per_day[per_day["method"].isin(WINDOW_FAMILY) & per_day["day"].isin(dev_days)]
    scores = dev.groupby("method").apply(lambda g: weighted_mean(g, "log_loss"))
    return scores.idxmin()


def hindsight_best_window(per_day: pd.DataFrame, test_days) -> pd.Series:
    """For each test day, which fixed window (WINDOW_FAMILY) had the best
    log loss in hindsight. Diagnostic only -- never used by a deployed method
    (PDF section 5)."""
    test = per_day[per_day["method"].isin(WINDOW_FAMILY) & per_day["day"].isin(test_days)]
    best = test.loc[test.groupby("day")["log_loss"].idxmin()]
    return best.set_index("day")["method"]


def plot_per_day_logloss(per_day: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, g in per_day.groupby("method"):
        g = g.sort_values("day")
        ax.plot(g["day"], g["log_loss"], marker="o", markersize=3, label=method)
    ax.set_xlabel("prediction day")
    ax.set_ylabel("log loss")
    ax.set_title("Per-day log loss, P0 baselines")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hindsight_window(hindsight: pd.Series, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    categories = WINDOW_FAMILY
    y = [categories.index(m) for m in hindsight.values]
    ax.scatter(hindsight.index, y, marker="s")
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xlabel("test day")
    ax.set_title("Hindsight best fixed window h*_t (diagnostic only)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/criteo_attribution_dataset.tsv.gz")
    parser.add_argument("--sample-frac", type=float, default=1.0,
                         help="Subsample rows for a faster preliminary run.")
    parser.add_argument("--n-features", type=int, default=2**18, help="Hashing-trick dimensionality.")
    parser.add_argument("--warmup-days", type=int, default=3,
                         help="Skip this many initial days (too little history to be meaningful).")
    parser.add_argument("--test-frac", type=float, default=0.3,
                         help="Fraction of eligible days held out as the locked test period.")
    parser.add_argument("--alpha", type=float, default=1e-4, help="L2 regularization strength.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data} (sample_frac={args.sample_frac}) ...")
    X, y, day = load_dataset(args.data, n_features=args.n_features,
                              sample_frac=args.sample_frac, seed=args.seed)
    print(f"Loaded {X.shape[0]} rows, {day.max() + 1} days, click rate {y.mean():.3f}")

    eligible_days, dev_days, test_days = compute_splits(day, args.warmup_days, args.test_frac)
    print(f"Dev days: {sorted(dev_days)}")
    print(f"Test days (locked): {sorted(test_days)}")

    candidates = build_candidates()
    print(f"Running {len(candidates)} P0 methods over {len(eligible_days)} prediction days ...")
    start = time.time()
    per_day = run_all_days(X, y, day, eligible_days, candidates, args.alpha, args.seed)
    print(f"Done in {time.time() - start:.1f}s")
    per_day.to_csv(out_dir / "per_day_metrics.csv", index=False)

    # Validation-selected window: freeze on dev, then read off its test-day rows.
    h_star = select_validation_window(per_day, dev_days)
    val_selected = per_day[(per_day["method"] == h_star) & (per_day["day"].isin(test_days))].copy()
    val_selected["method"] = f"validation_selected(h={h_star})"

    test_metrics = per_day[per_day["day"].isin(test_days)]
    table = aggregate_table(pd.concat([test_metrics, val_selected], ignore_index=True))
    table.to_csv(out_dir / "comparison_table.csv", index=False)
    print("\n=== Locked-test comparison table ===")
    print(table.to_string(index=False))

    hindsight = hindsight_best_window(per_day, test_days)
    hindsight.to_csv(out_dir / "hindsight_best_window.csv")

    plot_per_day_logloss(test_metrics, out_dir / "per_day_logloss.png")
    plot_hindsight_window(hindsight, out_dir / "hindsight_best_window.png")

    n_distinct = hindsight.nunique()
    findings = [
        f"Prediction days: {len(eligible_days)} total ({len(dev_days)} dev, {len(test_days)} locked test).",
        f"Best baseline on locked test by mean log loss: {table.iloc[0]['method']} "
        f"(log loss {table.iloc[0]['log_loss']:.4f}).",
        f"Validation-selected window on dev period: h={h_star}.",
        f"Hindsight best fixed window took {n_distinct} distinct value(s) across the {len(test_days)} "
        f"test days: {sorted(hindsight.unique())}.",
    ]
    if n_distinct == 1:
        findings.append("One fixed horizon dominated every test day -- no empirical motivation yet "
                         "for adaptive memory from this diagnostic alone (see PDF section 5, 10).")
    else:
        findings.append("The best horizon changed across test days -- motivates checking whether "
                         "the adaptive baselines (Han ARW, Differentiable Forgetting) track that "
                         "variation (see PDF section 5, 10).")
    findings_text = "\n".join(f"- {line}" for line in findings)
    (out_dir / "findings.md").write_text("# Preliminary findings\n\n" + findings_text + "\n")
    print("\n=== Findings ===")
    print(findings_text)
    print(f"\nAll outputs written to {out_dir}/")


if __name__ == "__main__":
    main()
