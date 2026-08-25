"""Run M1 (global adaptive short/long mixing) and M2 (context-dependent
short/long gating) -- adaptive-training-methods-implementation-plan.md
Stage 0-2 -- and merge them into the existing P0/P1/P2 comparison.

Both methods read their short-memory (rolling_3) and long-memory
(expanding) predictions straight from the shared WINDOW_FAMILY candidate
bank (the same one han_arw.py/adamoe.py use), so no new base-model fitting
is needed here -- only the candidate bank itself (fast: ~1min) plus the
cheap mixing/gating logic.

Run run_baselines.py and run_advanced.py first if you want this script's
comparison table/plots to include the full P0/P1/P2 ladder -- otherwise it
still runs and reports M1/M2 against the short/long candidates alone.

Example:
    python run_new_methods.py --source synthetic --synthetic-days 120 \\
        --synthetic-drift abrupt --synthetic-shift-day 95 --out results_synthetic_abrupt
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
from m1_global_mix import run_m1
from m2_context_gate import run_m2
from metrics import day_metrics
from run_advanced import rows_from_list
from run_baselines import aggregate_table
from short_long import LONG_NAME, SHORT_NAME
from splits import compute_splits

NEW_METHODS = ["m1_global_mix", "m2_context_gate"]


def group_breakdown(bank: dict, m1_rows: list, m2_rows: list, group: np.ndarray,
                     day: np.ndarray, test_days: set) -> pd.DataFrame:
    """Per-day, per-group (A=drifted-in-local-mode, B=stable) log loss for
    the short/long candidates and both new methods -- plan section 15's
    "local adaptation gap" (S4 diagnostic)."""
    by_day = {"m1_global_mix": {r["day"]: r for r in m1_rows},
              "m2_context_gate": {r["day"]: r for r in m2_rows}}
    rows = []
    for t in sorted(test_days):
        if t not in bank[SHORT_NAME] or t not in bank[LONG_NAME]:
            continue
        group_t = group[day == t]
        sources = {SHORT_NAME: bank[SHORT_NAME][t], LONG_NAME: bank[LONG_NAME][t]}
        for name, m in by_day.items():
            if t in m:
                sources[name] = m[t]
        for name, r in sources.items():
            y_true, y_pred = r["y_true"], r["y_pred"]
            for label, mask in [("A", group_t), ("B", ~group_t), ("overall", np.ones_like(group_t, dtype=bool))]:
                if mask.sum() == 0:
                    continue
                m = day_metrics(y_true[mask], y_pred[mask])
                rows.append({"day": t, "method": name, "group": label, **m})
    return pd.DataFrame(rows)


def plot_m1_diagnostics(m1_rows: list, out_path: Path, shift_day=None):
    days = [r["day"] for r in m1_rows]
    alpha = [r["alpha"] for r in m1_rows]
    alpha_oracle = [r["alpha_oracle"] for r in m1_rows]

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    ax = axes[0]
    ax.plot(days, alpha, marker="o", markersize=3, label="alpha_t (deployed, M1b)")
    ax.plot(days, alpha_oracle, marker="x", markersize=3, linestyle="--", label="alpha*_t (oracle, M1a, diagnostic)")
    ax.set_ylabel("alpha (0=long/expanding, 1=short/rolling_3)")
    ax.set_title("M1: global mixing weight over time")
    ax.legend(fontsize=8)
    if shift_day is not None:
        ax.axvline(shift_day, color="red", linestyle=":", linewidth=1, label="shift day")

    ax2 = axes[1]
    headroom = [r["oracle_headroom"] for r in m1_rows]
    ax2.plot(days, headroom, marker="o", markersize=3, color="darkorange")
    ax2.set_xlabel("prediction day")
    ax2.set_ylabel("deployed - oracle log loss")
    ax2.set_title("M1a oracle headroom (diagnostic only, never fed back)")
    if shift_day is not None:
        ax2.axvline(shift_day, color="red", linestyle=":", linewidth=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_m2_diagnostics(m2_rows: list, out_path: Path, group: np.ndarray = None,
                         day: np.ndarray = None, shift_day=None):
    days = [r["day"] for r in m2_rows]
    mean_a = np.array([r["mean_alpha"] for r in m2_rows])
    std_a = np.array([r["std_alpha"] for r in m2_rows])

    n_panels = 2 if group is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 3.5 * n_panels), sharex=True)
    axes = np.atleast_1d(axes)

    ax = axes[0]
    ax.plot(days, mean_a, marker="o", markersize=3, label="mean alpha_t(x)")
    ax.fill_between(days, mean_a - std_a, mean_a + std_a, alpha=0.2, label="+-1 std across x")
    ax.set_ylabel("alpha(x)")
    ax.set_title("M2: context-dependent gate weight over time (overall)")
    ax.legend(fontsize=8)
    if shift_day is not None:
        ax.axvline(shift_day, color="red", linestyle=":", linewidth=1)

    if group is not None:
        mean_a_grp = {"A (drifted)": [], "B (stable)": []}
        for r in m2_rows:
            t = r["day"]
            group_t = group[day == t]
            alpha_arr = r["alpha"]
            mean_a_grp["A (drifted)"].append(float(alpha_arr[group_t].mean()) if group_t.sum() else np.nan)
            mean_a_grp["B (stable)"].append(float(alpha_arr[~group_t].mean()) if (~group_t).sum() else np.nan)
        ax2 = axes[1]
        ax2.plot(days, mean_a_grp["A (drifted)"], marker="o", markersize=3, label="group A (drifted)")
        ax2.plot(days, mean_a_grp["B (stable)"], marker="s", markersize=3, label="group B (stable)")
        ax2.set_xlabel("prediction day")
        ax2.set_ylabel("mean alpha(x)")
        ax2.set_title("M2: gate weight by subpopulation (S4 local drift)")
        ax2.legend(fontsize=8)
        if shift_day is not None:
            ax2.axvline(shift_day, color="red", linestyle=":", linewidth=1)
    else:
        axes[0].set_xlabel("prediction day")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_source_args(parser)
    parser.add_argument("--n-features", type=int, default=2**18)
    parser.add_argument("--warmup-days", type=int, default=3)
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=1e-4, help="L2 reg for the candidate-bank base models.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--m1-val-window", type=int, default=3)
    parser.add_argument("--m2-lr", type=float, default=0.05)
    parser.add_argument("--m2-l2", type=float, default=1e-3)
    parser.add_argument("--m2-entropy-reg", type=float, default=1e-3)
    parser.add_argument("--m2-smooth-reg", type=float, default=1e-3)
    parser.add_argument("--m2-epochs-per-day", type=int, default=3)
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

    print("Running M1 (global adaptive mixing) ...")
    t0 = time.time()
    m1_rows = run_m1(bank, eligible_days, val_window=args.m1_val_window)
    print(f"  done in {time.time() - t0:.1f}s ({len(m1_rows)} prediction days)")

    print("Running M2 (context-dependent gating) ...")
    t0 = time.time()
    m2_rows = run_m2(bank, eligible_days, T=int(day.max()), lr=args.m2_lr, l2=args.m2_l2,
                      entropy_reg=args.m2_entropy_reg, smooth_reg=args.m2_smooth_reg,
                      epochs_per_day=args.m2_epochs_per_day, seed=args.seed,
                      context=context, day=day)
    print(f"  done in {time.time() - t0:.1f}s ({len(m2_rows)} prediction days)")

    new_per_day = pd.DataFrame(rows_from_list(m1_rows, "m1_global_mix") + rows_from_list(m2_rows, "m2_context_gate"))
    new_per_day.to_csv(out_dir / "m1_m2_per_day_metrics.csv", index=False)

    pd.DataFrame([{"day": r["day"], "alpha": r["alpha"], "alpha_oracle": r["alpha_oracle"],
                    "oracle_headroom": r["oracle_headroom"]} for r in m1_rows]) \
        .to_csv(out_dir / "m1_alpha.csv", index=False)
    pd.DataFrame([{"day": r["day"], "mean_alpha": r["mean_alpha"], "std_alpha": r["std_alpha"]}
                  for r in m2_rows]).to_csv(out_dir / "m2_alpha.csv", index=False)

    # Merge with whatever P0/P1/P2 (and SFTL, if present) results already exist.
    existing_candidates = [
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
        print("Warning: no existing P0/P1/P2 results found in out dir -- "
              "table will only include the short/long candidates + M1/M2.")
        existing_per_day = pd.DataFrame(columns=new_per_day.columns)
        existing_table = pd.DataFrame(columns=["method", "log_loss", "brier", "pr_auc",
                                                "mean_train_rows", "mean_fit_time_s", "n_test_days"])

    all_per_day = pd.concat([existing_per_day, new_per_day], ignore_index=True)
    all_per_day.to_csv(out_dir / "all_methods_with_m1_m2_per_day_metrics.csv", index=False)

    new_test = new_per_day[new_per_day["day"].isin(test_days)]
    new_table = aggregate_table(new_test) if len(new_test) else pd.DataFrame(columns=existing_table.columns)
    table = pd.concat([existing_table, new_table], ignore_index=True).sort_values("log_loss").reset_index(drop=True)
    table.to_csv(out_dir / "all_methods_with_m1_m2_comparison_table.csv", index=False)
    print("\n=== Locked-test comparison table (all methods + M1/M2) ===")
    print(table.to_string(index=False))

    shift_day = args.synthetic_shift_day
    if shift_day is None and args.synthetic_drift in ("abrupt", "local"):
        shift_day = args.synthetic_days // 2
    plot_m1_diagnostics(m1_rows, out_dir / "m1_diagnostics.png", shift_day=shift_day)
    plot_m2_diagnostics(m2_rows, out_dir / "m2_diagnostics.png", group=group, day=day, shift_day=shift_day)

    findings = [
        f"M1 (global adaptive mixing, val_window={args.m1_val_window}) and M2 (context-dependent gating) "
        "mix the shared rolling_3 (short) / expanding (long) candidate-bank predictions.",
    ]
    m1_test = new_per_day[(new_per_day["method"] == "m1_global_mix") & (new_per_day["day"].isin(test_days))]
    m2_test = new_per_day[(new_per_day["method"] == "m2_context_gate") & (new_per_day["day"].isin(test_days))]
    if len(m1_test):
        m1_alpha_test = [r["alpha"] for r in m1_rows if r["day"] in test_days]
        m1_headroom_test = [r["oracle_headroom"] for r in m1_rows if r["day"] in test_days]
        findings.append(f"M1 locked-test log loss {np.average(m1_test['log_loss'], weights=m1_test['n']):.4f}; "
                         f"deployed alpha ranged {min(m1_alpha_test):.2f}-{max(m1_alpha_test):.2f} "
                         f"(mean {np.mean(m1_alpha_test):.2f}); mean oracle headroom {np.mean(m1_headroom_test):.4f}.")
    if len(m2_test):
        m2_mean_alpha_test = [r["mean_alpha"] for r in m2_rows if r["day"] in test_days]
        findings.append(f"M2 locked-test log loss {np.average(m2_test['log_loss'], weights=m2_test['n']):.4f}; "
                         f"mean gate weight ranged {min(m2_mean_alpha_test):.2f}-{max(m2_mean_alpha_test):.2f}.")
    best = table.iloc[0]
    findings.append(f"Best method overall on locked test: {best['method']} (log loss {best['log_loss']:.4f}).")

    if group is not None:
        gb = group_breakdown(bank, m1_rows, m2_rows, group, day, test_days)
        gb.to_csv(out_dir / "group_breakdown.csv", index=False)
        summary = gb.groupby(["method", "group"]).apply(lambda g: np.average(g["log_loss"], weights=g["n"])) \
            .unstack("group").reset_index()
        summary.to_csv(out_dir / "group_breakdown_summary.csv", index=False)
        print("\n=== Group breakdown (locked test, mean log loss) ===")
        print(summary.to_string(index=False))
        findings.append("Group breakdown (locked-test mean log loss, A=drifted subpopulation, "
                         "B=stable): see group_breakdown_summary.csv.")

    findings_text = "\n".join(f"- {line}" for line in findings)
    (out_dir / "m1_m2_findings.md").write_text("# M1/M2 findings\n\n" + findings_text + "\n")
    print("\n=== Findings ===")
    print(findings_text)
    print(f"\nAll outputs written to {out_dir}/")


if __name__ == "__main__":
    main()
