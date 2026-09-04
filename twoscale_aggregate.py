"""Aggregate two-timescale cells (one per seed / dataset) into tables, a few
figures and a REPORT.md. Reads every ``summary.json`` under a stage dir.

    python twoscale_aggregate.py --stage twoscale_experiments/criteo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

def _md_table(df, index=True):
    return "```\n" + df.to_string(index=index) + "\n```"


METHOD_ORDER = ["expanding", "rolling_3", "rolling_7", "equal_ensemble", "long_only",
                "short_only", "combined", "time_of_day", "online_platt", "oracle_intercept"]
KEY_COMPARISONS = ["combined_vs_long_only", "combined_vs_short_only", "combined_vs_expanding",
                   "combined_vs_time_of_day", "short_only_vs_expanding", "long_only_vs_expanding",
                   "online_platt_vs_combined"]


def wilcoxon_sign(vals):
    """Sign test p-value that the paired seed deltas are < 0 (small n)."""
    v = np.asarray([x for x in vals if np.isfinite(x)])
    if len(v) == 0:
        return np.nan
    k = int(np.sum(v < 0)); n = len(v)
    from math import comb
    # one-sided: P(X >= k) under Binom(n, 0.5)
    p = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    stage = Path(args.stage)
    out = Path(args.out) if args.out else stage
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    cells = []
    for sj in sorted(stage.glob("**/summary.json")):
        s = json.loads(sj.read_text())
        s["_dir"] = str(sj.parent)
        cells.append(s)
    if not cells:
        raise SystemExit(f"no summary.json under {stage}")
    print(f"{len(cells)} cells")

    # ---- per-method imp-weighted log loss across seeds ------------------
    rows = []
    for s in cells:
        seed = s["config"]["seed"]
        for m, ms in s["methods"].items():
            rows.append({"seed": seed, "method": m,
                         "imp_wt_ll": ms["imp_weighted_log_loss"],
                         "daily_ll": ms["daily_mean_log_loss"],
                         "worst_day": ms["worst_day_log_loss"],
                         "brier": ms["brier"], "ece": ms["ece"]})
    df = pd.DataFrame(rows)
    agg = (df.groupby("method")
             .agg(imp_wt_ll_mean=("imp_wt_ll", "mean"), imp_wt_ll_std=("imp_wt_ll", "std"),
                  daily_ll_mean=("daily_ll", "mean"), worst_day_mean=("worst_day", "mean"),
                  brier_mean=("brier", "mean"), ece_mean=("ece", "mean"), n=("imp_wt_ll", "size"))
             .reindex([m for m in METHOD_ORDER if m in set(df.method)]))
    agg.to_csv(out / "tables" / "methods.csv")

    # ---- decisive paired comparisons ---------------------------------
    comp_rows = []
    for key in KEY_COMPARISONS:
        deltas = [s["comparisons"][key]["mean_delta"] for s in cells if key in s["comparisons"]]
        won = [s["comparisons"][key]["days_won_frac"] for s in cells if key in s["comparisons"]]
        sig = [s["comparisons"][key]["significant_below_zero"] for s in cells if key in s["comparisons"]]
        comp_rows.append({"comparison": key, "n_seeds": len(deltas),
                          "mean_delta": float(np.mean(deltas)) if deltas else np.nan,
                          "delta_sd": float(np.std(deltas)) if len(deltas) > 1 else 0.0,
                          "seeds_delta_neg": int(np.sum(np.array(deltas) < 0)),
                          "mean_days_won_frac": float(np.mean(won)) if won else np.nan,
                          "seeds_ci_below_zero": int(np.sum(sig)),
                          "sign_test_p": wilcoxon_sign(deltas)})
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(out / "tables" / "comparisons.csv", index=False)

    # ---- feasibility (mean over cells) ------------------------------
    feas = pd.DataFrame([{**{"seed": s["config"]["seed"]},
                          **{k: v for k, v in s["feasibility"].items() if not isinstance(v, dict)}}
                         for s in cells])
    feas.to_csv(out / "tables" / "feasibility.csv", index=False)

    # ---- ablations -------------------------------------------------
    ab_rows = []
    for s in cells:
        d = s["_dir"]
        f = Path(d) / "ablations.csv"
        if f.exists():
            a = pd.read_csv(f); a["seed"] = s["config"]["seed"]
            ab_rows.append(a)
    ab = pd.concat(ab_rows, ignore_index=True) if ab_rows else pd.DataFrame()
    if len(ab):
        ab_agg = ab.groupby(["ablation", "setting"]).imp_wt_ll.agg(["mean", "std", "size"]).reset_index()
        ab_agg.to_csv(out / "tables" / "ablations.csv", index=False)

    # ---- figures -------------------------------------------------
    if HAVE_MPL:
        _fig_methods(agg, out / "figures" / "methods_logloss.png")
        _fig_intraday(cells, out / "figures" / "intraday_residual.png")
        _fig_beta(cells, out / "figures" / "b_trace.png")

    # ---- REPORT.md ---------------------------------------------
    write_report(out / "REPORT.md", cells, agg, comp, feas, ab_agg if len(ab) else None)
    print(f"-> {out}/REPORT.md")


def _fig_methods(agg, path):
    a = agg.dropna(subset=["imp_wt_ll_mean"])
    plt.figure(figsize=(7, 4))
    plt.barh(range(len(a)), a["imp_wt_ll_mean"], xerr=a["imp_wt_ll_std"].fillna(0))
    plt.yticks(range(len(a)), a.index)
    plt.xlabel("locked-test impression-weighted log loss")
    plt.gca().invert_yaxis()
    lo = a["imp_wt_ll_mean"].min()
    plt.xlim(lo - 0.002, a["imp_wt_ll_mean"].max() + 0.003)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def _fig_intraday(cells, path):
    frames = []
    for s in cells:
        f = Path(s["_dir"]) / "intraday_blocks.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
    if not frames:
        return
    d = pd.concat(frames)
    plt.figure(figsize=(7, 4))
    for m, g in d.groupby("method"):
        gg = g.groupby("block").mean_residual.mean()
        plt.plot(gg.index, gg.values, marker=".", label=m)
    plt.axhline(0, color="k", lw=0.5)
    plt.xlabel("within-day hour block"); plt.ylabel("mean residual (y - p)")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def _fig_beta(cells, path):
    frames = []
    for s in cells:
        f = Path(s["_dir"]) / "calib_traces.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
    if not frames:
        return
    d = pd.concat(frames)
    plt.figure(figsize=(7, 4))
    for m, g in d.groupby("method"):
        gg = g.groupby("day").b_end.mean()
        plt.plot(gg.index, gg.values, marker=".", label=m)
    plt.xlabel("day"); plt.ylabel("end-of-day intercept b")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def write_report(path, cells, agg, comp, feas, ab_agg):
    src = cells[0]["config"]["source"]
    seeds = sorted(s["config"]["seed"] for s in cells)
    L = []
    L.append(f"# Two-timescale CTR forecasting -- {src}\n")
    L.append(f"{len(cells)} cell(s), seeds {seeds}. "
             f"Split: train {cells[0]['config']['train_days']}, "
             f"dev {cells[0]['config']['dev_days']}, test {cells[0]['config']['test_days']}.\n")
    L.append(f"Frozen calibrator config: `{json.dumps(cells[0]['config']['calib'])}`\n")

    L.append("\n## Feasibility (dev days, plan section 5)\n")
    L.append(f"- mean daily oracle-intercept improvement: "
             f"{feas['dev_mean_oracle_improvement'].mean():.5f} log loss "
             f"({feas['dev_mean_rel_oracle_improvement'].mean()*100:.3f}% relative)\n")
    L.append(f"- mean longest same-sign intraday residual run: "
             f"{feas['dev_mean_longest_same_sign_run'].mean():.1f} blocks; "
             f"lag-1 residual autocorr {feas['dev_mean_residual_autocorr_lag1'].mean():.3f}\n")

    L.append("\n## Locked-test log loss by method\n\n")
    L.append(_md_table(agg.round(5)) + "\n")

    L.append("\n## Decisive paired comparisons (Delta = method - baseline, <0 favours method)\n\n")
    L.append(_md_table(comp.round(5), index=False) + "\n")

    if ab_agg is not None:
        L.append("\n## Ablations (plan section 9)\n\n")
        L.append(_md_table(ab_agg.round(5), index=False) + "\n")

    # verdict
    sc = [s["success_criteria"] for s in cells]
    beats_long = np.mean([x["1_beats_long_only_ci"] for x in sc])
    beats_short = np.mean([x["2_beats_short_only"] for x in sc])
    interp = cells[0]["success_criteria"]["interpretation"]
    L.append("\n## Verdict (plan section 10)\n")
    L.append(f"- criterion 1 (combined beats long-only, paired CI<0): {beats_long*100:.0f}% of seeds\n")
    L.append(f"- criterion 2 (combined beats short-only): {beats_short*100:.0f}% of seeds\n")
    L.append(f"- interpretation: **{interp}**\n")
    path.write_text("".join(L))


if __name__ == "__main__":
    main()
