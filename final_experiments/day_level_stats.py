"""Day-level significance re-analysis (addresses review comment 1).

The headline runners (``run_final.py``) pool ``(seed, test day)`` as the
replicate unit for every CI. That over-states precision for a *temporal*
claim: the three seeds on the same calendar day share that day's traffic,
that day's label noise and that day's drift, so they are far from
independent. The calendar day is the exchangeable unit here, not the
(seed, day) pair.

This module re-does the analysis the way the comment asks:

    Lbar_{m,d} = (1/S) sum_s L_{m,d,s}         # average the seeds FIRST
    then bootstrap / sign-test across the D days (Criteo D=9, Avazu D=3).

It reads the per-seed ``per_day_metrics.csv`` files an experiment already
wrote (``<dir>/seed<k>/per_day_metrics.csv``) so it needs no recompute,
and reuses ``withinday.daystats.day_summary`` for the bootstrap CI / sign
test / leave-one-day-out / moving-block bootstrap rather than
reimplementing them.

Usage::

    PYTHONPATH=. .venv/bin/python final_experiments/day_level_stats.py \
        --dir final_experiments/criteo/final --label "Criteo (fixed origin)" \
        --dir final_experiments/avazu/final  --label "Avazu (fixed origin)" \
        --out final_experiments/DAY_LEVEL_STATS.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from withinday.daystats import day_summary

# baselines every other method is differenced against, in preference order
# (a dir may contain only some of these method rows)
BASELINES = ["expanding", "long_only"]


def load_per_seed(result_dir: Path) -> pd.DataFrame:
    """Long frame: one row per (seed, method, day) with log_loss and n."""
    rows = []
    for sd in sorted(result_dir.glob("seed*")):
        csv = sd / "per_day_metrics.csv"
        if not csv.exists():
            continue
        seed = int(sd.name.replace("seed", ""))
        df = pd.read_csv(csv)
        df["seed"] = seed
        rows.append(df[["seed", "method", "day", "n", "log_loss"]])
    if not rows:
        raise FileNotFoundError(f"no seed*/per_day_metrics.csv under {result_dir}")
    return pd.concat(rows, ignore_index=True)


def seed_averaged(df: pd.DataFrame) -> pd.DataFrame:
    """Lbar_{m,d}: mean log loss over seeds, one row per (method, day).

    ``n`` is taken as the per-day mean row count (identical across seeds at
    sample_frac=1.0; averaged defensively otherwise)."""
    g = (df.groupby(["method", "day"])
           .agg(lbar=("log_loss", "mean"), ll_sd=("log_loss", "std"),
                n=("n", "mean"), n_seeds=("seed", "nunique"))
           .reset_index())
    return g


def per_seed_day_means(df: pd.DataFrame, method: str, base: str) -> dict:
    """For each seed, that seed's own equal-day-weighted mean delta vs the
    baseline -- to show how little the day-level effect moves seed to seed
    (the thing the pooled CI conflates with genuine day-to-day noise)."""
    out = {}
    for seed, sub in df.groupby("seed"):
        piv = sub.pivot_table(index="day", columns="method", values="log_loss")
        if method not in piv or base not in piv:
            continue
        out[int(seed)] = float((piv[method] - piv[base]).mean())
    return out


def analyse_dir(result_dir: Path, label: str) -> dict:
    raw = load_per_seed(result_dir)
    sa = seed_averaged(raw)
    piv = sa.pivot_table(index="day", columns="method", values="lbar").sort_index()
    n_by_day = sa.groupby("day")["n"].first()
    days = list(piv.index)
    methods = [m for m in piv.columns]
    present_baselines = [b for b in BASELINES if b in methods]

    results = []
    for base in present_baselines:
        for m in methods:
            if m == base:
                continue
            deltas = (piv[m] - piv[base]).to_numpy()
            n_arr = n_by_day.reindex(days).to_numpy()
            s = day_summary(deltas, seed=0)
            imp_wt = float(np.sum(deltas * n_arr) / np.sum(n_arr))
            # tie-aware day counts (best_fixed can literally equal expanding
            # on origins where h* == 'expanding'); the sign test in
            # day_summary counts only deltas < 0 as wins
            tie = np.abs(deltas) < 1e-9
            n_won = int(np.sum(deltas < -1e-9))
            n_lost = int(np.sum(deltas > 1e-9))
            n_tied = int(np.sum(tie))
            sign_p_notie = (float(stats.binomtest(n_won, n_won + n_lost, 0.5).pvalue)
                            if (n_won + n_lost) else float("nan"))
            per_seed = per_seed_day_means(raw, m, base)
            ps_vals = list(per_seed.values())
            # exact binomial floor: best possible two-sided sign-test p at this D
            best_p = float(stats.binomtest(len(days), len(days), 0.5).pvalue)
            results.append({
                "label": label, "baseline": base, "method": m, "n_days": len(days),
                "mean_delta_daywt": s["mean_delta"], "median_delta": s["median_delta"],
                "imp_wt_delta": imp_wt,
                "ci95_lo": s["ci95_lo"], "ci95_hi": s["ci95_hi"],
                "ci_excludes_0": bool(s["ci95_hi"] < 0 or s["ci95_lo"] > 0),
                "n_days_won": n_won, "n_days_lost": n_lost, "n_days_tied": n_tied,
                "frac_days_won": s["frac_days_won"],
                "sign_test_p": s["sign_test_p"], "sign_test_p_notie": sign_p_notie,
                "sign_test_p_floor": best_p,
                "mbb_ci_lo": (s["moving_block_bootstrap"] or {}).get("ci95_lo"),
                "mbb_ci_hi": (s["moving_block_bootstrap"] or {}).get("ci95_hi"),
                "loo_reverses_sign": s["loo_reverses_sign"],
                "per_seed_day_means": per_seed,
                "per_seed_spread": (max(ps_vals) - min(ps_vals)) if ps_vals else float("nan"),
            })
    return {"label": label, "dir": str(result_dir), "days": days,
            "methods": methods, "table": results,
            "seed_averaged_losses": {m: {int(d): float(piv[m][d]) for d in days} for m in methods}}


def fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    return f"{p:.3f}" if p >= 1e-3 else f"{p:.1e}"


def to_markdown(analyses: list[dict]) -> str:
    lines = ["# Day-level significance (seeds averaged first, then day-level inference)",
             "",
             "Review comment 1. `Lbar_{m,d}` averages the 3 seeds on each calendar",
             "day; inference is then over the D calendar days (the exchangeable unit",
             "for a temporal claim), **not** the 27 / 9 pooled (seed, day) cells the",
             "headline table used. Deltas are equal-day-weighted mean log-loss",
             "differences; negative favours the method. Bootstrap CI is the",
             "percentile bootstrap over the D days; `sign p` is the two-sided",
             "sign test; `p floor` is the smallest two-sided sign-test p attainable",
             "at this D (a clean sweep).",
             ""]
    for a in analyses:
        lines += [f"## {a['label']}  (D = {len(a['days'])} days: {a['days']})", ""]
        by_base = {}
        for r in a["table"]:
            by_base.setdefault(r["baseline"], []).append(r)
        for base, rows in by_base.items():
            lines += [f"### vs `{base}`", "",
                      "| method | mean d (day-wt) | imp-wt d | 95% CI (day bootstrap) | CI excl 0 | W-L-T | sign p | p floor | seed spread |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for r in sorted(rows, key=lambda x: x["mean_delta_daywt"]):
                ci = f"[{r['ci95_lo']:+.6f}, {r['ci95_hi']:+.6f}]"
                wlt = f"{r['n_days_won']}-{r['n_days_lost']}-{r['n_days_tied']}"
                sp = fmt_p(r["sign_test_p_notie"]) if r["n_days_tied"] else fmt_p(r["sign_test_p"])
                lines.append(
                    f"| {r['method']} | {r['mean_delta_daywt']:+.6f} | {r['imp_wt_delta']:+.6f} | "
                    f"{ci} | {'yes' if r['ci_excludes_0'] else 'no'} | "
                    f"{wlt} | {sp} | {fmt_p(r['sign_test_p_floor'])} | {r['per_seed_spread']:.6f} |")
            lines.append("")
            if len(a["days"]) < 5:
                lines += [
                    f"> **{len(a['days'])} days is too few for a day-level bootstrap CI or a "
                    "significant sign test.** Even a clean sweep gives two-sided "
                    f"p = {fmt_p(rows[0]['sign_test_p_floor'])}. Treat these rows as "
                    "descriptive; the temporal significance statement for this dataset "
                    "has to come from the rolling-origin run (more origins) or a fresh "
                    "chronological stream.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", action="append", dest="dirs", required=True,
                    help="a result dir containing seed*/per_day_metrics.csv (repeatable)")
    ap.add_argument("--label", action="append", dest="labels", required=True,
                    help="one label per --dir, same order")
    ap.add_argument("--out", default=None, help="write the markdown report here")
    ap.add_argument("--json-out", default=None, help="also dump the full analysis as JSON")
    args = ap.parse_args()
    if len(args.dirs) != len(args.labels):
        ap.error("need exactly one --label per --dir")

    analyses = [analyse_dir(Path(d), lab) for d, lab in zip(args.dirs, args.labels)]

    for a in analyses:
        csv_rows = [{k: v for k, v in r.items() if k != "per_seed_day_means"} for r in a["table"]]
        out_csv = Path(a["dir"]) / "day_level_stats.csv"
        pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
        print(f"wrote {out_csv}")

    md = to_markdown(analyses)
    if args.out:
        Path(args.out).write_text(md)
        print(f"wrote {args.out}")
    else:
        print("\n" + md)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(analyses, indent=2, default=float))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
