"""Paired day-level uncertainty for the TTAM revised Section 6 (plan
section 7).

Estimands.  ``L_{m,d,s}`` is the mean impression log loss of method ``m``
on origin day ``d`` and seed ``s``.  Seeds are averaged **first**:

    A_{m,d} = (1/S) sum_s L_{m,d,s}                 seed-averaged daily loss
    A_m     = (1/D) sum_d A_{m,d}                   equal-day score
    G_{m,d} = A_{m,d} - A_{TTAM,d}                  daily gain (positive favours TTAM)
    G_m     = (1/D) sum_d G_{m,d}                   mean paired gain

For every comparator we report the mean score ``A_m``, the mean paired
gain ``G_m``, a 95% paired day-bootstrap interval for the gain (10,000
resamples, fixed resampling seed 20260908, the *same* sampled days
applied to the pair after seed-averaging), the origins won, and -- as a
serial-dependence sensitivity -- a moving-block bootstrap with block
length two.  Impressions and seed-by-day cells are never treated as
independent temporal replicates.  With only five Avazu origins both
interval estimates are descriptive; no significance stars, no universal
superiority language.

Secondary appendix metrics (impression-weighted log loss, Brier score,
calibration error) are reported separately and never mixed with the
equal-day gain.

Usage::

    PYTHONPATH=. python3 final_experiments/ttam_stats.py \
        --dir final_experiments/ttam/criteo/nested --label "Criteo" \
        --dir final_experiments/ttam/avazu/nested  --label "Avazu" \
        --out final_experiments/TTAM_SECTION6_STATS.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RESAMPLE_SEED = 20260908
N_BOOT = 10_000
TTAM = "ttam"

MAIN_METHODS = ["expanding", "best_fixed_window", "arw", "adamoe", "ops", "ttam"]
ABLATION_METHODS = ["without_both", "amgtp_only", "apops_only", "ttam"]
PRETTY = {
    "expanding": "Expanding", "best_fixed_window": "Best Fixed Window", "arw": "ARW",
    "adamoe": "AdaMoE", "ops": "OPS", "ttam": "TTAM",
    "without_both": "Without both modules", "amgtp_only": "AMG-TP only",
    "apops_only": "AP-OPS only",
}


def load_per_seed(result_dir: Path) -> pd.DataFrame:
    rows = []
    for sd in sorted(result_dir.glob("seed*")):
        csv = sd / "per_day_metrics.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        df["seed"] = int(sd.name.replace("seed", ""))
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"no seed*/per_day_metrics.csv under {result_dir}")
    return pd.concat(rows, ignore_index=True)


def seed_average(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """A_{m,d} for the given per-day value column -> wide (index=day,
    columns=method)."""
    g = df.groupby(["method", "day"])[value].mean().reset_index()
    return g.pivot(index="day", columns="method", values=value).sort_index()


def _boot_ci(per_day_gain: np.ndarray, rng: np.random.Generator) -> tuple:
    D = len(per_day_gain)
    idx = rng.integers(0, D, size=(N_BOOT, D))
    means = per_day_gain[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _mbb_ci(per_day_gain: np.ndarray, rng: np.random.Generator, block: int = 2) -> tuple:
    D = len(per_day_gain)
    if D < 2 * block:
        return (float("nan"), float("nan"))
    starts = np.arange(0, D - block + 1)
    n_blocks = int(np.ceil(D / block))
    means = []
    for _ in range(N_BOOT):
        s = rng.choice(starts, size=n_blocks, replace=True)
        sample = np.concatenate([per_day_gain[a:a + block] for a in s])[:D]
        means.append(sample.mean())
    means = np.asarray(means)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_table(wide_ll: pd.DataFrame, methods: list, wide_secondary: dict) -> pd.DataFrame:
    days = list(wide_ll.index)
    D = len(days)
    a_ttam = wide_ll[TTAM].to_numpy()
    rows = []
    for m in methods:
        if m not in wide_ll.columns:
            continue
        a_m = wide_ll[m].to_numpy()
        gain = a_m - a_ttam                       # G_{m,d}; positive => TTAM better
        rng = np.random.default_rng(RESAMPLE_SEED)
        lo, hi = _boot_ci(gain, rng)
        mlo, mhi = _mbb_ci(gain, rng)
        row = {
            "method": PRETTY.get(m, m),
            "A_m (mean score)": float(a_m.mean()),
            "G_m (mean gain vs TTAM)": float(gain.mean()),
            "gain_ci95_lo": lo, "gain_ci95_hi": hi,
            "mbb_ci95_lo": mlo, "mbb_ci95_hi": mhi,
            "origins_won_by_ttam": int(np.sum(gain > 0)),
            "origins_lost_by_ttam": int(np.sum(gain < 0)),
            "origins_tied": int(np.sum(gain == 0)),
            "D": D,
        }
        for sname, sw in wide_secondary.items():
            if m in sw.columns:
                row[f"appendix_{sname}"] = float(sw[m].to_numpy().mean())
                row[f"appendix_{sname}_ttam"] = float(sw[TTAM].to_numpy().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def analyse(result_dir: Path, label: str) -> dict:
    raw = load_per_seed(result_dir)
    wide_ll = seed_average(raw, "log_loss")

    # impression-weighted log loss per (method, day): sum_s sum_i ll / sum_s n
    raw["ll_sum"] = raw["log_loss"] * raw["n"]
    agg = raw.groupby(["method", "day"])[["ll_sum", "n"]].sum().reset_index()
    agg["iwll"] = agg["ll_sum"] / agg["n"]
    iw = agg.pivot(index="day", columns="method", values="iwll").sort_index()
    secondary = {"iw_log_loss": iw}
    for col in ("brier", "ece"):
        if col in raw.columns:
            secondary[col] = seed_average(raw, col)

    main = paired_table(wide_ll, MAIN_METHODS, secondary)
    ablation = paired_table(wide_ll, ABLATION_METHODS, secondary)
    return {
        "label": label, "dir": str(result_dir),
        "days": [int(d) for d in wide_ll.index],
        "main": main, "ablation": ablation,
        "seed_averaged_log_loss": {m: {int(d): float(wide_ll[m][d]) for d in wide_ll.index}
                                   for m in wide_ll.columns},
    }


def _fmt_row(r) -> str:
    ci = f"[{r['gain_ci95_lo']:+.6f}, {r['gain_ci95_hi']:+.6f}]"
    mbb = f"[{r['mbb_ci95_lo']:+.6f}, {r['mbb_ci95_hi']:+.6f}]"
    won = f"{r['origins_won_by_ttam']}/{r['D']}"
    return (f"| {r['method']} | {r['A_m (mean score)']:.6f} | {r['G_m (mean gain vs TTAM)']:+.6f} | "
            f"{ci} | {mbb} | {won} |")


def to_markdown(analyses: list) -> str:
    L = ["# TTAM revised Section 6 -- paired day-level statistics", "",
         "Seeds are averaged within each origin day *before* any interval is",
         "formed; the origin day is the unit. Gain `G_{m,d} = A_{m,d} -",
         "A_{TTAM,d}` is positive when TTAM has the lower loss. Bootstrap:",
         f"{N_BOOT:,} paired day resamples, resampling seed {RESAMPLE_SEED};",
         "`mbb` is the block-2 moving-block bootstrap (serial-dependence",
         "sensitivity). Impression-weighted loss / Brier / ECE are appendix",
         "metrics only (see the CSVs) and are not mixed with the equal-day gain.",
         ""]
    for a in analyses:
        D = len(a["days"])
        L += [f"## {a['label']}  (D = {D} origins: {a['days']})", ""]
        if D < 6:
            L += [f"> Only {D} origins -- both interval estimates are descriptive and "
                  "fragile. No significance claims; unfavourable origins are retained.", ""]
        L += ["### 6.2 Main comparison (gain of TTAM over each)", "",
              "| method | mean score A_m | mean gain G_m | 95% day-bootstrap CI | block-2 MBB CI | origins won by TTAM |",
              "|---|---|---|---|---|---|"]
        for _, r in a["main"].iterrows():
            if r["method"] == "TTAM":
                continue
            L.append(_fmt_row(r))
        ttam_row = a["main"][a["main"]["method"] == "TTAM"]
        if len(ttam_row):
            L.append(f"| TTAM | {ttam_row.iloc[0]['A_m (mean score)']:.6f} | (reference) | -- | -- | -- |")
        L += ["", "### 6.3 Ablation (gain of TTAM over each ablation)", "",
              "| variant | mean score A_m | mean gain G_m | 95% day-bootstrap CI | block-2 MBB CI | origins won by TTAM |",
              "|---|---|---|---|---|---|"]
        for _, r in a["ablation"].iterrows():
            if r["method"] == "TTAM":
                continue
            L.append(_fmt_row(r))
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", action="append", dest="dirs", required=True)
    ap.add_argument("--label", action="append", dest="labels", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    if len(args.dirs) != len(args.labels):
        ap.error("one --label per --dir")

    analyses = [analyse(Path(d), lab) for d, lab in zip(args.dirs, args.labels)]
    for a in analyses:
        a["main"].to_csv(Path(a["dir"]) / "section6_main.csv", index=False)
        a["ablation"].to_csv(Path(a["dir"]) / "section6_ablation.csv", index=False)
        print(f"wrote {a['dir']}/section6_main.csv + section6_ablation.csv")

    md = to_markdown(analyses)
    if args.out:
        Path(args.out).write_text(md)
        print(f"wrote {args.out}")
    else:
        print("\n" + md)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            [{k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v)
              for k, v in a.items()} for a in analyses], indent=2, default=float))


if __name__ == "__main__":
    main()
