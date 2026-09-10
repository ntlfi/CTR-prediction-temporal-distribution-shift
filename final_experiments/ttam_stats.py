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

# 2x2 ablation: historical predictor {AMG-TP, expanding-history}
#             x calibration {AP-OPS, none}.
#   (AMG-TP,     AP-OPS) = ttam            (AMG-TP,     none) = amgtp_only
#   (expanding,  AP-OPS) = expanding_apops (expanding,  none) = expanding
ABLATION_METHODS = ["expanding", "amgtp_only", "expanding_apops", "ttam"]
CELL = {                                    # 2x2 cell labels (for the factorial table)
    ("expanding", "none"): "expanding", ("expanding", "apops"): "expanding_apops",
    ("amgtp", "none"): "amgtp_only", ("amgtp", "apops"): "ttam",
}
PRETTY = {
    "expanding": "Expanding", "best_fixed_window": "Best Fixed Window", "arw": "ARW",
    "adamoe": "AdaMoE", "ops": "OPS", "ttam": "TTAM",
    "amgtp_only": "AMG-TP only", "expanding_apops": "Expanding + AP-OPS",
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
            if isinstance(sw, pd.Series):        # one impression-weighted scalar per method
                if m in sw.index:
                    row[f"appendix_{sname}"] = float(sw[m])
                    row[f"appendix_{sname}_ttam"] = float(sw[TTAM])
            elif m in sw.columns:                # equal-day-averaged per-day frame (Brier, ECE)
                row[f"appendix_{sname}"] = float(sw[m].to_numpy().mean())
                row[f"appendix_{sname}_ttam"] = float(sw[TTAM].to_numpy().mean())
        rows.append(row)
    return pd.DataFrame(rows)


# The four 2x2 marginal contrasts (a - b, `a` better when negative): each
# factor's effect at both levels of the other factor.
CONTRASTS = [
    ("expanding_apops", "expanding"),   # AP-OPS effect | historical = expanding
    ("ttam", "amgtp_only"),             # AP-OPS effect | historical = AMG-TP
    ("amgtp_only", "expanding"),        # AMG-TP effect | calibration = none
    ("ttam", "expanding_apops"),        # AMG-TP effect | calibration = AP-OPS
]


def factorial_2x2(wide_ll: pd.DataFrame) -> dict:
    """The 2x2 ablation as marginal effects + interaction, all with a
    paired day-bootstrap CI (seeds already averaged into `wide_ll`).

    cells:  e = expanding, ea = expanding+AP-OPS, a = amgtp_only, t = ttam
    AP-OPS effect  given expanding = ea - e ;  given AMG-TP = t - a
    AMG-TP effect  given none      = a  - e ;  given AP-OPS = t - ea
    interaction    = (t - a) - (ea - e)   [ == (t - ea) - (a - e) ]
      < 0 : AMG-TP makes AP-OPS help more (synergy)
      ~ 0 : the two timescales are additive
    """
    need = ["expanding", "expanding_apops", "amgtp_only", "ttam"]
    if any(c not in wide_ll.columns for c in need):
        return {}
    e, ea = wide_ll["expanding"].to_numpy(), wide_ll["expanding_apops"].to_numpy()
    a, t = wide_ll["amgtp_only"].to_numpy(), wide_ll["ttam"].to_numpy()
    D = len(e)

    def ci(vec):
        rng = np.random.default_rng(RESAMPLE_SEED)
        lo, hi = _boot_ci(vec, rng)
        return {"mean": float(vec.mean()), "ci95": [lo, hi],
                "neg_days": int(np.sum(vec < 0)), "D": D, "excl_0": bool(lo * hi > 0)}

    return {
        "cell_scores": {"expanding": float(e.mean()), "expanding_apops": float(ea.mean()),
                        "amgtp_only": float(a.mean()), "ttam": float(t.mean())},
        "apops_given_expanding": ci(ea - e),
        "apops_given_amgtp": ci(t - a),
        "amgtp_given_none": ci(a - e),
        "amgtp_given_apops": ci(t - ea),
        "interaction": ci((t - a) - (ea - e)),
    }


def contrast_table(wide_ll: pd.DataFrame, pairs: list) -> pd.DataFrame:
    """Paired day-bootstrap for arbitrary method pairs (a - b): same
    seeds-averaged-first, resample-the-D-days machinery as `paired_table`,
    just not anchored on TTAM. Positive mean => `a` has the higher loss."""
    days = list(wide_ll.index)
    D = len(days)
    rows = []
    for a, b in pairs:
        if a not in wide_ll.columns or b not in wide_ll.columns:
            continue
        diff = wide_ll[a].to_numpy() - wide_ll[b].to_numpy()
        rng = np.random.default_rng(RESAMPLE_SEED)
        lo, hi = _boot_ci(diff, rng)
        mlo, mhi = _mbb_ci(diff, rng)
        rows.append({
            "a": PRETTY.get(a, a), "b": PRETTY.get(b, b),
            "mean_diff (a - b)": float(diff.mean()),
            "ci95_lo": lo, "ci95_hi": hi, "mbb_ci95_lo": mlo, "mbb_ci95_hi": mhi,
            "a_better_days": int(np.sum(diff < 0)), "D": D,
            "ci_excludes_0": bool(lo * hi > 0),
        })
    return pd.DataFrame(rows)


def analyse(result_dir: Path, label: str) -> dict:
    raw = load_per_seed(result_dir)
    wide_ll = seed_average(raw, "log_loss")

    # appendix impression-weighted log loss: ONE scalar per method =
    #   sum over every (day, seed) cell of (mean_loss * n)  /  sum of n
    # (the old code impression-weighted within a day then averaged days
    #  equally -- double-counting the equal-day weighting).
    raw["ll_sum"] = raw["log_loss"] * raw["n"]
    iw = raw.groupby("method").apply(
        lambda g: g["ll_sum"].sum() / g["n"].sum(), include_groups=False)
    secondary = {"iw_log_loss": iw}
    for col in ("brier", "ece"):     # secondary appendix metrics: seed-averaged, equal-day
        if col in raw.columns:
            secondary[col] = seed_average(raw, col)

    main = paired_table(wide_ll, MAIN_METHODS, secondary)
    ablation = paired_table(wide_ll, ABLATION_METHODS, secondary)
    contrasts = contrast_table(wide_ll, CONTRASTS)
    factorial = factorial_2x2(wide_ll)
    return {
        "label": label, "dir": str(result_dir),
        "days": [int(d) for d in wide_ll.index],
        "main": main, "ablation": ablation, "contrasts": contrasts, "factorial": factorial,
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
        f = a.get("factorial") or {}
        if f:
            cs = f["cell_scores"]
            L += ["", "### 6.3 Ablation -- 2x2: historical predictor x calibration", "",
                  "Mean score A_m per cell (lower = better):", "",
                  "| historical \\ calibration | none | AP-OPS |",
                  "|---|---|---|",
                  f"| **expanding history** | {cs['expanding']:.6f} | {cs['expanding_apops']:.6f} |",
                  f"| **AMG-TP** | {cs['amgtp_only']:.6f} | {cs['ttam']:.6f} (= TTAM) |",
                  "",
                  "Marginal effects (mean difference, negative => the added component lowers loss;",
                  "95% paired day-bootstrap CI; \"excl. 0\" = interval does not contain zero):", "",
                  "| effect | at | mean | 95% day-bootstrap CI | lower on | excl. 0 |",
                  "|---|---|---|---|---|---|"]
            def frow(name, at, c):
                return (f"| {name} | {at} | {c['mean']:+.2e} | "
                        f"[{c['ci95'][0]:+.2e}, {c['ci95'][1]:+.2e}] | "
                        f"{c['neg_days']}/{c['D']} | {'yes' if c['excl_0'] else 'no'} |")
            L += [frow("AP-OPS (calibration)", "historical = expanding", f["apops_given_expanding"]),
                  frow("AP-OPS (calibration)", "historical = AMG-TP", f["apops_given_amgtp"]),
                  frow("AMG-TP (historical)", "calibration = none", f["amgtp_given_none"]),
                  frow("AMG-TP (historical)", "calibration = AP-OPS", f["amgtp_given_apops"]),
                  frow("interaction", "(t-a) - (ea-e)", f["interaction"]),
                  "",
                  "Interaction < 0 would mean AMG-TP makes AP-OPS help more (synergy); "
                  "a CI containing 0 means the two timescales are additive.", ""]
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
        a["contrasts"].to_csv(Path(a["dir"]) / "section6_contrasts.csv", index=False)
        (Path(a["dir"]) / "section6_factorial.json").write_text(
            json.dumps(a["factorial"], indent=2, default=float))
        print(f"wrote {a['dir']}/section6_{{main,ablation,contrasts}}.csv + section6_factorial.json")

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
