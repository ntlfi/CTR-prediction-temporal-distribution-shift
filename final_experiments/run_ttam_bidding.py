"""Downstream bidding replay for the TTAM revised Section 6 (plan section
10).

The nested run's frozen origin-day predictions
(``<nested_out>/final_predictions/origin{d}_seed{k}.npz``) are each fed
into the **same** auction and pacing policy; only the prediction input
changes.  The main table compares TTAM with the same five headline
baselines (Expanding, Best Fixed Window, ARW, AdaMoE, OPS).

Criteo is the primary downstream experiment: the recorded per-impression
display ``cost`` is used as the price proxy the replay model needs.  It is
the price that had to be beaten to win that logged impression; we do
**not** treat it as an observed clearing price.  Auction rule: an
advertiser bidding ``b_i = scale * pctr_i`` wins impression ``i`` iff
``b_i >= cost_i``, pays ``cost_i`` on a win, and receives the logged
``click_i``.

Budget: a per-origin daily budget of 25% of that origin day's fixed
reference cost (the sum of ``cost`` over all its eligible impressions);
budgets reset daily and are identical across the paired methods for a
given origin.  Bid multipliers are taken from a predeclared geometric
grid; if a method's realised spend differs from the 25% target we
interpolate its clicks to the common spend using its own value-vs-spend
frontier, bracketing on **spend only** (never on test clicks), and label
the result interpolated.

Uncertainty: days are paired after averaging training seeds; the
aggregate click-gain ratio of TTAM over each baseline is recomputed in
every one of 10,000 paired day resamples (resampling seed 20260908),
matching the prediction-side statistics.

Avazu: no recorded price field is available and any generated price model
would be a simulation that must be frozen on past data; per the plan this
is left **pending** and Criteo is reported as the primary downstream
experiment.

    PYTHONPATH=. python3 final_experiments/run_ttam_bidding.py \
        --nested-out final_experiments/ttam/criteo/nested \
        --data data/criteo_attribution_dataset.tsv.gz \
        --out final_experiments/ttam/criteo/bidding
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from autobid import linear_frontier, paced_auction, value_at_matched_spend
from ttam_stats import RESAMPLE_SEED, N_BOOT, PRETTY

SECONDS_PER_DAY = 86_400
SEEDS = (0, 1, 2)
MAIN_METHODS = ["expanding", "best_fixed_window", "arw", "adamoe", "ops", "ttam"]
BUDGET_FRACS = [0.10, 0.25, 0.50, 0.75]
PRIMARY_FRAC = 0.25
N_SCALES = 80


def load_criteo_cost_by_day(tsv_path: str) -> dict:
    """``{day: (click, cost)}`` with rows in the exact (day, sec_in_day)
    stable-sorted order ``twoscale.data.load_criteo`` produces at
    sample_frac=1.0, so ``cost`` aligns row-for-row with the nested run's
    saved per-origin arrays."""
    df = pd.read_csv(tsv_path, sep="\t", usecols=["timestamp", "click", "cost"])
    df["day"] = (df["timestamp"] // SECONDS_PER_DAY).astype(np.int64)
    df["sec_in_day"] = (df["timestamp"] % SECONDS_PER_DAY).astype(np.int64)
    df = df.sort_values(["day", "sec_in_day"], kind="stable").reset_index(drop=True)
    out = {}
    for d, g in df.groupby("day"):
        out[int(d)] = (g["click"].to_numpy(np.int64), g["cost"].to_numpy(np.float64))
    return out


def clicks_at_spend(pctr, click, cost, target_spend: float) -> tuple:
    """Interpolate clicks won to ``target_spend`` on the method's own
    global-scale frontier; bracket on spend only. Returns
    ``(clicks_interp, bracket_lo_spend, bracket_hi_spend)``."""
    fr = linear_frontier(pctr, click, cost, np.zeros(len(pctr)), n_scales=N_SCALES).sort_values("spend")
    sp = fr["spend"].to_numpy()
    cl = fr["clicks"].to_numpy()
    if target_spend <= sp[0]:
        return float(cl[0]), float(sp[0]), float(sp[0])
    if target_spend >= sp[-1]:
        return float(cl[-1]), float(sp[-1]), float(sp[-1])
    j = int(np.searchsorted(sp, target_spend))
    return float(np.interp(target_spend, sp, cl)), float(sp[j - 1]), float(sp[j])


def analyse(nested_out: Path, cost_by_day: dict) -> dict:
    pred_dir = nested_out / "final_predictions"
    origins = sorted({int(p.stem.split("_")[0].replace("origin", ""))
                      for p in pred_dir.glob("origin*_seed*.npz")})
    per_cell = []           # one row per (origin, seed, method, budget_frac)
    for d in origins:
        _, cost = cost_by_day[d]
        ref_cost = float(cost.sum())
        for s in SEEDS:
            npz = pred_dir / f"origin{d}_seed{s}.npz"
            if not npz.exists():
                continue
            z = np.load(npz)
            y = z["y"].astype(np.int64)
            click_log, cost_d = cost_by_day[d]
            assert np.array_equal(y, click_log), f"origin {d} seed {s}: label/auction misalignment"
            for m in MAIN_METHODS:
                if m not in z:
                    continue
                p = z[m].astype(np.float64)
                for frac in BUDGET_FRACS:
                    cl, blo, bhi = clicks_at_spend(p, y, cost_d, frac * ref_cost)
                    # paced daily-budget realisation (secondary / spend-honest)
                    ps, _ = paced_auction(p, y, cost_d, np.zeros(len(p), int), frac * ref_cost)
                    per_cell.append({"origin": d, "seed": s, "method": m, "budget_frac": frac,
                                     "ref_cost": ref_cost, "clicks_matched": cl,
                                     "bracket_spend_lo": blo, "bracket_spend_hi": bhi,
                                     "paced_clicks": ps["clicks"], "paced_spend": ps["spend"]})
    return pd.DataFrame(per_cell), origins


def paired_bidding_table(cell: pd.DataFrame, frac: float) -> pd.DataFrame:
    sub = cell[cell.budget_frac == frac]
    # A_{m,d}: seed-averaged clicks at matched spend
    wide = (sub.groupby(["method", "origin"])["clicks_matched"].mean()
               .reset_index().pivot(index="origin", columns="method", values="clicks_matched")
               .sort_index())
    days = list(wide.index)
    a_ttam = wide["ttam"].to_numpy()
    rows = []
    for m in MAIN_METHODS:
        if m == "ttam" or m not in wide.columns:
            continue
        a_m = wide[m].to_numpy()
        gain = a_ttam - a_m                       # extra clicks for TTAM at matched spend
        ratio_per_day = a_ttam / np.maximum(a_m, 1e-9)
        rng = np.random.default_rng(RESAMPLE_SEED)
        idx = rng.integers(0, len(days), size=(N_BOOT, len(days)))
        # recompute the AGGREGATE click-gain ratio in every resample
        agg_ratio = a_ttam[idx].sum(axis=1) / np.maximum(a_m[idx].sum(axis=1), 1e-9)
        mean_gain_boot = gain[idx].mean(axis=1)
        rows.append({
            "method": PRETTY.get(m, m),
            "ttam_clicks": float(a_ttam.mean()), "baseline_clicks": float(a_m.mean()),
            "mean_extra_clicks": float(gain.mean()),
            "extra_clicks_ci95": [float(np.percentile(mean_gain_boot, 2.5)),
                                  float(np.percentile(mean_gain_boot, 97.5))],
            "aggregate_click_gain_pct": float((a_ttam.sum() / a_m.sum() - 1.0) * 100),
            "click_gain_pct_ci95": [float((np.percentile(agg_ratio, 2.5) - 1) * 100),
                                    float((np.percentile(agg_ratio, 97.5) - 1) * 100)],
            "origins_won_by_ttam": int(np.sum(gain > 0)), "D": len(days),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nested-out", required=True, help="the criteo nested run dir")
    ap.add_argument("--data", required=True, help="criteo tsv.gz (recorded cost field)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nested_out = Path(args.nested_out)

    pred_dir = nested_out / "final_predictions"
    if not any(pred_dir.glob("origin*_seed*.npz")):
        print(f"no per-origin prediction dumps under {pred_dir} -- nothing to replay.\n"
              f"(they are gitignored; regenerate by re-running run_ttam_nested.py, or use the\n"
              f" committed {out}/ tables if this replay was already done.)")
        return

    cost_by_day = load_criteo_cost_by_day(args.data)
    cell, origins = analyse(nested_out, cost_by_day)
    cell.to_csv(out / "bidding_cells.csv", index=False)

    tables = {}
    for frac in BUDGET_FRACS:
        t = paired_bidding_table(cell, frac)
        t.to_csv(out / f"bidding_matched_{int(frac * 100)}pct.csv", index=False)
        tables[f"{int(frac * 100)}pct"] = t.to_dict("records")

    primary = paired_bidding_table(cell, PRIMARY_FRAC)
    (out / "summary.json").write_text(json.dumps({
        "source": "criteo", "primary_budget_frac": PRIMARY_FRAC,
        "reference_cost": "sum of recorded display cost over each origin day's impressions",
        "budgets_reset": "daily; identical across paired methods per origin",
        "auction_rule": "b_i = scale * pctr_i ; win iff b_i >= cost_i ; pay cost_i on win",
        "interpolation": "clicks interpolated to the common spend on each method's own "
                         "global-scale frontier; bracketed on spend only; never on test clicks",
        "origins": origins, "resample_seed": RESAMPLE_SEED, "n_boot": N_BOOT,
        "main_table_primary": primary.to_dict("records"), "appendix_tables": tables,
        "avazu": "pending -- no recorded price field; a simulated price model would need a "
                 "separate frozen specification (plan section 10).",
    }, indent=2, default=float))

    print("\n=== TTAM downstream bidding (Criteo, 25% daily budget, matched spend) ===")
    print(primary.to_string(index=False))
    print(f"\n-> {out}/")


if __name__ == "__main__":
    main()
