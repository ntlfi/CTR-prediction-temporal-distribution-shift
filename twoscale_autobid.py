"""Downstream matched-budget autobidding evaluation (plan section 7.3 / step 9).

Rebuilds the two-timescale method suite on Criteo with the frozen config,
then feeds each method's locked-test-day pCTRs into one fixed auction +
pacing policy on the logged Criteo auctions. Primary metric: clicks won at
matched spend.

Run only after the prediction result is in (step 9 is gated on step 8).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from twoscale.autobid import (linear_frontier, load_criteo_bidding, paced_auction,
                              value_at_matched_spend)
from twoscale.calib import CalibConfig
from twoscale.data import load_criteo
from twoscale.longterm import build_bank
from twoscale.methods import build_suite
from twoscale.splits import make_split
from twoscale_run import DATA_PATHS

BUDGET_FRACS = [0.1, 0.25, 0.5]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=DATA_PATHS["criteo"])
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-features", type=int, default=2 ** 18)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    cfg_d = json.loads(Path(args.config).read_text())
    cfg = CalibConfig(**{k: v for k, v in cfg_d.items() if k in CalibConfig.__annotations__})

    ds = load_criteo(args.data, n_features=args.n_features, sample_frac=args.sample_frac, seed=args.seed)
    split = make_split(ds.n_days, warmup=args.warmup)
    test_days = set(int(d) for d in split.test_days)
    bank = build_bank(ds, split.eval_days, seed=args.seed, n_jobs=args.n_jobs)
    methods, _, _ = build_suite(bank, sorted(bank), cfg,
                                mixture_eta=cfg_d.get("mixture_eta", 60.0),
                                mixture_halflife=cfg_d.get("mixture_halflife", 5.0))

    bd = load_criteo_bidding(args.data, n_features=args.n_features,
                             sample_frac=args.sample_frac, seed=args.seed)
    # test-day slice of the bidding log, in the same (day, sec_in_day) order
    mask = np.isin(bd.day, sorted(test_days))
    cost, click, conv, day = bd.cost[mask], bd.y[mask], bd.conversion[mask], bd.day[mask]

    # assemble each method's pctr over the test rows, aligned to bd order
    pctr = {}
    for name, recs in methods.items():
        by_day = {r["day"]: r["p"] for r in recs}
        pctr[name] = np.concatenate([by_day[d] for d in sorted(test_days) if d in by_day])
    # reference bidders
    pctr["_oracle"] = click.astype(float)
    pctr["_noskill"] = np.full(len(click), float(click.mean()))

    assert all(len(v) == len(cost) for v in pctr.values()), \
        {k: len(v) for k, v in pctr.items()} | {"cost": len(cost)}

    frontiers = {n: linear_frontier(p, click, cost, conv) for n, p in pctr.items()}
    for n, fr in frontiers.items():
        fr.insert(0, "method", n)
    pd.concat(frontiers.values()).to_csv(out / "frontiers.csv", index=False)

    total = cost.sum()
    grid = np.linspace(0.05 * total, 0.8 * total, 25)
    vms = value_at_matched_spend(frontiers, grid, "clicks")
    vms.to_csv(out / "clicks_at_matched_spend.csv", index=False)

    paced = []
    for bf in BUDGET_FRACS:
        for n, p in pctr.items():
            r = paced_auction(p, click, cost, day, budget=bf * total)
            paced.append({"method": n, "budget_frac": bf, **r})
    paced_df = pd.DataFrame(paced)
    paced_df.to_csv(out / "paced.csv", index=False)

    # headline: paced clicks vs long_only / short_only at each budget
    summ = {"config": {"seed": args.seed, "sample_frac": args.sample_frac,
                       "test_days": sorted(test_days), "total_logged_spend": float(total)},
            "paced_clicks": {}}
    piv = paced_df.pivot(index="budget_frac", columns="method", values="clicks")
    for bf in BUDGET_FRACS:
        row = piv.loc[bf]
        summ["paced_clicks"][str(bf)] = {
            m: {"clicks": int(row[m]),
                "vs_long_only_pct": float(100 * (row[m] - row["long_only"]) / row["long_only"]),
                "vs_short_only_pct": float(100 * (row[m] - row["short_only"]) / row["short_only"])}
            for m in piv.columns}
    (out / "summary.json").write_text(json.dumps(summ, indent=2, default=float))
    print(paced_df.pivot(index="budget_frac", columns="method", values="clicks").to_string())
    print(f"\n-> {out}/")


if __name__ == "__main__":
    main()
