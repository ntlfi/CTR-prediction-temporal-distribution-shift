"""Hyperparameter selection (plan section 6) -- dev days only.

Keeps the search intentionally small. Fits the long-term bank once, then
sweeps the calibrator grid, scoring the *combined* method's
impression-weighted log loss on the development days. Picks the configuration
with the lowest mean dev loss, with the plan's stability tie-breaker
(smaller B, fewer updates, smaller worst-day loss). Writes ``FROZEN.json``.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from twoscale.calib import CalibConfig
from twoscale.data import load
from twoscale.longterm import adaptive_weights, build_bank, long_term_predictions
from twoscale.methods import _calibrated
from twoscale.metrics import day_logloss, impression_weighted_logloss
from twoscale.splits import make_split
from twoscale_run import DATA_PATHS

GRID = dict(
    B=[0.25, 0.5, 1.0, 2.0],
    eta0=[0.01, 0.03, 0.1, 0.3],
    eta_schedule=["inv_sqrt", "const"],
    # per-impression updates are covered by the section-9 ablations, not the
    # HPO grid (11M-row python loops per dev cell are too slow for a 100-cell
    # grid); the block sizes span the same speed/latency trade-off.
    update=[("block", 900), ("block", 1800), ("block", 3600)],
    eps=[1e-5],
    delay_sec=[1800],           # main setting; sensitivity handled in ablations
)

# Long-term adaptive-mixture grid (plan section 2.1): tuned separately on the
# uncalibrated long_only dev loss before the calibrator grid.
MIX_GRID = dict(eta=[10.0, 30.0, 60.0, 150.0, 1e6], halflife=[3.0, 5.0, 10.0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["criteo", "avazu"], default="criteo")
    ap.add_argument("--data", default=None)
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-features", type=int, default=2 ** 18)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--mixture-eta", type=float, default=60.0)
    ap.add_argument("--mixture-halflife", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = load(args.source, args.data or DATA_PATHS[args.source],
              n_features=args.n_features, sample_frac=args.sample_frac, seed=args.seed)
    split = make_split(ds.n_days, warmup=args.warmup)
    dev_days = set(int(d) for d in split.dev_days)
    print(f"{ds.name}: {ds.n_days} days, dev {sorted(dev_days)}", flush=True)

    # bank only needs the dev days (+ their history is days < d, always available)
    bank = build_bank(ds, split.dev_days, seed=args.seed, n_jobs=args.n_jobs)
    eval_days = sorted(bank)

    # --- stage 1: long-term adaptive mixture (uncalibrated long_only dev loss) ---
    from twoscale.methods import _uncalibrated
    mix_rows = []
    for me in MIX_GRID["eta"]:
        for mh in MIX_GRID["halflife"]:
            w = adaptive_weights(bank, eval_days, eta=me, halflife=mh)
            q = long_term_predictions(bank, eval_days, "adaptive", weights=w)
            recs = _uncalibrated(q, bank, eval_days)
            dev = [r for r in recs if r["day"] in dev_days]
            mix_rows.append({"mix_eta": me, "mix_halflife": mh,
                             "dev_imp_wt_ll": impression_weighted_logloss(dev)})
    mix_df = pd.DataFrame(mix_rows).sort_values("dev_imp_wt_ll").reset_index(drop=True)
    mix_df.to_csv(out / "hpo_mixture.csv", index=False)
    best_mix = mix_df.iloc[0]
    mix_eta, mix_halflife = float(best_mix["mix_eta"]), float(best_mix["mix_halflife"])
    print(f"\nbest mixture: eta={mix_eta:g} halflife={mix_halflife:g} "
          f"(dev ll {best_mix['dev_imp_wt_ll']:.5f})", flush=True)

    # --- stage 2: calibrator grid (combined dev loss with the chosen mixture) ---
    w = adaptive_weights(bank, eval_days, eta=mix_eta, halflife=mix_halflife)
    q_adaptive = long_term_predictions(bank, eval_days, "adaptive", weights=w)

    rows = []
    combos = list(itertools.product(GRID["B"], GRID["eta0"], GRID["eta_schedule"],
                                    GRID["update"], GRID["eps"], GRID["delay_sec"]))
    for B, eta0, sched, (upd, blk), eps, delay in combos:
        cfg = CalibConfig(B=B, eta0=eta0, eta_schedule=sched, update=upd,
                          block_sec=blk or 1800, delay_sec=delay, eps=eps)
        recs, _ = _calibrated(q_adaptive, bank, eval_days, cfg)
        dev = [r for r in recs if r["day"] in dev_days]
        iwll = impression_weighted_logloss(dev)
        worst = max(day_logloss(r["y"], r["p"]) for r in dev)
        n_updates = {"impression": 3, "block": {900: 2, 1800: 1, 3600: 0}.get(blk, 1)}[upd]
        rows.append({"B": B, "eta0": eta0, "eta_schedule": sched, "update": upd,
                     "block_sec": blk or 0, "delay_sec": delay, "eps": eps,
                     "dev_imp_wt_ll": iwll, "dev_worst_day_ll": worst,
                     "tiebreak": (round(iwll, 6), n_updates, B, worst)})
    df = pd.DataFrame(rows).sort_values("dev_imp_wt_ll").reset_index(drop=True)
    df.drop(columns="tiebreak").to_csv(out / "hpo_grid.csv", index=False)

    best = min(rows, key=lambda r: r["tiebreak"])
    frozen = dict(B=best["B"], eta0=best["eta0"], eta_schedule=best["eta_schedule"],
                  update=best["update"], block_sec=best["block_sec"] or 1800,
                  delay_sec=best["delay_sec"], eps=best["eps"],
                  init_b=0.0, carryover_rho=0.0,
                  mixture_eta=mix_eta, mixture_halflife=mix_halflife)
    (out / "FROZEN.json").write_text(json.dumps(frozen, indent=2))
    print("\n=== dev grid (top 10) ===", flush=True)
    print(df.drop(columns="tiebreak").head(10).to_string(index=False), flush=True)
    print(f"\nFROZEN -> {out}/FROZEN.json\n{json.dumps(frozen, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
