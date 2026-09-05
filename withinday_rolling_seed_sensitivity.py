"""Model-initialization-seed sensitivity check (rolling-protocol section 2:
"distinguish data-sampling seeds from model-initialization seeds ... seeds
should be used to assess algorithmic sensitivity, not to increase the
claimed temporal sample size").

Reads the *already-chosen* per-outer-day V5 hyperparameters from a
completed ``withinday_rolling_run.py`` run's ``rolling_origin_manifest.csv``
(same data-sampling seed, same train-day windows -- nothing about the
nested hyperparameter search is redone here) and retrains just the final
model at each outer day under a few different ``torch`` initialization
seeds, to see how much of the day's outcome is attributable to model
initialization noise rather than genuine signal. Reported as a spread
per day, never pooled into the day count.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from twoscale.calib import CalibConfig
from twoscale.data import load
from twoscale.longterm import adaptive_weights, build_bank, long_term_predictions
from twoscale.methods import build_suite
from twoscale.metrics import day_logloss
from twoscale_run import DATA_PATHS

from withinday.blocks import summary_dim, token_dim
from withinday.cache import build_cache
from withinday.rolling import _split_train_dev
from withinday.train import DEFAULT_CFG, predict_records, train_variant
from withinday_rolling_run import CALIB_CONFIG, FROZEN_BLOCK_SEC, FROZEN_DELAY_SEC, FROZEN_M, OUTER_DAYS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["criteo", "avazu"], required=True)
    ap.add_argument("--manifest", required=True, help="rolling_origin_manifest.csv from the main run")
    ap.add_argument("--data", default=None)
    ap.add_argument("--sample-frac", type=float, default=None)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-features", type=int, default=2 ** 18)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    sample_frac = args.sample_frac if args.sample_frac is not None else (1.0 if args.source == "criteo" else 0.2)

    path = args.data or DATA_PATHS[args.source]
    ds = load(args.source, path, n_features=args.n_features, sample_frac=sample_frac, seed=args.data_seed)
    outer_days = OUTER_DAYS[args.source]
    eval_days = list(range(1, max(outer_days) + 1))
    bank = build_bank(ds, eval_days, seed=args.data_seed, n_jobs=args.n_jobs)
    eval_days = sorted(bank)
    w = adaptive_weights(bank, eval_days)
    q_by_day = long_term_predictions(bank, eval_days, "adaptive", weights=w)
    methods, _, _ = build_suite(bank, eval_days, CalibConfig(**CALIB_CONFIG), include_platt=True)
    cache = build_cache(ds, bank, q_by_day, eval_days, block_sec=FROZEN_BLOCK_SEC,
                        delay_sec=FROZEN_DELAY_SEC, m=FROZEN_M, seed=args.data_seed)
    a_dim, tok_dim, summ_dim = FROZEN_M + 2, token_dim(FROZEN_M), summary_dim(FROZEN_M)
    long_by_day = {r["day"]: r for r in methods["long_only"]}

    rows = []
    for _, row in manifest.iterrows():
        d = int(row["outer_test_day"])
        overlay = {"cross_dim": int(row["chosen_cross_dim"]), "lr": float(row["chosen_lr"]),
                  "weight_decay": float(row["chosen_weight_decay"])}
        candidate_days = sorted(e for e in cache if e < d)
        adtr, addev = _split_train_dev(candidate_days)
        lls = []
        for model_seed in args.model_seeds:
            cfg = {**DEFAULT_CFG, **overlay, "seed": model_seed}
            model, _ = train_variant("v5_linear", [cache[e] for e in adtr], [cache[e] for e in addev],
                                     a_dim, tok_dim, summ_dim, cfg=cfg)
            recs = predict_records("v5_linear", model, [cache[d]], K=cfg["K"])
            ll = day_logloss(cache[d].y, recs[0]["p"])
            lls.append(ll)
            rows.append({"dataset": args.source, "day": d, "model_seed": model_seed,
                        "log_loss": ll, "delta_vs_long_only": ll - day_logloss(long_by_day[d]["y"], long_by_day[d]["p"])})
        print(f"day {d}: log_loss range across model seeds {args.model_seeds} = "
              f"[{min(lls):.6f}, {max(lls):.6f}], spread={max(lls) - min(lls):.6f}", flush=True)

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    spread = df.groupby("day")["log_loss"].agg(lambda s: s.max() - s.min())
    print(f"\nmax spread across days: {spread.max():.6f}  mean spread: {spread.mean():.6f}", flush=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
