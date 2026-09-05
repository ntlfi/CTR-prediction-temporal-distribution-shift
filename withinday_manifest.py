"""Day-level data-use manifest (rolling-protocol section 4 / output 1).

Classifies every calendar day of both datasets by how it has actually been
used so far, using the exact splits every prior run in this repo used
(``twoscale.splits.make_split``). See
``withinday_experiments/ROLLING_PROTOCOL_FREEZE.md`` for why both datasets
come back with zero "genuinely untouched" days, decided *before* this
script inspects any per-day loss numbers -- this script only classifies
role and reports impressions/CTR, it makes no modeling decision.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from twoscale.data import load
from twoscale.splits import make_split
from twoscale_run import DATA_PATHS

# rolling-origin outer test days actually used by withinday_rolling_run.py,
# frozen in ROLLING_PROTOCOL_FREEZE.md -- recorded here so the manifest
# shows which days get a second look, without this script computing any
# new day-level losses itself.
ROLLING_OUTER_DAYS = {"criteo": list(range(16, 22)), "avazu": list(range(5, 10))}


def classify(day: int, split) -> str:
    if day in split.train_days:
        return "base_model_training"
    if day in split.test_days:
        return "original_locked_test"
    if day in split.dev_days:
        return "adapter_training_and_hyperparameter_validation"
    return "genuinely_untouched"


def build_manifest(source: str, data_path: str, sample_frac: float, warmup: int,
                   n_features: int, seed: int) -> pd.DataFrame:
    ds = load(source, data_path, n_features=n_features, sample_frac=sample_frac, seed=seed)
    split = make_split(ds.n_days, warmup=warmup)
    rows = []
    for d in range(ds.n_days):
        sl = ds.day_slice(d)
        y = ds.y[sl]
        role = classify(d, split)
        rows.append({
            "dataset": source,
            "day_index": d,
            "role": role,
            "previously_inspected": role != "genuinely_untouched",
            "reused_in_rolling_analysis": d in ROLLING_OUTER_DAYS[source],
            "n_impressions": int(len(y)),
            "empirical_ctr": float(y.mean()) if len(y) else float("nan"),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="withinday_experiments/day_usage_manifest.csv")
    ap.add_argument("--avazu-sample-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frames = [
        build_manifest("criteo", DATA_PATHS["criteo"], sample_frac=1.0, warmup=4,
                       n_features=2 ** 18, seed=args.seed),
        build_manifest("avazu", DATA_PATHS["avazu"], sample_frac=args.avazu_sample_frac,
                       warmup=3, n_features=2 ** 18, seed=args.seed),
    ]
    df = pd.concat(frames, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(df.to_string(index=False), flush=True)
    for source in ("criteo", "avazu"):
        sub = df[df["dataset"] == source]
        n_untouched = int((sub["role"] == "genuinely_untouched").sum())
        print(f"\n{source}: {len(sub)} days total, {n_untouched} genuinely untouched", flush=True)


if __name__ == "__main__":
    main()
