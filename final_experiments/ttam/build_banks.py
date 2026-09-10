"""Build and cache the shared five-horizon expert banks used by the TTAM
nested runner.  Each bank is the full rolling-origin structure for one
(source, seed): ``bank[d]`` holds the predictions of every distinct
effective expert for day ``d`` from a model fitted on days ``< d``.

Cached as ``<cache_dir>/<source>_seed<k>_nf<...>_sf<...>.npz`` so the
three nested-runner passes (module selection / calibration selection /
final scoring) each load the bank from disk instead of refitting it.

    PYTHONPATH=. python3 final_experiments/ttam/build_banks.py \
        --source criteo --data data/criteo_attribution_dataset.tsv.gz \
        --cache-dir final_experiments/ttam/_bankcache --n-jobs 3
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from twoscale.data import load
from final_experiments.ttam.bank import (BACKBONE_ALPHA, DayBank5, HORIZONS5,
                                         build_bank5)
from final_experiments.ttam.amgtp import CONTEXT_M
from final_experiments.ttam.maturity import DELAY_SEC
from withinday.contextsketch import build_projection, context_sketch

SEEDS = (0, 1, 2)
MAX_DAY = {"criteo": 31, "avazu": 10}


def cache_path(cache_dir: Path, source: str, seed: int, n_features: int, sample_frac: float) -> Path:
    # `_d<delay>` tags the label-maturity rule -- old (leaky) caches without
    # it are simply not found and get rebuilt.
    return cache_dir / f"{source}_seed{seed}_nf{n_features}_sf{sample_frac:g}_d{DELAY_SEC}.npz"


def context_path(cache_dir: Path, source: str, seed: int, n_features: int, sample_frac: float) -> Path:
    return cache_dir / f"{source}_seed{seed}_nf{n_features}_sf{sample_frac:g}_ctx{CONTEXT_M}.npz"


def save_context(path: Path, ds, days) -> None:
    """Fixed seeded signed-hash context sketch (L2-normalised) per day --
    the per-example gate context AMG-TP uses, cached so the nested runner
    never reloads the raw hashed matrix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    R = build_projection(ds.X.shape[1], CONTEXT_M, seed=0)   # shared across seeds/days
    store = {"days": np.array(sorted(days), dtype=np.int64)}
    for d in sorted(days):
        sl = ds.day_slice(d)
        if sl.stop <= sl.start:
            continue
        store[f"c_{d}"] = context_sketch(ds.X[sl], R=R).astype(np.float32)
    np.savez_compressed(path, **store)


def load_context(path: Path) -> dict:
    # kept float32 in-memory (the sketch is a projected feature summary; AMG-TP
    # upcasts one day at a time to float64 in run_amgtp5). float32 halves the
    # resident context so the nested runner can fan origins out to more workers.
    z = np.load(path)
    return {int(d): z[f"c_{d}"].astype(np.float32) for d in z["days"] if f"c_{d}" in z}


def save_bank(path: Path, bank: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    days = sorted(bank)
    store = {"days": np.array(days, dtype=np.int64)}
    for d in days:
        db = bank[d]
        store[f"y_{d}"] = np.asarray(db.y, dtype=np.int8)
        store[f"sec_{d}"] = np.asarray(db.sec_in_day, dtype=np.int64)
        for h in HORIZONS5:
            store[f"p_{d}_{h}"] = np.asarray(db.preds[h], dtype=np.float32)
        store[f"eff_{d}"] = np.array([db.effective[h] for h in HORIZONS5], dtype=np.int8)
        store[f"nt_{d}"] = np.array([db.n_train[h] for h in HORIZONS5], dtype=np.int64)
    np.savez_compressed(path, **store)


def load_bank(path: Path) -> dict:
    z = np.load(path)
    days = [int(d) for d in z["days"]]
    bank = {}
    for d in days:
        eff = z[f"eff_{d}"]
        bank[d] = DayBank5(
            d=d, y=z[f"y_{d}"], sec_in_day=z[f"sec_{d}"],
            preds={h: z[f"p_{d}_{h}"].astype(np.float64) for h in HORIZONS5},
            effective={h: int(eff[i]) for i, h in enumerate(HORIZONS5)},
            n_train={h: int(z[f"nt_{d}"][i]) for i, h in enumerate(HORIZONS5)},
            n_effective=int(len(set(eff.tolist()))))
    return bank


def load_or_build(source: str, data_path: str, seed: int, n_features: int,
                  sample_frac: float, cache_dir: Path, n_jobs: int, verbose: bool = False,
                  want_context: bool = True):
    """Returns ``(bank, context_by_day_or_None)``. Both are cached; the raw
    hashed matrix is loaded only on a bank cache miss."""
    bpath = cache_path(cache_dir, source, seed, n_features, sample_frac)
    cpath = context_path(cache_dir, source, seed, n_features, sample_frac)
    if bpath.exists() and (cpath.exists() or not want_context):
        print(f"  bank cache hit: {bpath}", flush=True)
        bank = load_bank(bpath)
        ctx = load_context(cpath) if (want_context and cpath.exists()) else None
        return bank, ctx
    t0 = time.time()
    ds = load(source, data_path, n_features=n_features, sample_frac=sample_frac, seed=seed)
    print(f"  seed {seed}: loaded {len(ds.y):,} rows / {ds.n_days} days "
          f"in {time.time() - t0:.0f}s", flush=True)
    days = list(range(1, min(MAX_DAY[source], ds.n_days)))
    if bpath.exists():
        bank = load_bank(bpath)
    else:
        bank = build_bank5(ds, days, alpha=BACKBONE_ALPHA, seed=seed, n_jobs=n_jobs, verbose=verbose)
        save_bank(bpath, bank)
        print(f"  seed {seed}: bank cached -> {bpath}", flush=True)
    ctx = None
    if want_context:
        if not cpath.exists():
            save_context(cpath, ds, days)
            print(f"  seed {seed}: context cached -> {cpath}", flush=True)
        ctx = load_context(cpath)
    print(f"  seed {seed}: ready ({time.time() - t0:.0f}s total)", flush=True)
    del ds
    return bank, ctx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["criteo", "avazu"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-features", type=int, default=2 ** 18)
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--cache-dir", default="final_experiments/ttam/_bankcache")
    ap.add_argument("--n-jobs", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()
    for seed in args.seeds:
        b, c = load_or_build(args.source, args.data, seed, args.n_features, args.sample_frac,
                             Path(args.cache_dir), args.n_jobs, verbose=True)
        del b, c


if __name__ == "__main__":
    main()
