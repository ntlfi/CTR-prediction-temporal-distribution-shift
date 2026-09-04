"""Hyperparameter selection (plan section 5.2) -- development days only.

The plan's validation grid (context-sketch dim, hidden dim, lr, weight
decay, dropout, bilinear rank, correction cap) has ~200 combinations; at
neural-net training cost that full cross product per variant is not
affordable the way twoscale's cheap sklearn calibrator grid was. Instead
this is a staged coordinate search per variant, in the same spirit as
``twoscale_hpo.py``'s two-stage (mixture, then calibrator) search:

  1. pick the context-sketch dimension ``m`` once, with a single proxy
     variant (V3 MLP) at default hyperparameters -- ``m`` changes the cache
     itself (Stage A), so it is not something each variant can search
     independently without rebuilding the cache five times over;
  2. for each variant independently, holding ``m`` fixed: (a) a small
     learning-rate x weight-decay grid, (b) that variant's capacity knob
     (hidden dim for V1/V2, MLP width for V3, rank for V4, cross-feature
     width for V5), (c) dropout x correction-cap -- each stage fixing the
     winner of the previous one, so the total cost per variant is additive
     in grid size, not multiplicative.

Long-term backbone (the adaptive mixture) is *not* retuned here -- it is
frozen at the values [[twoscale]] already picked, per plan section 2.3's
identical-backbone rule. The bank is restricted to the development days,
matching ``twoscale_hpo.py``.

Writes ``FROZEN.json`` in the shape ``withinday_run.py --config`` expects:
``{"sketch_dim": m, "per_variant": {variant: {cfg overrides}}}``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from twoscale.data import load
from twoscale.longterm import adaptive_weights, build_bank, long_term_predictions
from twoscale.metrics import impression_weighted_logloss
from twoscale.splits import make_split
from twoscale_run import DATA_PATHS

from withinday.adapters import VARIANTS
from withinday.blocks import summary_dim, token_dim
from withinday.cache import build_cache
from withinday.train import DEFAULT_CFG, predict_records, train_variant

MIXTURE_ETA, MIXTURE_HALFLIFE = 60.0, 5.0   # frozen at twoscale's plan-default center

LR_WD_GRID = [(3e-4, 1e-5), (3e-4, 1e-4), (1e-3, 1e-5), (1e-3, 1e-4)]
DROPOUT_DELTAMAX_GRID = [(0.0, 1.0), (0.0, 2.0), (0.1, 1.0), (0.1, 2.0)]
CAPACITY_GRID = {
    "v1_transformer": [{"hidden": 16}, {"hidden": 32}, {"hidden": 64}],
    "v2_gru": [{"hidden": 16}, {"hidden": 32}, {"hidden": 64}],
    "v3_mlp": [{"mlp_hidden": (16, 16)}, {"mlp_hidden": (32, 16)}, {"mlp_hidden": (64, 32)}],
    "v4_bilinear": [{"rank": 4}, {"rank": 8}],
    "v5_linear": [{"cross_dim": 16}, {"cross_dim": 32}, {"cross_dim": 64}],
}
SKETCH_DIM_GRID = [32, 64]


def _dev_ll(name, model, caches_dev, cfg):
    recs = predict_records(name, model, caches_dev, K=cfg["K"])
    return impression_weighted_logloss(recs)


def _train_and_score(name, caches_train, caches_dev, a_dim, tok_dim, summ_dim, cfg, seed, verbose):
    model, _ = train_variant(name, caches_train, caches_dev, a_dim, tok_dim, summ_dim,
                             cfg=cfg, verbose=verbose)
    return _dev_ll(name, model, caches_dev, cfg)


def sweep(name, base_cfg, grid, caches_train, caches_dev, a_dim, tok_dim, summ_dim, seed, verbose):
    """Trains ``name`` once per overlay in ``grid`` (each merged onto
    ``base_cfg``); returns ``(rows, best_overlay)``."""
    rows = []
    for overlay in grid:
        cfg = {**base_cfg, **overlay, "seed": seed}
        t0 = time.time()
        ll = _train_and_score(name, caches_train, caches_dev, a_dim, tok_dim, summ_dim, cfg, seed, verbose)
        rows.append({**{k: str(v) for k, v in overlay.items()}, "dev_imp_wt_ll": ll, "seconds": time.time() - t0})
    best = min(range(len(grid)), key=lambda i: rows[i]["dev_imp_wt_ll"])
    return rows, grid[best]


def split_dev(dev_days, frac):
    n_adtr = max(1, int(round(len(dev_days) * frac)))
    n_adtr = min(n_adtr, len(dev_days) - 1) if len(dev_days) > 1 else len(dev_days)
    return dev_days[:n_adtr], (dev_days[n_adtr:] or dev_days[-1:])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["criteo", "avazu"], default="criteo")
    ap.add_argument("--data", default=None)
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-features", type=int, default=2 ** 18)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--block-sec", type=int, default=900)
    ap.add_argument("--delay-sec", type=int, default=1800)
    ap.add_argument("--adapter-train-frac", type=float, default=0.7)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--verbose-train", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    ds = load(args.source, args.data or DATA_PATHS[args.source],
             n_features=args.n_features, sample_frac=args.sample_frac, seed=args.seed)
    split = make_split(ds.n_days, warmup=args.warmup)
    print(f"{ds.name}: {ds.n_days} days, dev {list(map(int, split.dev_days))}", flush=True)

    # HPO only ever needs the development days (their history is days < d,
    # always available), matching twoscale_hpo.py.
    bank = build_bank(ds, split.dev_days, seed=args.seed, n_jobs=args.n_jobs)
    eval_days = sorted(bank)
    w = adaptive_weights(bank, eval_days, eta=MIXTURE_ETA, halflife=MIXTURE_HALFLIFE)
    q_by_day = long_term_predictions(bank, eval_days, "adaptive", weights=w)

    # --- stage 1: context-sketch dimension, proxy = v3_mlp @ default cfg ---
    sketch_rows = []
    caches_by_m = {}
    for m in SKETCH_DIM_GRID:
        cache = build_cache(ds, bank, q_by_day, eval_days, block_sec=args.block_sec,
                            delay_sec=args.delay_sec, m=m, seed=args.seed)
        caches_by_m[m] = cache
        dev_days = sorted(cache)
        adtr, addev = split_dev(dev_days, args.adapter_train_frac)
        a_dim, tok_dim, summ_dim = m + 2, token_dim(m), summary_dim(m)
        cfg = {**DEFAULT_CFG, "seed": args.seed}
        t0 = time.time()
        ll = _train_and_score("v3_mlp", [cache[d] for d in adtr], [cache[d] for d in addev],
                              a_dim, tok_dim, summ_dim, cfg, args.seed, args.verbose_train)
        sketch_rows.append({"sketch_dim": m, "dev_imp_wt_ll": ll, "seconds": time.time() - t0})
        print(f"  sketch_dim={m}: dev_ll={ll:.6f} ({time.time() - t0:.1f}s)", flush=True)
    pd.DataFrame(sketch_rows).to_csv(out / "hpo_sketch_dim.csv", index=False)
    best_m = min(sketch_rows, key=lambda r: r["dev_imp_wt_ll"])["sketch_dim"]
    print(f"\nbest sketch_dim = {best_m}", flush=True)

    cache = caches_by_m[best_m]
    dev_days = sorted(cache)
    adapter_train_days, adapter_dev_days = split_dev(dev_days, args.adapter_train_frac)
    caches_train = [cache[d] for d in adapter_train_days]
    caches_dev = [cache[d] for d in adapter_dev_days]
    a_dim, tok_dim, summ_dim = best_m + 2, token_dim(best_m), summary_dim(best_m)
    print(f"adapter-train days {adapter_train_days}  adapter-dev days {adapter_dev_days}", flush=True)

    # --- stage 2: per-variant staged search --------------------------------
    per_variant = {}
    all_rows = []
    for name in args.variants:
        print(f"\n=== {name} ===", flush=True)
        base = dict(DEFAULT_CFG)

        rows, best_overlay = sweep(name, base, [{"lr": lr, "weight_decay": wd} for lr, wd in LR_WD_GRID],
                                   caches_train, caches_dev, a_dim, tok_dim, summ_dim, args.seed, args.verbose_train)
        for r in rows:
            all_rows.append({"variant": name, "stage": "lr_wd", **r})
        base.update(best_overlay)
        print(f"  lr/wd -> {best_overlay}", flush=True)

        cap_grid = CAPACITY_GRID[name]
        rows, best_overlay = sweep(name, base, cap_grid, caches_train, caches_dev,
                                   a_dim, tok_dim, summ_dim, args.seed, args.verbose_train)
        for r in rows:
            all_rows.append({"variant": name, "stage": "capacity", **r})
        base.update(best_overlay)
        print(f"  capacity -> {best_overlay}", flush=True)

        rows, best_overlay = sweep(name, base, [{"dropout": dp, "delta_max": dm} for dp, dm in DROPOUT_DELTAMAX_GRID],
                                   caches_train, caches_dev, a_dim, tok_dim, summ_dim, args.seed, args.verbose_train)
        for r in rows:
            all_rows.append({"variant": name, "stage": "dropout_deltamax", **r})
        base.update(best_overlay)
        print(f"  dropout/delta_max -> {best_overlay}", flush=True)

        base.pop("seed", None)
        per_variant[name] = base
        print(f"  FROZEN[{name}] = {base}", flush=True)

    pd.DataFrame(all_rows).to_csv(out / "hpo_grid.csv", index=False)
    frozen = {"sketch_dim": int(best_m), "per_variant": per_variant}
    (out / "FROZEN.json").write_text(json.dumps(frozen, indent=2, default=float))
    print(f"\nFROZEN -> {out}/FROZEN.json  ({time.time() - t_start:.1f}s total)", flush=True)


if __name__ == "__main__":
    main()
