"""Sections 6-10: pick ONE common configuration per dataset, selected by
mean development-day loss across seeds {0,1,2} (never a per-seed
config -- section 6's explicit requirement, different from every earlier
HPO script in this repo, which picked independently per seed).

Staged, because later stages depend on earlier ones being frozen first
(section 8: "tune the long-term parameters before tuning either
within-day method"):

  1. Best Fixed Window: h* in {roll3, roll7, expanding} (section 7.2)
  2. ARW: delta in {0.05, 0.10, 0.20} (section 7.3)
  3. AdaMoE: lambda in {0, .25, .5, .75, .99} (section 7.4)
  4. shared adaptive cross-day mixture: eta x halflife, 15 configs,
     selected on the *uncalibrated* adaptive-mixture dev loss (section 8)
  5. OPS: B x eta0 x schedule, 32 configs, on top of the frozen mixture (section 10)
  6. DualTime-CTR: B_w in {.25,.5,1,2,4}, on top of the SAME frozen mixture (section 11)

Every stage writes its full grid table (not just the winner). Writes
``selected_configs.json`` at the end.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from twoscale.calib import CalibConfig
from twoscale.data import load
from twoscale.longterm import HORIZONS, build_bank
from twoscale.metrics import day_logloss, impression_weighted_logloss
from twoscale.splits import make_split
from twoscale_run import DATA_PATHS

from dualtime.online import DualTimeConfig
from methods import adamoe_method, adaptive_q_by_day, arw_method, dualtime_method, ops_method

SEEDS = (0, 1, 2)
ARW_DELTA_GRID = [0.05, 0.10, 0.20]
ADAMOE_LAMBDA_GRID = [0.0, 0.25, 0.50, 0.75, 0.99]
MIX_ETA_GRID = [10.0, 30.0, 60.0, 150.0, 1e6]
MIX_HALFLIFE_GRID = [3.0, 5.0, 10.0]
OPS_B_GRID = [0.25, 0.5, 1.0, 2.0]
OPS_ETA0_GRID = [0.01, 0.03, 0.1, 0.3]
OPS_SCHEDULE_GRID = ["const", "inv_sqrt"]
DUALTIME_BW_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]

BLOCK_SEC = {"criteo": 900, "avazu": 3600}   # section 9.2
DELAY_SEC = 1800                             # section 9.1, frozen, not tuned


def _dev_loss(records, dev_days):
    dev_days = set(dev_days)
    return impression_weighted_logloss([r for r in records if r["day"] in dev_days])


def _worst_day_loss(records, dev_days):
    dev_days = set(dev_days)
    rows = [r for r in records if r["day"] in dev_days]
    return max((day_logloss(r["y"], r["p"]) for r in rows), default=float("inf"))


def select_by_mean_across_seeds(rows_per_seed: list[list[dict]], loss_key: str = "dev_loss",
                                worst_key: str = "worst_day_loss", complexity_key=None):
    """Section 6: average ``loss_key`` across seeds for each config (rows
    must be in the same config order for every seed), pick the lowest
    mean; ties within 1e-6 broken by lower mean worst-day loss, then by
    ``complexity_key`` (simpler/more conservative first)."""
    n_configs = len(rows_per_seed[0])
    merged = []
    for i in range(n_configs):
        base = {k: v for k, v in rows_per_seed[0][i].items() if k not in (loss_key, worst_key)}
        mean_loss = float(np.mean([rows_per_seed[s][i][loss_key] for s in range(len(rows_per_seed))]))
        mean_worst = float(np.mean([rows_per_seed[s][i][worst_key] for s in range(len(rows_per_seed))]))
        merged.append({**base, f"mean_{loss_key}": mean_loss, f"mean_{worst_key}": mean_worst})
    merged.sort(key=lambda r: r[f"mean_{loss_key}"])
    best = merged[0][f"mean_{loss_key}"]
    tied = [r for r in merged if r[f"mean_{loss_key}"] - best < 1e-6]
    if len(tied) > 1:
        tied.sort(key=lambda r: r[f"mean_{worst_key}"])
        w = tied[0][f"mean_{worst_key}"]
        tied = [r for r in tied if r[f"mean_{worst_key}"] - w < 1e-9]
        if complexity_key is not None and len(tied) > 1:
            tied.sort(key=complexity_key)
    return tied[0], merged


def load_seed_bank(source, path, sample_frac, n_features, warmup, seed, n_jobs):
    ds = load(source, path, n_features=n_features, sample_frac=sample_frac, seed=seed)
    split = make_split(ds.n_days, warmup=warmup)
    dev_days = list(split.dev_days)
    bank = build_bank(ds, dev_days, seed=seed, n_jobs=n_jobs, verbose=False)
    return ds, split, bank


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["criteo", "avazu"], required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--sample-frac", type=float, default=None)
    ap.add_argument("--n-features", type=int, default=2 ** 18)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sample_frac = args.sample_frac if args.sample_frac is not None else 1.0
    warmup = args.warmup if args.warmup is not None else (4 if args.source == "criteo" else 3)
    block_sec = BLOCK_SEC[args.source]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Each seed's dataset/bank is loaded fresh and dropped (falls out of
    # scope) at the end of its own iteration -- for full-data Avazu
    # (~40M rows), holding all 3 seeds' datasets simultaneously risks
    # exceeding a compute node's memory (this cluster's nodes cap at
    # ~250G), so phase A and phase B below each load every seed once,
    # sequentially, rather than once per seed for the whole script.

    # ---- phase A: stages 1-4, one seed at a time ---------------------------
    bfw_rows_per_seed, arw_rows_per_seed, moe_rows_per_seed, mix_rows_per_seed = [], [], [], []
    for seed in SEEDS:
        ds, split, bank = load_seed_bank(args.source, args.data or DATA_PATHS[args.source],
                                         sample_frac, args.n_features, warmup, seed, args.n_jobs)
        dev_days = list(split.dev_days)
        print(f"  seed {seed} (phase A): {len(ds.y):,} rows, dev days {dev_days}", flush=True)

        rows = []
        for h in HORIZONS:
            recs = [{"day": d, "y": bank[d].y, "p": bank[d].preds[h], "sec_in_day": bank[d].sec_in_day}
                   for d in dev_days if d in bank]
            rows.append({"window": h, "dev_loss": impression_weighted_logloss(recs),
                        "worst_day_loss": _worst_day_loss(recs, dev_days)})
        bfw_rows_per_seed.append(rows)

        rows = []
        for delta in ARW_DELTA_GRID:
            recs, _ = arw_method(bank, dev_days, delta=delta)
            rows.append({"delta": delta, "dev_loss": _dev_loss(recs, dev_days),
                        "worst_day_loss": _worst_day_loss(recs, dev_days)})
        arw_rows_per_seed.append(rows)

        rows = []
        for lam in ADAMOE_LAMBDA_GRID:
            recs, _ = adamoe_method(bank, dev_days, lam=lam)
            rows.append({"lambda": lam, "dev_loss": _dev_loss(recs, dev_days),
                        "worst_day_loss": _worst_day_loss(recs, dev_days)})
        moe_rows_per_seed.append(rows)

        rows = []
        for eta, halflife in itertools.product(MIX_ETA_GRID, MIX_HALFLIFE_GRID):
            q_by_day, _ = adaptive_q_by_day(bank, dev_days, eta=eta, halflife=halflife)
            recs = [{"day": d, "y": bank[d].y, "p": q_by_day[d], "sec_in_day": bank[d].sec_in_day}
                   for d in dev_days if d in bank]
            rows.append({"mix_eta": eta, "mix_halflife": halflife, "dev_loss": _dev_loss(recs, dev_days),
                        "worst_day_loss": _worst_day_loss(recs, dev_days)})
        mix_rows_per_seed.append(rows)
        del ds, split, bank   # explicit: don't hold this seed's data into the next iteration

    bfw_best, bfw_table = select_by_mean_across_seeds(bfw_rows_per_seed)
    pd.DataFrame(bfw_table).to_csv(out / "hpo_best_fixed.csv", index=False)
    print(f"Best Fixed Window -> {bfw_best['window']}", flush=True)

    arw_best, arw_table = select_by_mean_across_seeds(arw_rows_per_seed)
    pd.DataFrame(arw_table).to_csv(out / "hpo_arw.csv", index=False)
    print(f"ARW -> delta={arw_best['delta']}", flush=True)

    moe_best, moe_table = select_by_mean_across_seeds(moe_rows_per_seed)
    pd.DataFrame(moe_table).to_csv(out / "hpo_adamoe.csv", index=False)
    print(f"AdaMoE -> lambda={moe_best['lambda']}", flush=True)

    mix_best, mix_table = select_by_mean_across_seeds(mix_rows_per_seed)
    pd.DataFrame(mix_table).to_csv(out / "hpo_longterm.csv", index=False)
    mix_eta, mix_halflife = mix_best["mix_eta"], mix_best["mix_halflife"]
    print(f"shared adaptive mixture -> eta={mix_eta} halflife={mix_halflife}", flush=True)

    # ---- phase B: stages 5-6, one seed at a time (reload -- see above) -----
    ops_rows_per_seed, dt_rows_per_seed = [], []
    for seed in SEEDS:
        ds, split, bank = load_seed_bank(args.source, args.data or DATA_PATHS[args.source],
                                         sample_frac, args.n_features, warmup, seed, args.n_jobs)
        dev_days = list(split.dev_days)
        print(f"  seed {seed} (phase B): {len(ds.y):,} rows", flush=True)
        q_by_day, _ = adaptive_q_by_day(bank, dev_days, eta=mix_eta, halflife=mix_halflife)

        rows = []
        for B, eta0, sched in itertools.product(OPS_B_GRID, OPS_ETA0_GRID, OPS_SCHEDULE_GRID):
            cfg = CalibConfig(B=B, eta0=eta0, eta_schedule=sched, update="block",
                              block_sec=block_sec, delay_sec=DELAY_SEC, platt=True)
            recs, _ = ops_method(bank, dev_days, q_by_day, cfg)
            rows.append({"B": B, "eta0": eta0, "schedule": sched, "dev_loss": _dev_loss(recs, dev_days),
                        "worst_day_loss": _worst_day_loss(recs, dev_days)})
        ops_rows_per_seed.append(rows)

        rows = []
        for B_w in DUALTIME_BW_GRID:
            dt_cfg = DualTimeConfig(block_sec=block_sec, delay_sec=DELAY_SEC, m=32, cross_dim=32, B_w=B_w)
            recs = dualtime_method(ds, bank, dev_days, q_by_day, dt_cfg, sketch_seed=seed, hash_seed=seed)
            rows.append({"B_w": B_w, "dev_loss": _dev_loss(recs, dev_days),
                        "worst_day_loss": _worst_day_loss(recs, dev_days)})
        dt_rows_per_seed.append(rows)
        del ds, split, bank, q_by_day

    ops_best, ops_table = select_by_mean_across_seeds(
        ops_rows_per_seed, complexity_key=lambda r: r["B"])
    pd.DataFrame(ops_table).to_csv(out / "hpo_ops.csv", index=False)
    print(f"OPS -> B={ops_best['B']} eta0={ops_best['eta0']} schedule={ops_best['schedule']}", flush=True)

    dt_best, dt_table = select_by_mean_across_seeds(dt_rows_per_seed, complexity_key=lambda r: r["B_w"])
    pd.DataFrame(dt_table).to_csv(out / "hpo_dualtime.csv", index=False)
    print(f"DualTime-CTR -> B_w={dt_best['B_w']}", flush=True)

    selected = {
        "source": args.source, "block_sec": block_sec, "delay_sec": DELAY_SEC,
        "best_fixed_window": {"h_star": bfw_best["window"]},
        "arw": {"delta": arw_best["delta"], "min_history": 3},
        "adamoe": {"lambda": moe_best["lambda"]},
        "shared_mixture": {"eta": mix_eta, "halflife": mix_halflife},
        "ops": {"B": ops_best["B"], "eta0": ops_best["eta0"], "schedule": ops_best["schedule"],
               "a_bounds": [0.2, 5.0]},
        "dualtime": {"B_w": dt_best["B_w"], "m": 32, "cross_dim": 32,
                    "ewma_halflives": [1.0, 4.0, 16.0]},
        "seeds": list(SEEDS), "runtime_s": time.time() - t0,
    }
    (out / "selected_configs.json").write_text(json.dumps(selected, indent=2, default=float))
    print(f"\nselected_configs -> {out}/selected_configs.json  ({time.time() - t0:.1f}s total)", flush=True)


if __name__ == "__main__":
    main()
