"""Rolling-origin confirmation for the six headline methods (review comment
1: the temporal significance statement should not rest on one 22-30 /
7-9 cut).

Walk-forward over every eligible outer day -- Criteo 16-30 (15 origins),
Avazu 5-9 (5 origins) -- with each origin's hyperparameters re-selected
using **only** the days before it (inner rolling-origin CV on the trailing
``INNER_K`` eligible days). Deltas are then analysed at the calendar-day
level (``day_level_stats.py``): average the 3 seeds on each origin day
first, then bootstrap / sign-test across the origins.

What is re-selected per origin (cheap -- all are bank reads):

  * shared adaptive cross-day mixture  eta x halflife   (15 configs)
  * Best Fixed Window                  h*               (3)
  * ARW                                delta            (3)
  * AdaMoE                             lambda           (5)

What is held at the frozen ``selected_configs.json`` value (documented
deviation): OPS ``(B, eta0, schedule)`` and DualTime ``B_w``, plus
``block_sec`` / ``delay_sec`` / ``m`` / ``cross_dim``. Their dev-day HPO
grids were essentially flat (see ``*/hpo/hpo_ops.csv``, ``hpo_dualtime.csv``)
and -- because OPS and DualTime-CTR both reset their within-day state every
day in this protocol (``carryover_rho=0``, ``w<-0`` each morning) -- a
day's OPS/DualTime prediction depends only on *that day's* ``q``, so a full
per-origin grid re-search over 32 + 5 configs would cost far more than the
knob movement could justify. ARW and AdaMoE are the only methods carrying
genuine cross-day state, and they are re-selected.

Outputs under ``--out`` (mirrors ``run_final.py`` schema so
``day_level_stats.py`` runs on it directly):
  seed<k>/per_day_metrics.csv     one row per (method, origin day)
  rolling_origin_manifest.csv     per (seed, origin) selected configs + inner days
  summary.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from twoscale.calib import CalibConfig
from twoscale.data import load
from twoscale.longterm import HORIZONS, build_bank
from twoscale.metrics import impression_weighted_logloss, per_day_frame
from twoscale.splits import make_split
from twoscale_run import DATA_PATHS

from dualtime.online import DualTimeConfig, build_hash_projection
from dualtime.online import replay_day as dt_replay_day
from methods import adamoe_method, adaptive_q_by_day, arw_method
from withinday.blocks import summary_dim
from withinday.contextsketch import build_projection, context_sketch

SEEDS = (0, 1, 2)
INNER_K = 3
OUTER_DAYS = {"criteo": list(range(16, 31)), "avazu": list(range(5, 10))}

MIX_ETA_GRID = [10.0, 30.0, 60.0, 150.0, 1e6]
MIX_HALFLIFE_GRID = [3.0, 5.0, 10.0]
ARW_DELTA_GRID = [0.05, 0.10, 0.20]
ADAMOE_LAMBDA_GRID = [0.0, 0.25, 0.50, 0.75, 0.99]


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def inner_days_for(d: int, bank_days: list[int]) -> list[int]:
    eligible = [e for e in bank_days if e < d and any(x < e for x in bank_days)]
    return eligible[-INNER_K:]


def _iw_loss_on(records, days: set) -> float:
    return impression_weighted_logloss([r for r in records if r["day"] in days])


def select_mixture(bank, inner_days, prefix_days):
    best, best_loss = None, np.inf
    for eta in MIX_ETA_GRID:
        for hl in MIX_HALFLIFE_GRID:
            q, _ = adaptive_q_by_day(bank, prefix_days, eta=eta, halflife=hl)
            recs = [{"day": e, "y": bank[e].y, "p": q[e]} for e in inner_days if e in q]
            loss = impression_weighted_logloss(recs)
            if loss < best_loss - 1e-12:
                best, best_loss = (eta, hl), loss
    return best


def select_best_fixed(bank, inner_days):
    losses = {h: impression_weighted_logloss(
        [{"day": e, "y": bank[e].y, "p": bank[e].preds[h]} for e in inner_days]) for h in HORIZONS}
    return min(losses, key=losses.get)


def select_arw(bank, inner_days, prefix_days):
    best, best_loss = None, np.inf
    for delta in ARW_DELTA_GRID:
        recs, _ = arw_method(bank, prefix_days, delta=delta)
        loss = _iw_loss_on(recs, set(inner_days))
        if loss < best_loss - 1e-12:
            best, best_loss = delta, loss
    return best


def select_adamoe(bank, inner_days, prefix_days):
    best, best_loss = None, np.inf
    for lam in ADAMOE_LAMBDA_GRID:
        recs, _ = adamoe_method(bank, prefix_days, lam=lam)
        loss = _iw_loss_on(recs, set(inner_days))
        if loss < best_loss - 1e-12:
            best, best_loss = lam, loss
    return best


def dualtime_day(q_d, bank_d, X_day, R_sketch, Ra, Rs, dt_cfg, csketch):
    out = dt_replay_day(q_d, bank_d.y, bank_d.sec_in_day, X_day, R_sketch, Ra, Rs, dt_cfg,
                        csketch=csketch)
    return out["p_hat"]


def ops_day(q_d, bank_d, cfg: CalibConfig):
    from twoscale.calib import replay_day as calib_replay
    return calib_replay(q_d, bank_d.y, bank_d.sec_in_day, cfg, init_b=0.0)["p_hat"]


def run_seed(source, data_path, sample_frac, n_features, warmup, n_jobs, seed, selected, out_dir):
    t = time.time()
    ds = load(source, data_path, n_features=n_features, sample_frac=sample_frac, seed=seed)
    outer_days = [d for d in OUTER_DAYS[source] if d < ds.n_days]
    bank_days = list(range(1, max(outer_days) + 1))
    print(f"  seed {seed}: {len(ds.y):,} rows | outer {outer_days} | bank over {bank_days[0]}-{bank_days[-1]}",
          flush=True)
    bank = build_bank(ds, bank_days, seed=seed, n_jobs=n_jobs)
    bank_days = sorted(bank)

    block_sec, delay_sec = selected["block_sec"], selected["delay_sec"]
    m = selected["dualtime"]["m"]
    ops_c = selected["ops"]
    ops_cfg = CalibConfig(B=ops_c["B"], eta0=ops_c["eta0"], eta_schedule=ops_c["schedule"],
                          update="block", block_sec=block_sec, delay_sec=delay_sec, platt=True)
    dt = selected["dualtime"]
    dt_cfg = DualTimeConfig(block_sec=block_sec, delay_sec=delay_sec, m=dt["m"],
                            cross_dim=dt["cross_dim"], B_w=dt["B_w"])
    R_sketch = build_projection(ds.X.shape[1], m, seed=seed)
    Ra, Rs = build_hash_projection(m + 2, summary_dim(m), cross_dim=dt["cross_dim"], seed=seed)

    csketch_cache = {}

    def csketch_for(d):
        if d not in csketch_cache:
            sl = ds.day_slice(d)
            csketch_cache[d] = context_sketch(ds.X[sl], m=m, R=R_sketch)
        return csketch_cache[d]

    per_day_rows, manifest_rows = [], []
    for d in outer_days:
        inner = inner_days_for(d, bank_days)
        prefix = [e for e in bank_days if e <= d]
        prefix_hist = [e for e in bank_days if e < d]

        eta_hl = select_mixture(bank, inner, prefix_hist)
        h_star = select_best_fixed(bank, inner)
        arw_delta = select_arw(bank, inner, prefix_hist)
        moe_lam = select_adamoe(bank, inner, prefix_hist)

        q_by_day, _ = adaptive_q_by_day(bank, prefix, eta=eta_hl[0], halflife=eta_hl[1])
        arw_recs, _ = arw_method(bank, prefix, delta=arw_delta)
        moe_recs, _ = adamoe_method(bank, prefix, lam=moe_lam)
        arw_d = next(r for r in arw_recs if r["day"] == d)
        moe_d = next(r for r in moe_recs if r["day"] == d)

        sl = ds.day_slice(d)
        preds = {
            "expanding": bank[d].preds["expanding"],
            "best_fixed": bank[d].preds[h_star],
            "arw": arw_d["p"],
            "adamoe": moe_d["p"],
            "long_only": q_by_day[d],
            "ops": ops_day(q_by_day[d], bank[d], ops_cfg),
            "dualtime": dualtime_day(q_by_day[d], bank[d], ds.X[sl], R_sketch, Ra, Rs, dt_cfg,
                                     csketch_for(d)),
        }
        for name, p in preds.items():
            rec = [{"day": d, "y": bank[d].y, "p": np.asarray(p), "sec_in_day": bank[d].sec_in_day}]
            for row in per_day_frame(rec):
                per_day_rows.append({"method": name, **row})
        manifest_rows.append({
            "seed": seed, "outer_day": d, "inner_days": ";".join(map(str, inner)),
            "mix_eta": eta_hl[0], "mix_halflife": eta_hl[1], "best_fixed_h": h_star,
            "arw_delta": arw_delta, "adamoe_lambda": moe_lam,
            "ops_B": ops_c["B"], "ops_eta0": ops_c["eta0"], "dualtime_B_w": dt["B_w"],
        })
        print(f"    day {d}: mix=({eta_hl[0]},{eta_hl[1]}) h*={h_star} "
              f"arw_delta={arw_delta} moe_lam={moe_lam}", flush=True)

    seed_out = out_dir / f"seed{seed}"
    seed_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_day_rows).to_csv(seed_out / "per_day_metrics.csv", index=False)
    (seed_out / "summary.json").write_text(json.dumps(
        {"seed": seed, "n_rows": int(len(ds.y)), "outer_days": outer_days,
         "runtime_s": time.time() - t}, indent=2, default=float))
    del ds, bank
    return manifest_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["criteo", "avazu"], required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--n-features", type=int, default=2 ** 18)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    selected = json.loads(Path(args.config).read_text())
    assert selected["source"] == args.source
    warmup = args.warmup if args.warmup is not None else (4 if args.source == "criteo" else 3)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    manifest = []
    for seed in SEEDS:
        manifest += run_seed(args.source, args.data or DATA_PATHS[args.source], args.sample_frac,
                             args.n_features, warmup, args.n_jobs, seed, selected, out)
    pd.DataFrame(manifest).to_csv(out / "rolling_origin_manifest.csv", index=False)
    (out / "summary.json").write_text(json.dumps({
        "source": args.source, "config_used": str(args.config), "seeds": list(SEEDS),
        "outer_days": OUTER_DAYS[args.source], "inner_k": INNER_K,
        "code_commit": git_commit_hash(),
        "frozen_from_selected_configs": ["ops.B", "ops.eta0", "ops.schedule", "dualtime.B_w",
                                         "block_sec", "delay_sec", "m", "cross_dim"],
        "reselected_per_origin": ["shared_mixture.eta", "shared_mixture.halflife",
                                  "best_fixed_window.h_star", "arw.delta", "adamoe.lambda"],
        "runtime_s": time.time() - t0,
    }, indent=2, default=float))
    print(f"\nruntime {time.time() - t0:.1f}s -> {out}/", flush=True)


if __name__ == "__main__":
    main()
