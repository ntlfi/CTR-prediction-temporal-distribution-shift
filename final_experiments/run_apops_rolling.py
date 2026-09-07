"""Plan section 3, "Rolling origin": the primary evidence for temporal
adaptation.  Evaluate the frozen AP-OPS rows at every established
chronological origin -- Criteo outer days 16-30, Avazu 5-9 -- 3 seeds.

Nothing is re-selected per origin: the plan's minimal study reuses the
cross-day predictions and replays only the calibration layer, and every
calibration hyper-parameter is already frozen in ``apops_selected.json``.
At origin ``d`` each method is replayed over the whole prefix ``1..d``
(the persistent experts and the meta-weights carry exactly the state a
real deployment would have had) and only day ``d``'s prediction is scored.

Output mirrors ``run_rolling.py`` so ``day_level_stats.py`` runs on it
directly: ``seed<k>/per_day_metrics.csv`` with one row per (method, origin
day).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from twoscale.data import load
from twoscale.longterm import build_bank
from twoscale.metrics import per_day_frame
from twoscale_run import DATA_PATHS

from methods import adaptive_q_by_day
from apops.method import APOPSConfig, build_rows

SEEDS = (0, 1, 2)
OUTER_DAYS = {"criteo": list(range(16, 31)), "avazu": list(range(5, 10))}
ROW_ORDER = ["current_ops", "ap_ops", "single_memory", "no_slope"]


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def cfg_from_selected(sel) -> APOPSConfig:
    return APOPSConfig(block_sec=sel["block_sec"], delay_sec=sel["delay_sec"],
                       ons_lam=sel["ons_lam"], ons_eta=sel["ons_eta"], eta_m=sel["eta_m"],
                       switch_half_life_h=sel["switch_half_life_h"],
                       a_bounds=tuple(sel["a_bounds"]), b_bounds=tuple(sel["b_bounds"]),
                       single_memory_pick=sel["single_memory_pick"])


def run_seed(source, data_path, sample_frac, n_features, warmup, n_jobs, seed, sel, out_dir):
    t = time.time()
    ds = load(source, data_path, n_features=n_features, sample_frac=sample_frac, seed=seed)
    outer_days = [d for d in OUTER_DAYS[source] if d < ds.n_days]
    bank_days = list(range(1, max(outer_days) + 1))
    bank = build_bank(ds, bank_days, seed=seed, n_jobs=n_jobs)
    bank_days = sorted(bank)
    print(f"  seed {seed}: {len(ds.y):,} rows | outer {outer_days}", flush=True)

    mix = sel["shared_mixture"]
    cfg = cfg_from_selected(sel)

    per_day_rows = []
    for d in outer_days:
        prefix = [e for e in bank_days if e <= d]
        q_by_day, _ = adaptive_q_by_day(bank, prefix, eta=mix["eta"], halflife=mix["halflife"])
        rows, _ = build_rows(bank, prefix, q_by_day, cfg, sel["ops"])
        for name in ROW_ORDER:
            rec_d = next(r for r in rows[name] if r["day"] == d)
            rec = [{"day": d, "y": bank[d].y, "p": np.asarray(rec_d["p"]),
                    "sec_in_day": bank[d].sec_in_day}]
            for row in per_day_frame(rec):
                per_day_rows.append({"method": name, **row})
        print(f"    origin day {d} done", flush=True)

    seed_out = out_dir / f"seed{seed}"
    seed_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_day_rows).to_csv(seed_out / "per_day_metrics.csv", index=False)
    (seed_out / "summary.json").write_text(json.dumps(
        {"seed": seed, "n_rows": int(len(ds.y)), "outer_days": outer_days,
         "runtime_s": time.time() - t}, indent=2, default=float))
    del ds, bank


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

    sel = json.loads(Path(args.config).read_text())
    assert sel["source"] == args.source
    warmup = args.warmup if args.warmup is not None else (4 if args.source == "criteo" else 3)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for seed in SEEDS:
        run_seed(args.source, args.data or DATA_PATHS[args.source], args.sample_frac,
                 args.n_features, warmup, args.n_jobs, seed, sel, out)
    (out / "summary.json").write_text(json.dumps({
        "source": args.source, "config_used": str(args.config), "seeds": list(SEEDS),
        "outer_days": OUTER_DAYS[args.source], "code_commit": git_commit_hash(),
        "reselected_per_origin": [], "runtime_s": time.time() - t0,
    }, indent=2, default=float))
    print(f"\nruntime {time.time() - t0:.1f}s -> {out}/", flush=True)


if __name__ == "__main__":
    main()
