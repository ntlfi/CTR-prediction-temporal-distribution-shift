"""Plan section 3, "Development": tune ONLY the ONS scale / regularization
and the two meta parameters (learning rate ``eta_m`` and share rate via a
switching half-life ``tau``), on the established dev days, seeds {0,1,2},
selected by mean dev-day impression-weighted log loss across seeds.  Then
freeze ONE configuration per dataset into ``apops_selected.json``.

Everything upstream of calibration is reused unchanged: the frozen shared
adaptive cross-day mixture and the locked OPS hyper-parameters both come
straight from this dataset's existing ``selected_configs.json``.

Also recorded in the manifest (needed by the decision rule, section 4):

    delta_NI = 0.10 * | L_OPS,dev  -  L_base-q,dev |

with ``L_base-q`` the *uncalibrated* adaptive-mixture dev loss.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from twoscale.data import load
from twoscale.longterm import build_bank
from twoscale.metrics import day_logloss, impression_weighted_logloss
from twoscale.splits import make_split
from twoscale_run import DATA_PATHS

from methods import adaptive_q_by_day
from apops.aggregate import MetaConfig, aggregate
from apops.experts import replay_ons_stream
from apops.method import LONG_HL_H, SHORT_HL_H, APOPSConfig, current_ops

SEEDS = (0, 1, 2)
ONS_LAM_GRID = [1e-3, 1e-2, 1e-1, 1.0]
ONS_ETA_GRID = [0.25, 1.0, 4.0]
ETA_M_GRID = [1.0, 10.0, 100.0]
TAU_GRID = [4.0, 16.0, 64.0, float("inf")]


def _dev_loss(records, dev_days):
    dev_days = set(dev_days)
    return impression_weighted_logloss([r for r in records if r["day"] in dev_days])


def _worst_day(records, dev_days):
    dev_days = set(dev_days)
    return max((day_logloss(r["y"], r["p"]) for r in records if r["day"] in dev_days), default=float("inf"))


def load_seed(source, path, sample_frac, n_features, warmup, seed, n_jobs):
    ds = load(source, path, n_features=n_features, sample_frac=sample_frac, seed=seed)
    split = make_split(ds.n_days, warmup=warmup)
    dev_days = list(split.dev_days)
    bank = build_bank(ds, dev_days, seed=seed, n_jobs=n_jobs, verbose=False)
    return ds, split, bank, [d for d in dev_days if d in bank]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["criteo", "avazu"], required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--n-features", type=int, default=2 ** 18)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--ops-config", required=True, help="the dataset's frozen selected_configs.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = json.loads(Path(args.ops_config).read_text())
    assert base["source"] == args.source, "--ops-config is for a different --source"
    block_sec, delay_sec = base["block_sec"], base["delay_sec"]
    mix = base["shared_mixture"]
    ops_hp = base["ops"]
    warmup = args.warmup if args.warmup is not None else (4 if args.source == "criteo" else 3)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ons_combos = list(itertools.product(ONS_LAM_GRID, ONS_ETA_GRID))
    meta_combos = list(itertools.product(ETA_M_GRID, TAU_GRID))

    # per seed: rows keyed by (lam, eta, eta_m, tau); plus single-memory + margin pieces
    grid_per_seed, smem_per_seed, margin_per_seed = [], [], []
    for seed in SEEDS:
        ds, split, bank, dev_days = load_seed(
            args.source, args.data or DATA_PATHS[args.source], args.sample_frac,
            args.n_features, warmup, seed, args.n_jobs)
        print(f"  seed {seed}: {len(ds.y):,} rows, dev days {dev_days}", flush=True)
        q_by_day, _ = adaptive_q_by_day(bank, dev_days, eta=mix["eta"], halflife=mix["halflife"])

        r_recs, _ = current_ops(bank, dev_days, q_by_day, ops_hp, block_sec, delay_sec)
        l_ops = _dev_loss(r_recs, dev_days)
        l_baseq = impression_weighted_logloss(
            [{"day": d, "y": bank[d].y, "p": q_by_day[d]} for d in dev_days])
        margin_per_seed.append({"l_ops": l_ops, "l_baseq": l_baseq})

        rows, smem_rows = [], []
        for lam, eta in ons_combos:
            cfg = APOPSConfig(block_sec=block_sec, delay_sec=delay_sec, ons_lam=lam, ons_eta=eta)
            s_recs, _ = replay_ons_stream(q_by_day, bank, dev_days, cfg.ons_cfg(SHORT_HL_H))
            l_recs, _ = replay_ons_stream(q_by_day, bank, dev_days, cfg.ons_cfg(LONG_HL_H))
            smem_rows.append({"ons_lam": lam, "ons_eta": eta,
                              "S_dev_loss": _dev_loss(s_recs, dev_days),
                              "L_dev_loss": _dev_loss(l_recs, dev_days)})
            experts = {"R": r_recs, "S": s_recs, "L": l_recs}
            for eta_m, tau in meta_combos:
                mcfg = MetaConfig(eta_m=eta_m, switch_half_life_h=tau,
                                  block_sec=block_sec, delay_sec=delay_sec)
                ap_recs, _ = aggregate(experts, bank, dev_days, mcfg)
                rows.append({"ons_lam": lam, "ons_eta": eta, "eta_m": eta_m, "tau": tau,
                             "dev_loss": _dev_loss(ap_recs, dev_days),
                             "worst_day_loss": _worst_day(ap_recs, dev_days)})
        grid_per_seed.append(rows)
        smem_per_seed.append(smem_rows)
        del ds, split, bank, q_by_day

    # ---- aggregate across seeds (mean dev loss per config) ----------------
    n = len(grid_per_seed[0])
    merged = []
    for i in range(n):
        b = {k: grid_per_seed[0][i][k] for k in ("ons_lam", "ons_eta", "eta_m", "tau")}
        b["mean_dev_loss"] = float(np.mean([grid_per_seed[s][i]["dev_loss"] for s in range(len(SEEDS))]))
        b["mean_worst_day"] = float(np.mean([grid_per_seed[s][i]["worst_day_loss"] for s in range(len(SEEDS))]))
        merged.append(b)
    merged.sort(key=lambda r: r["mean_dev_loss"])
    best = merged[0]["mean_dev_loss"]
    tied = [r for r in merged if r["mean_dev_loss"] - best < 1e-6]
    if len(tied) > 1:
        tied.sort(key=lambda r: (r["mean_worst_day"], r["eta_m"], -(_tau_key(r["tau"]))))
    win = tied[0]
    pd.DataFrame(merged).to_csv(out / "apops_grid.csv", index=False)

    # single-memory pick at the winning ONS config
    sm = []
    m = len(smem_per_seed[0])
    for i in range(m):
        b = {k: smem_per_seed[0][i][k] for k in ("ons_lam", "ons_eta")}
        b["mean_S"] = float(np.mean([smem_per_seed[s][i]["S_dev_loss"] for s in range(len(SEEDS))]))
        b["mean_L"] = float(np.mean([smem_per_seed[s][i]["L_dev_loss"] for s in range(len(SEEDS))]))
        sm.append(b)
    pd.DataFrame(sm).to_csv(out / "apops_single_memory.csv", index=False)
    at_win = next(r for r in sm if r["ons_lam"] == win["ons_lam"] and r["ons_eta"] == win["ons_eta"])
    single_pick = "S" if at_win["mean_S"] <= at_win["mean_L"] else "L"

    l_ops = float(np.mean([m["l_ops"] for m in margin_per_seed]))
    l_baseq = float(np.mean([m["l_baseq"] for m in margin_per_seed]))
    delta_ni = 0.10 * abs(l_ops - l_baseq)

    selected = {
        "source": args.source, "block_sec": block_sec, "delay_sec": delay_sec,
        "ops_config_ref": str(args.ops_config),
        "shared_mixture": mix, "ops": ops_hp,
        "ons_lam": win["ons_lam"], "ons_eta": win["ons_eta"],
        "eta_m": win["eta_m"],
        "switch_half_life_h": (1e12 if not np.isfinite(win["tau"]) else win["tau"]),
        "short_half_life_h": SHORT_HL_H, "long_half_life_h": LONG_HL_H,
        "a_bounds": [0.2, 5.0], "b_bounds": [-0.25, 0.25],
        "single_memory_pick": single_pick,
        "init_weights": [0.50, 0.25, 0.25],
        "decision_rule": {"l_ops_dev": l_ops, "l_baseq_dev": l_baseq, "delta_NI": delta_ni},
        "seeds": list(SEEDS), "runtime_s": time.time() - t0,
    }
    (out / "apops_selected.json").write_text(json.dumps(selected, indent=2, default=float))
    print(f"\nselected: ons_lam={win['ons_lam']} ons_eta={win['ons_eta']} "
          f"eta_m={win['eta_m']} tau={win['tau']} single_memory={single_pick}", flush=True)
    print(f"delta_NI = {delta_ni:.6g}  (L_OPS,dev={l_ops:.6f}  L_base-q,dev={l_baseq:.6f})", flush=True)
    print(f"-> {out}/apops_selected.json   ({time.time() - t0:.1f}s)", flush=True)


def _tau_key(tau):
    return 1e9 if tau == float("inf") or tau == "Infinity" else float(tau)


if __name__ == "__main__":
    main()
