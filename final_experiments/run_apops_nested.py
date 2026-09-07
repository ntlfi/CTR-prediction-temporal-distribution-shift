"""Fully nested rolling-origin evaluation (2026-09-06 additional-experiments
spec).  Four headline methods on identical cross-day q_{d,i}:

  ops             projected gradient, reset daily      (existing reference)
  reset_ons       discounted ONS, reset daily          (optimizer change only)
  persistent_ons  discounted ONS, carried across days  (value of persistence)
  ap_ops          reset R + persistent S/L, adaptively aggregated (lambda_AP)

plus the no-slope ablation.  Outer origins: Criteo days 16-30, Avazu 5-9.

For each outer day d: history = days < d; inner validation = the trailing
3 eligible days; select ONE common configuration for all three seeds by
mean inner-validation loss across seeds 0/1/2; replay each method
chronologically through days <= d (absolute-time feedback queue, so
cross-midnight maturation is handled); score day d exactly once; day d
influences only later origins.

Re-selected per origin (common across seeds):
  * shared cross-day mixture  (eta x halflife, 15 configs; on uncalibrated inner loss)
  * AP-OPS  (lambda_AP in {0, .25, .5, .75, 1}) x (tau in {4, 16, inf} h)  -- 15, grid CONTAINS lambda_AP = 0
  * persistent / reset ONS half-life  (in {4, 16} h)

Frozen (from the dataset's apops_selected.json / selected_configs.json,
dev only): eta_m, ONS (lam, eta), OPS (B, eta0, schedule), block_sec,
delay_sec, S/L half-lives 4h/16h, adaptive-mass prior shape.

Two seed-loops (memory: full-data Avazu is ~40M rows x 3 seeds -- never
hold more than one seed's bank).  Loop 1 picks the common mixture per
origin; loop 2 finishes selection and scores day d for every candidate,
then the frozen pick is looked up.  Output mirrors run_rolling.py.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from twoscale.data import load
from twoscale.longterm import HORIZONS, build_bank
from twoscale.metrics import impression_weighted_logloss, per_day_frame
from twoscale_run import DATA_PATHS

from methods import adaptive_q_by_day
from apops.method import (APOPSConfig, SHORT_HL_H, LONG_HL_H, build_ap_experts,
                          ops_row, reset_ons_row)
from apops.aggregate import MetaConfig, aggregate

SEEDS = (0, 1, 2)
INNER_K = 3
OUTER_DAYS = {"criteo": list(range(16, 31)), "avazu": list(range(5, 10))}

MIX_ETA_GRID = [10.0, 30.0, 60.0, 150.0, 1e6]
MIX_HL_GRID = [3.0, 5.0, 10.0]
LAMBDA_AP_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]          # MUST contain 0
TAU_GRID = [4.0, 16.0, float("inf")]
PERSIST_HL_GRID = [SHORT_HL_H, LONG_HL_H]
METHODS = ["ops", "reset_ons", "persistent_ons", "ap_ops", "no_slope"]


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def inner_days_for(d, bank_days):
    eligible = [e for e in bank_days if e < d and any(x < e for x in bank_days)]
    return eligible[-INNER_K:]


def iw_on(records, day_set):
    return impression_weighted_logloss([r for r in records if r["day"] in day_set])


def tau_key(t):
    return "inf" if not np.isfinite(t) else f"{t:g}"


# --------------------------------------------------------------------------- #
def loop1_mixture(source, data_path, sample_frac, n_features, warmup, n_jobs, seed, outer_days):
    """Per seed: uncalibrated inner-val loss for every mixture, per origin."""
    ds = load(source, data_path, n_features=n_features, sample_frac=sample_frac, seed=seed)
    outer_days = [d for d in outer_days if d < ds.n_days]
    bank_days = list(range(1, max(outer_days) + 1))
    bank = build_bank(ds, bank_days, seed=seed, n_jobs=n_jobs, verbose=False)
    bank_days = sorted(bank)
    print(f"  loop1 seed {seed}: {len(ds.y):,} rows", flush=True)

    rows = {}
    for d in outer_days:
        inner = inner_days_for(d, bank_days)
        prefix_hist = [e for e in bank_days if e < d]
        iset = set(inner)
        for eta, hl in product(MIX_ETA_GRID, MIX_HL_GRID):
            q, _ = adaptive_q_by_day(bank, prefix_hist, eta=eta, halflife=hl)
            recs = [{"day": e, "y": bank[e].y, "p": q[e]} for e in inner if e in q]
            rows[(d, eta, hl)] = impression_weighted_logloss(recs)
    del ds, bank
    return rows


def loop2_select_and_score(source, data_path, sample_frac, n_features, warmup, n_jobs,
                           seed, outer_days, mix_star, apops_cfg_base, ops_hp):
    """Per seed: with the common mixture per origin fixed, replay every
    method over days<=d; store inner-val loss for every candidate config
    and the day-d per-day metric for every candidate config."""
    ds = load(source, data_path, n_features=n_features, sample_frac=sample_frac, seed=seed)
    outer_days = [d for d in outer_days if d < ds.n_days]
    bank_days = list(range(1, max(outer_days) + 1))
    bank = build_bank(ds, bank_days, seed=seed, n_jobs=n_jobs, verbose=False)
    bank_days = sorted(bank)
    print(f"  loop2 seed {seed}: {len(ds.y):,} rows", flush=True)

    inner_loss = {}     # (d, method, cfgkey) -> inner-val iw log loss
    dayd = {}           # (d, method, cfgkey) -> per_day_frame row for day d

    for d in outer_days:
        inner = inner_days_for(d, bank_days)
        iset = set(inner)
        prefix = [e for e in bank_days if e <= d]
        eta, hl = mix_star[d]
        q_by_day, _ = adaptive_q_by_day(bank, prefix, eta=eta, halflife=hl)
        cfg = apops_cfg_base

        def stash(method, cfgkey, recs):
            inner_loss[(d, method, cfgkey)] = iw_on(recs, iset)
            rd = [r for r in recs if r["day"] == d]
            dayd[(d, method, cfgkey)] = per_day_frame(rd)[0] if rd else None

        # ops
        ops_recs, _ = ops_row(bank, prefix, q_by_day, ops_hp, cfg.block_sec, cfg.delay_sec)
        stash("ops", "-", ops_recs)

        # persistent S / L (also the ap_ops experts) + reset S / L
        experts, _ = build_ap_experts(bank, prefix, q_by_day, cfg, ops_hp, learn_slope=True)
        ns_experts, _ = build_ap_experts(bank, prefix, q_by_day, cfg, ops_hp, learn_slope=False)
        reset_recs = {hlv: reset_ons_row(bank, prefix, q_by_day, cfg, hlv)[0] for hlv in PERSIST_HL_GRID}
        persist_recs = {SHORT_HL_H: experts["S"], LONG_HL_H: experts["L"]}

        for hlv in PERSIST_HL_GRID:
            stash("persistent_ons", f"hl{hlv:g}", persist_recs[hlv])
            stash("reset_ons", f"hl{hlv:g}", reset_recs[hlv])

        # ap_ops / no_slope over the meta grid
        for lam, tau in product(LAMBDA_AP_GRID, TAU_GRID):
            mcfg = MetaConfig(eta_m=cfg.eta_m, switch_half_life_h=tau,
                              block_sec=cfg.block_sec, delay_sec=cfg.delay_sec, lambda_ap=lam)
            ap_recs, _ = aggregate(experts, bank, prefix, mcfg)
            stash("ap_ops", f"lam{lam:g}_tau{tau_key(tau)}", ap_recs)
            ns_recs, _ = aggregate(ns_experts, bank, prefix, mcfg)
            stash("no_slope", f"lam{lam:g}_tau{tau_key(tau)}", ns_recs)
        print(f"    loop2 seed {seed} origin {d}: mix=({eta:g},{hl:g})", flush=True)

    del ds, bank
    return inner_loss, dayd


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["criteo", "avazu"], required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--n-features", type=int, default=2 ** 18)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--config", required=True, help="apops_selected.json (for frozen eta_m/ONS/OPS/mixture-shape)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sel = json.loads(Path(args.config).read_text())
    assert sel["source"] == args.source
    block_sec, delay_sec = sel["block_sec"], sel["delay_sec"]
    ops_hp = sel["ops"]
    apops_cfg_base = APOPSConfig(block_sec=block_sec, delay_sec=delay_sec,
                                 ons_lam=sel["ons_lam"], ons_eta=sel["ons_eta"],
                                 eta_m=sel["eta_m"], a_bounds=tuple(sel["a_bounds"]),
                                 b_bounds=tuple(sel["b_bounds"]))
    warmup = args.warmup if args.warmup is not None else (4 if args.source == "criteo" else 3)
    data_path = args.data or DATA_PATHS[args.source]
    outer = OUTER_DAYS[args.source]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- loop 1: common mixture per origin --------------------------------
    mix_rows = []
    for seed in SEEDS:
        r = loop1_mixture(args.source, data_path, args.sample_frac, args.n_features,
                          warmup, args.n_jobs, seed, outer)
        for (d, eta, hl), loss in r.items():
            mix_rows.append({"origin": d, "eta": eta, "halflife": hl, "seed": seed, "inner_loss": loss})
    mixdf = pd.DataFrame(mix_rows)
    mix_star = {}
    for d, g in mixdf.groupby("origin"):
        m = g.groupby(["eta", "halflife"])["inner_loss"].mean().reset_index()
        best = m.loc[m["inner_loss"].idxmin()]
        mix_star[int(d)] = (float(best["eta"]), float(best["halflife"]))
    mixdf.to_csv(out / "nested_mixture_inner.csv", index=False)
    print("common mixture per origin:", {k: v for k, v in mix_star.items()}, flush=True)

    # ---- loop 2: selection + scoring ------------------------------------
    all_inner, all_dayd = [], []
    for seed in SEEDS:
        il, dd = loop2_select_and_score(args.source, data_path, args.sample_frac, args.n_features,
                                        warmup, args.n_jobs, seed, outer, mix_star,
                                        apops_cfg_base, ops_hp)
        for (d, method, ck), loss in il.items():
            all_inner.append({"origin": d, "method": method, "cfg": ck, "seed": seed, "inner_loss": loss})
        for (d, method, ck), row in dd.items():
            if row is not None:
                all_dayd.append({"origin": d, "method": method, "cfg": ck, "seed": seed, **row})
    inner_df = pd.DataFrame(all_inner)
    dayd_df = pd.DataFrame(all_dayd)
    inner_df.to_csv(out / "nested_inner_losses.csv", index=False)

    # ---- freeze one common config per origin --------------------------
    def pick(method, origin, restrict=None):
        g = inner_df[(inner_df.method == method) & (inner_df.origin == origin)]
        if restrict is not None:
            g = g[g.cfg.isin(restrict)]
        mean = g.groupby("cfg")["inner_loss"].mean()
        return mean.idxmin()

    manifest, per_day_rows = [], []
    for d in [x for x in outer if x < 1000]:
        if not len(inner_df[inner_df.origin == d]):
            continue
        hl_star = pick("persistent_ons", d)                       # "hl4" / "hl16"
        ap_star = pick("ap_ops", d)                               # "lam..._tau..."
        cfg_for = {"ops": "-", "reset_ons": hl_star, "persistent_ons": hl_star,
                   "ap_ops": ap_star, "no_slope": ap_star}
        lam_star = float(ap_star.split("lam")[1].split("_")[0])
        manifest.append({"origin": d, "mix_eta": mix_star[d][0], "mix_halflife": mix_star[d][1],
                         "persist_half_life": hl_star, "ap_ops_cfg": ap_star, "lambda_ap": lam_star})
        for method in METHODS:
            ck = cfg_for[method]
            sub = dayd_df[(dayd_df.origin == d) & (dayd_df.method == method) & (dayd_df.cfg == ck)]
            for seed in SEEDS:
                r = sub[sub.seed == seed]
                if len(r):
                    row = r.iloc[0]
                    per_day_rows.append({"seed": int(seed), "method": method, "day": int(d),
                                         "n": int(row["n"]), "log_loss": float(row["log_loss"]),
                                         "brier": float(row["brier"]), "ece": float(row["ece"]),
                                         "clicks": int(row["clicks"])})

    pdf = pd.DataFrame(per_day_rows)
    for seed in SEEDS:
        sd = out / f"seed{seed}"
        sd.mkdir(parents=True, exist_ok=True)
        pdf[pdf.seed == seed].drop(columns=["seed"]).to_csv(sd / "per_day_metrics.csv", index=False)
    pd.DataFrame(manifest).to_csv(out / "nested_origin_manifest.csv", index=False)

    lam_used = [m["lambda_ap"] for m in manifest]
    (out / "summary.json").write_text(json.dumps({
        "source": args.source, "config": str(args.config), "seeds": list(SEEDS),
        "outer_days": outer, "inner_k": INNER_K, "code_commit": git_commit_hash(),
        "frozen": {"eta_m": sel["eta_m"], "ons_lam": sel["ons_lam"], "ons_eta": sel["ons_eta"],
                   "ops": ops_hp, "block_sec": block_sec, "delay_sec": delay_sec,
                   "S_L_half_lives_h": [SHORT_HL_H, LONG_HL_H]},
        "reselected_per_origin": ["shared_mixture(eta,halflife)", "lambda_AP", "tau",
                                  "persistent/reset ONS half-life"],
        "lambda_ap_grid": LAMBDA_AP_GRID, "lambda_ap_selected_per_origin": lam_used,
        "lambda_ap_gt0_fraction": float(np.mean([x > 0 for x in lam_used])) if lam_used else float("nan"),
        "runtime_s": time.time() - t0,
    }, indent=2, default=float))
    print(f"\nlambda_AP selected per origin: {lam_used}", flush=True)
    print(f"lambda_AP > 0 on {np.sum([x > 0 for x in lam_used])}/{len(lam_used)} origins", flush=True)
    print(f"runtime {time.time() - t0:.1f}s -> {out}/", flush=True)


if __name__ == "__main__":
    main()
