"""Corrected fully-nested rolling-origin evaluation of the 6 main-table
TTAM methods + the 2x2 ablation (TTAM_FROZEN.md).

Every data-tuned setting is chosen from data strictly before the
evaluated origin.  For each outer origin ``d`` (Criteo 16-30, Avazu 5-9;
zero-based):

  * inner validation = up to the three latest eligible days before ``d``
    (the first Avazu origin has only two);
  * one configuration is chosen per knob, shared by seeds 0/1/2, by
    **mean inner-validation impression-weighted log loss across seeds**;
  * the chosen configuration's state is replayed over the whole prefix
    ``<= d`` to initialise it, then day ``d`` is scored exactly once;
  * day ``d`` influences only later origins.

Staged selection:
  pass 1  -- historical module (BFW h*, arw delta, adamoe lambda, amgtp
             rho, fixed-mixture weights) on *uncalibrated* inner loss.
  pass 2a -- the AP-OPS **reset-anchor** daily-reset-OPS settings, one
             grid per historical stream (fixed mixture / expanding /
             AMG-TP), on that stream's causal inner predictions.
  pass 2b -- the AP-OPS settings (ons_lam, ons_eta, lambda_AP, tau **and
             the meta learning rate eta_m**) with the anchor from 2a
             fixed.
  pass 3  -- replay the frozen winners, score the origin day, store it.

Label maturity (30 min / 1800 s) is enforced *everywhere* a cutoff reads
labels (``final_experiments/ttam/maturity.py``): the expert-bank fits
(day ``d-1``'s post-84600 s tail excluded), the inner-validation scoring
(``iw_on``), AMG-TP's state summaries and the ARW / AdaMoE loss histories,
in addition to the online calibrator queues that already had it. Pending
labels are kept and picked up by the next cutoff that follows their
arrival.

Output: ``seed{k}/per_day_metrics.csv``, ``nested_origin_manifest.csv``,
``selected_anchors.json``, ``selected_configs.json``,
``final_predictions/``, ``summary.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from itertools import product
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
import pandas as pd

from twoscale.metrics import impression_weighted_logloss, per_day_frame

from final_experiments.ttam.bank import HORIZONS5
from final_experiments.ttam.build_banks import load_or_build, MAX_DAY
from final_experiments.ttam.fixed_mix import (effective_weight_report,
                                              fit_simplex_weights, fixed_mixture_q)
from final_experiments.ttam.amgtp import AMGTPConfig, run_amgtp5
from final_experiments.ttam import method as V
from final_experiments.ttam.method import apops_config, apops_on
from final_experiments.ttam.maturity import available_mask
from apops.method import build_ap_experts
from apops.aggregate import MetaConfig, aggregate

SEEDS = (0, 1, 2)
INNER_K = 3
OUTER_DAYS = {"criteo": list(range(16, 31)), "avazu": list(range(5, 10))}
BLOCK_SEC = {"criteo": 900, "avazu": 3600}
DELAY_SEC = 1800
WARMUP = {"criteo": 4, "avazu": 3}

# ---- selection grids (frozen before results -- see TTAM_FROZEN.md) --------- #
ARW_DELTA_GRID = [0.05, 0.10, 0.20]
ADAMOE_LAMBDA_GRID = [0.0, 0.25, 0.50, 0.75, 0.99]
AMGTP_RHO_GRID = [0.2, 0.3, 0.5]                    # gate/persistence memory rate

# daily-reset OPS grid -- used both for the main-table `ops` arm and, per
# origin per historical stream, for the AP-OPS **reset-anchor** R (pass 2a).
OPS_B_GRID = [0.25, 0.5, 1.0]
OPS_ETA0_GRID = [0.03, 0.1, 0.3]
OPS_SCHED_GRID = ["const", "inv_sqrt"]

ONS_LAM_GRID = [0.1, 1.0]
ONS_ETA_GRID = [0.25, 1.0, 4.0]
APOPS_LAMBDA_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]
APOPS_TAU_GRID = [4.0, 16.0, float("inf")]
APOPS_ETA_M_GRID = [30.0, 100.0, 300.0]             # meta learning rate -- now selected per origin (pass 2b)

SEL_BOUNDS = {"a_bounds": [0.2, 5.0], "b_bounds": [-0.25, 0.25]}   # persistent-Platt (a,b) box, fixed a priori
OPS_HP_BASE = {"a_bounds": (0.2, 5.0)}
S_L_HALF_LIVES = (4.0, 16.0)                        # S/L expert memory, fixed a priori (AP-OPS component study)
# Main table (6) + the 2x2 ablation. The ablation crosses
#   historical predictor in {AMG-TP, expanding-history}  x  calibration in {AP-OPS, none}
# giving: ttam (AMG-TP + AP-OPS) / amgtp_only (AMG-TP + none) /
#         expanding_apops (expanding + AP-OPS) / expanding (expanding + none).
# ``expanding`` and ``ttam`` are shared with the main table, so only
# ``expanding_apops`` and ``amgtp_only`` are ablation-only. Identical
# historical predictions are shared within each row: amgtp_only and ttam
# both feed q_amgtp; expanding and expanding_apops both feed
# bank[d].preds["expanding"].
ALL_VARIANTS = ["expanding", "best_fixed_window", "arw", "adamoe", "ops",
                "ttam", "amgtp_only", "expanding_apops"]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _limit_worker_threads():
    """Each parallel origin is CPU-bound serial Python (Platt replay) plus
    small numpy/torch ops; keep BLAS/torch single-threaded inside a worker
    so N workers do not oversubscribe the machine."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def tau_key(t):
    return "inf" if not np.isfinite(t) else f"{t:g}"


def inner_days_for(d: int, bank_days, warmup: int):
    eligible = [e for e in bank_days if e < d and e >= warmup and any(x < e for x in bank_days)]
    return eligible[-INNER_K:]


def iw_on(records, day_set, origin):
    """Impression-weighted inner-validation log loss over ``day_set``,
    restricted to labels available by the start of day ``origin`` -- the
    latest inner day's post-``86400 - DELAY_SEC`` tail is excluded."""
    sub = []
    for r in records:
        e = r["day"]
        if e not in day_set:
            continue
        sec = np.asarray(r["sec_in_day"])
        avail = available_mask(np.full(len(sec), e, np.int64), sec, origin, DELAY_SEC)
        if not avail.any():
            continue
        sub.append({"y": np.asarray(r["y"])[avail], "p": np.asarray(r["p"], float)[avail]})
    return impression_weighted_logloss(sub)


def amgtp_cfg(rho: float, delay_sec: int) -> AMGTPConfig:
    return AMGTPConfig(rho=rho, delay_sec=delay_sec)


# --------------------------------------------------------------------------- #
#  pass 1 -- historical-module selection (uncalibrated inner loss)             #
# --------------------------------------------------------------------------- #
def _pass1_origin(bank, ctx, d, bank_days, warmup, seed):
    """Uncalibrated inner-validation loss of every historical-module
    candidate for one outer origin ``d``. Pure function of the frozen
    grids + the bank -> safe to evaluate origins in parallel."""
    _limit_worker_threads()
    inner = inner_days_for(d, bank_days, warmup)
    iset = set(inner)
    hist = [e for e in bank_days if e < d]
    rec = {"inner_days": inner}

    w = fit_simplex_weights(bank, inner)
    rec["simplex_w"] = w.tolist()

    rec["bfw"] = {h: iw_on(V.q_records(bank, hist, {e: bank[e].preds[h] for e in hist}), iset, d)
                  for h in HORIZONS5}
    rec["arw"] = {}
    for delta in ARW_DELTA_GRID:
        recs, _ = V.arw(bank, hist, delta=delta)
        rec["arw"][f"{delta:g}"] = iw_on(recs, iset, d)
    rec["adamoe"] = {}
    for lam in ADAMOE_LAMBDA_GRID:
        recs, _ = V.adamoe(bank, hist, lam=lam)
        rec["adamoe"][f"{lam:g}"] = iw_on(recs, iset, d)
    rec["amgtp"] = {}
    for rho in AMGTP_RHO_GRID:
        qa, _ = run_amgtp5(bank, ctx, hist, amgtp_cfg(rho, DELAY_SEC), seed=seed)
        recs = V.q_records(bank, hist, qa)
        rec["amgtp"][f"{rho:g}"] = iw_on(recs, iset, d)
    print(f"    pass1 origin {d}: inner={inner}", flush=True)
    return d, rec


def pass1_seed(bank, ctx, outer_days, warmup, seed, n_workers=1):
    bank_days = sorted(bank)
    todo = [d for d in outer_days if d in bank]
    results = Parallel(n_jobs=n_workers, prefer="processes")(
        delayed(_pass1_origin)(bank, ctx, d, bank_days, warmup, seed) for d in todo)
    return dict(results)


def pick_modules(pass1: dict, outer_days) -> dict:
    """Mean inner loss across seeds -> one shared choice per origin."""
    sel = {}
    for d in outer_days:
        per_seed = [pass1[str(s)][str(d)] for s in SEEDS if str(d) in pass1[str(s)]]
        if not per_seed:
            continue

        def best(key, cands):
            means = {c: float(np.mean([ps[key][c] for ps in per_seed])) for c in cands}
            return min(means, key=means.get), means

        h_star, h_means = best("bfw", list(HORIZONS5))
        delta_star, d_means = best("arw", [f"{x:g}" for x in ARW_DELTA_GRID])
        lam_star, l_means = best("adamoe", [f"{x:g}" for x in ADAMOE_LAMBDA_GRID])
        rho_star, r_means = best("amgtp", [f"{x:g}" for x in AMGTP_RHO_GRID])
        # simplex weights: the plan fits one shared vector -> average the
        # per-seed fits (all fitted on the identical inner days; the fit is
        # deterministic, differences are only the seed's own backbone)
        w = np.mean([ps["simplex_w"] for ps in per_seed], axis=0)
        w = (w / w.sum()).tolist()
        sel[str(d)] = {
            "inner_days": per_seed[0]["inner_days"],
            "best_fixed_window": h_star, "arw_delta": float(delta_star),
            "adamoe_lambda": float(lam_star), "amgtp_rho": float(rho_star),
            "simplex_w": w,
            "inner_means": {"bfw": h_means, "arw": d_means, "adamoe": l_means, "amgtp": r_means},
        }
    return sel


# --------------------------------------------------------------------------- #
#  pass 2 -- calibration selection on the chosen module's inner stream         #
# --------------------------------------------------------------------------- #
def _q_expanding(bank, days) -> dict:
    """The plain expanding-history expert prediction as a q-stream -- the
    2x2 ablation's fixed 'window' predictor, shared verbatim by its
    calibrated (expanding_apops) and uncalibrated (expanding) cells."""
    return {int(e): np.asarray(bank[e].preds["expanding"], float)
            for e in sorted(int(x) for x in days if x in bank)}


def _q_streams(bank, ctx, days, modsel, seed):
    w = np.asarray(modsel["simplex_w"], float)
    q_fixed = fixed_mixture_q(bank, days, w)              # for the main-table `ops` arm
    q_exp = _q_expanding(bank, days)                      # for the 2x2 expanding cells
    qa, _ = run_amgtp5(bank, ctx, days, amgtp_cfg(modsel["amgtp_rho"], DELAY_SEC), seed=seed)
    return q_fixed, q_exp, qa


def _ops_grid_inner(bank, days, q_by_day, iset, block_sec, origin):
    rows = {}
    for B, eta0, sched in product(OPS_B_GRID, OPS_ETA0_GRID, OPS_SCHED_GRID):
        hp = {"B": B, "eta0": eta0, "schedule": sched, **OPS_HP_BASE}
        recs = V.ops(bank, days, q_by_day, hp, block_sec, DELAY_SEC)
        rows[f"B{B:g}_e{eta0:g}_{sched}"] = (iw_on(recs, iset, origin), hp)
    return rows


def _apops_grid_inner(bank, days, q_by_day, iset, block_sec, anchor_hp, origin):
    """AP-OPS grid with the reset-anchor OPS **fixed** to ``anchor_hp``
    (selected in pass 2a). (ons_lam, ons_eta) fix the persistent experts --
    built once per pair; (lambda_AP, tau, eta_m) are the meta knobs."""
    rows = {}
    for lam, eta in product(ONS_LAM_GRID, ONS_ETA_GRID):
        base = apops_config(SEL_BOUNDS, block_sec, DELAY_SEC, ons_lam=lam, ons_eta=eta,
                            eta_m=APOPS_ETA_M_GRID[0], lambda_ap=0.0, tau_h=4.0,
                            persist_hl_h=S_L_HALF_LIVES[0])
        experts, _ = build_ap_experts(bank, days, q_by_day, base, anchor_hp, learn_slope=True)
        for lap, tau, em in product(APOPS_LAMBDA_GRID, APOPS_TAU_GRID, APOPS_ETA_M_GRID):
            mcfg = MetaConfig(eta_m=em, switch_half_life_h=tau,
                              block_sec=block_sec, delay_sec=DELAY_SEC, lambda_ap=lap)
            recs, _ = aggregate(experts, bank, days, mcfg)
            key = f"lam{lam:g}_eta{eta:g}_lap{lap:g}_tau{tau_key(tau)}_em{em:g}"
            rows[key] = (iw_on(recs, set(iset), origin),
                         {"ons_lam": lam, "ons_eta": eta, "lambda_ap": lap, "tau_h": tau, "eta_m": em})
    return rows


# ---- pass 2a: reset-anchor OPS selection (one grid per historical stream) --- #
def _pass2a_origin(bank, ctx, d, bank_days, warmup, modsel, block_sec, seed):
    _limit_worker_threads()
    inner = inner_days_for(d, bank_days, warmup)
    iset = set(inner)
    hist = [e for e in bank_days if e < d]
    q_fixed, q_exp, q_amgtp = _q_streams(bank, ctx, hist, modsel, seed)
    rec = {
        "ops_fixed": {k: v[0] for k, v in _ops_grid_inner(bank, hist, q_fixed, iset, block_sec, d).items()},
        "ops_expanding": {k: v[0] for k, v in _ops_grid_inner(bank, hist, q_exp, iset, block_sec, d).items()},
        "ops_amgtp": {k: v[0] for k, v in _ops_grid_inner(bank, hist, q_amgtp, iset, block_sec, d).items()},
    }
    print(f"    pass2a origin {d}: seed {seed}", flush=True)
    return d, rec


# ---- pass 2b: AP-OPS selection with the anchor fixed (incl. eta_m) --------- #
def _pass2b_origin(bank, ctx, d, bank_days, warmup, modsel, anchors, block_sec, seed):
    _limit_worker_threads()
    inner = inner_days_for(d, bank_days, warmup)
    iset = set(inner)
    hist = [e for e in bank_days if e < d]
    q_exp = _q_expanding(bank, hist)
    q_amgtp, _ = run_amgtp5(bank, ctx, hist, amgtp_cfg(modsel["amgtp_rho"], DELAY_SEC), seed=seed)
    rec = {
        "apops_expanding": {k: v[0] for k, v in _apops_grid_inner(
            bank, hist, q_exp, iset, block_sec, anchors["anchor_expanding"], d).items()},
        "apops_amgtp": {k: v[0] for k, v in _apops_grid_inner(
            bank, hist, q_amgtp, iset, block_sec, anchors["anchor_amgtp"], d).items()},
    }
    print(f"    pass2b origin {d}: seed {seed}", flush=True)
    return d, rec


def pass2a_seed(bank, ctx, outer_days, warmup, modules, block_sec, seed, n_workers=1):
    bank_days = sorted(bank)
    todo = [d for d in outer_days if d in bank and str(d) in modules]
    r = Parallel(n_jobs=n_workers, prefer="processes")(
        delayed(_pass2a_origin)(bank, ctx, d, bank_days, warmup, modules[str(d)], block_sec, seed)
        for d in todo)
    return dict(r)


def pass2b_seed(bank, ctx, outer_days, warmup, modules, anchors, block_sec, seed, n_workers=1):
    bank_days = sorted(bank)
    todo = [d for d in outer_days if d in bank and str(d) in modules and str(d) in anchors]
    r = Parallel(n_jobs=n_workers, prefer="processes")(
        delayed(_pass2b_origin)(bank, ctx, d, bank_days, warmup, modules[str(d)],
                                anchors[str(d)], block_sec, seed)
        for d in todo)
    return dict(r)


def _decode_ops(key: str) -> dict:
    b, e, sched = key.split("_", 2)
    return {"B": float(b[1:]), "eta0": float(e[1:]), "schedule": sched, **OPS_HP_BASE}


def _decode_apops(key: str) -> dict:
    lam, eta, lap, tau, em = key.split("_")
    return {"ons_lam": float(lam[3:]), "ons_eta": float(eta[3:]), "lambda_ap": float(lap[3:]),
            "tau_h": (float("inf") if tau[3:] == "inf" else float(tau[3:])), "eta_m": float(em[2:])}


def _seed_mean_argmin(per_seed, key):
    cands = per_seed[0][key].keys()
    means = {c: float(np.mean([ps[key][c] for ps in per_seed])) for c in cands}
    return min(means, key=means.get)


def pick_anchors(pass2a: dict, modules: dict, outer_days) -> dict:
    """Per origin, the reset-anchor daily-reset-OPS config for each
    historical stream, shared across seeds (mean inner loss)."""
    sel = {}
    for d in outer_days:
        per_seed = [pass2a[str(s)][str(d)] for s in SEEDS if str(d) in pass2a.get(str(s), {})]
        if not per_seed:
            continue
        sel[str(d)] = {
            "ops": _decode_ops(_seed_mean_argmin(per_seed, "ops_fixed")),           # main-table `ops` arm
            "anchor_expanding": _decode_ops(_seed_mean_argmin(per_seed, "ops_expanding")),
            "anchor_amgtp": _decode_ops(_seed_mean_argmin(per_seed, "ops_amgtp")),
        }
    return sel


def pick_calibration(pass2b: dict, anchors: dict, modules: dict, outer_days) -> dict:
    sel = {}
    for d in outer_days:
        per_seed = [pass2b[str(s)][str(d)] for s in SEEDS if str(d) in pass2b.get(str(s), {})]
        if not per_seed or str(d) not in anchors:
            continue
        cfg = dict(modules[str(d)])
        cfg["ops"] = anchors[str(d)]["ops"]
        cfg["anchor_expanding"] = anchors[str(d)]["anchor_expanding"]
        cfg["anchor_amgtp"] = anchors[str(d)]["anchor_amgtp"]
        cfg["apops_expanding"] = _decode_apops(_seed_mean_argmin(per_seed, "apops_expanding"))
        cfg["apops_amgtp"] = _decode_apops(_seed_mean_argmin(per_seed, "apops_amgtp"))
        sel[str(d)] = cfg
    return sel


# --------------------------------------------------------------------------- #
#  pass 3 -- replay winners, score origin day, store predictions               #
# --------------------------------------------------------------------------- #
def score_origin(bank, ctx, d, warmup, sel, block_sec, seed):
    bank_days = sorted(bank)
    prefix = [e for e in bank_days if e <= d]
    hist = [e for e in bank_days if e < d]
    w = np.asarray(sel["simplex_w"], float)
    q_fixed = fixed_mixture_q(bank, prefix, w)            # main-table `ops` arm only
    q_exp = _q_expanding(bank, prefix)                    # 2x2 expanding cells
    q_amgtp, amg_trace = run_amgtp5(bank, ctx, prefix, amgtp_cfg(sel["amgtp_rho"], DELAY_SEC), seed=seed)

    ae, aa = sel["apops_expanding"], sel["apops_amgtp"]
    ac_exp = apops_config(SEL_BOUNDS, block_sec, DELAY_SEC,
                          ons_lam=ae["ons_lam"], ons_eta=ae["ons_eta"], eta_m=ae["eta_m"],
                          lambda_ap=ae["lambda_ap"], tau_h=ae["tau_h"], persist_hl_h=S_L_HALF_LIVES[0])
    ac_amgtp = apops_config(SEL_BOUNDS, block_sec, DELAY_SEC,
                            ons_lam=aa["ons_lam"], ons_eta=aa["ons_eta"], eta_m=aa["eta_m"],
                            lambda_ap=aa["lambda_ap"], tau_h=aa["tau_h"], persist_hl_h=S_L_HALF_LIVES[0])

    streams = {
        "expanding": V.expanding(bank, prefix),
        "best_fixed_window": V.best_fixed_window(bank, prefix, sel["best_fixed_window"]),
        "arw": V.arw(bank, prefix, delta=sel["arw_delta"])[0],
        "adamoe": V.adamoe(bank, prefix, lam=sel["adamoe_lambda"])[0],
        "ops": V.ops(bank, prefix, q_fixed, sel["ops"], block_sec, DELAY_SEC),
        # 2x2 ablation: {AMG-TP, expanding} x {AP-OPS, none}
        "amgtp_only": V.q_records(bank, prefix, q_amgtp),
        "expanding_apops": apops_on(bank, prefix, q_exp, ac_exp, sel["anchor_expanding"])[0],
        "ttam": apops_on(bank, prefix, q_amgtp, ac_amgtp, sel["anchor_amgtp"])[0],
    }
    day_rows, preds = [], {}
    for name, recs in streams.items():
        rd = [r for r in recs if r["day"] == d]
        if not rd:
            continue
        fr = per_day_frame(rd)[0]
        day_rows.append({"method": name, **fr})
        preds[name] = np.asarray(rd[0]["p"], float)
    y = np.asarray(bank[d].y, float)
    sec = np.asarray(bank[d].sec_in_day, float)
    beta_d = next((t["beta"] for t in amg_trace if t["day"] == d), float("nan"))
    return day_rows, preds, y, sec, beta_d


def _pass3_origin(bank, ctx, d, warmup, sc, block_sec, seed, pred_dir, want_manifest):
    """Replay the frozen winners over the prefix, score origin ``d`` once,
    write its per-origin prediction npz. Origins are independent; only the
    (origin, seed) npz file is a side effect and each is written once."""
    _limit_worker_threads()
    day_rows, preds, y, sec, beta_d = score_origin(bank, ctx, d, warmup, sc, block_sec, seed)
    np.savez_compressed(Path(pred_dir) / f"origin{d}_seed{seed}.npz",
                        y=y.astype(np.int8), sec=sec.astype(np.int64),
                        **{m: preds[m].astype(np.float32) for m in preds})
    mrow = None
    if want_manifest:
        mrow = {
            "origin": d, "inner_days": str(sc["inner_days"]),
            "best_fixed_window": sc["best_fixed_window"], "arw_delta": sc["arw_delta"],
            "adamoe_lambda": sc["adamoe_lambda"], "amgtp_rho": sc["amgtp_rho"],
            "n_effective_experts": bank[d].n_effective,
            "simplex_w": str([round(x, 4) for x in sc["simplex_w"]]),
            "fixed_effective_w": str(effective_weight_report(bank, d, np.asarray(sc["simplex_w"]))),
            "ops": str(sc["ops"]), "apops_expanding": str(sc["apops_expanding"]),
            "apops_amgtp": str(sc["apops_amgtp"]), "ttam_beta_origin": round(float(beta_d), 4)}
    print(f"    pass3 seed {seed} origin {d}: scored", flush=True)
    return d, day_rows, mrow


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["criteo", "avazu"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-features", type=int, default=2 ** 18)
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--cache-dir", default="final_experiments/ttam/_bankcache")
    ap.add_argument("--n-jobs", type=int, default=3, help="threads for the bank fit (build_bank5)")
    ap.add_argument("--n-workers", type=int, default=1,
                    help="parallel outer origins per pass (each pass's origins are independent)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-pass", type=int, default=None, help="run just pass 1/2/3 then stop")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    outer = [d for d in OUTER_DAYS[args.source] if d < MAX_DAY[args.source]]
    warmup = WARMUP[args.source]
    block_sec = BLOCK_SEC[args.source]
    t0 = time.time()

    def bank_for(seed):
        return load_or_build(args.source, args.data, seed, args.n_features, args.sample_frac,
                             Path(args.cache_dir), args.n_jobs, want_context=True)

    # ---- pass 1 --------------------------------------------------------
    p1_path = out / "pass1_module_inner.json"
    if p1_path.exists():
        pass1 = json.loads(p1_path.read_text())
    else:
        pass1 = {}
        for seed in SEEDS:
            sp = out / f"pass1_seed{seed}.json"          # per-seed resume point
            if sp.exists():
                pass1[str(seed)] = json.loads(sp.read_text())
                print(f"  pass1 seed {seed}: resumed from {sp.name}", flush=True)
                continue
            bank, ctx = bank_for(seed)
            pass1[str(seed)] = {str(k): v for k, v in
                                pass1_seed(bank, ctx, outer, warmup, seed,
                                           n_workers=args.n_workers).items()}
            del bank, ctx
            sp.write_text(json.dumps(pass1[str(seed)], indent=2, default=float))
        p1_path.write_text(json.dumps(pass1, indent=2, default=float))
    modules = pick_modules(pass1, outer)
    (out / "selected_modules.json").write_text(json.dumps(modules, indent=2, default=float))
    if args.only_pass == 1:
        print(f"pass 1 done ({time.time() - t0:.0f}s)", flush=True)
        return

    # ---- pass 2a: reset-anchor OPS selection --------------------------
    def _run_seeded(path, seed_fn, per_seed_name):
        agg_path = out / path
        if agg_path.exists():
            return json.loads(agg_path.read_text())
        agg = {}
        for seed in SEEDS:
            sp = out / per_seed_name(seed)
            if sp.exists():
                agg[str(seed)] = json.loads(sp.read_text())
                print(f"  {sp.stem}: resumed", flush=True)
                continue
            bank, ctx = bank_for(seed)
            agg[str(seed)] = {str(k): v for k, v in seed_fn(bank, ctx, seed).items()}
            del bank, ctx
            sp.write_text(json.dumps(agg[str(seed)], indent=2, default=float))
        agg_path.write_text(json.dumps(agg, indent=2, default=float))
        return agg

    pass2a = _run_seeded(
        "pass2a_anchor_inner.json",
        lambda bank, ctx, seed: pass2a_seed(bank, ctx, outer, warmup, modules, block_sec, seed,
                                            n_workers=args.n_workers),
        lambda s: f"pass2a_seed{s}.json")
    anchors = pick_anchors(pass2a, modules, outer)
    (out / "selected_anchors.json").write_text(json.dumps(anchors, indent=2, default=float))

    # ---- pass 2b: AP-OPS selection (anchor fixed, eta_m in grid) ------
    pass2b = _run_seeded(
        "pass2b_apops_inner.json",
        lambda bank, ctx, seed: pass2b_seed(bank, ctx, outer, warmup, modules, anchors, block_sec, seed,
                                            n_workers=args.n_workers),
        lambda s: f"pass2b_seed{s}.json")
    selected = pick_calibration(pass2b, anchors, modules, outer)
    (out / "selected_configs.json").write_text(json.dumps(
        {"source": args.source, "code_commit": git_commit(), "outer_days": outer,
         "block_sec": block_sec, "delay_sec": DELAY_SEC,
         "eta_m_grid": APOPS_ETA_M_GRID, "anchor_ops_grid": "OPS_{B,eta0,sched}",
         "s_l_half_lives_h": list(S_L_HALF_LIVES), "per_origin": selected}, indent=2, default=float))
    if args.only_pass == 2:
        print(f"pass 2 done ({time.time() - t0:.0f}s)", flush=True)
        return

    # ---- pass 3 -------------------------------------------------------
    pred_dir = out / "final_predictions"
    pred_dir.mkdir(exist_ok=True)
    manifest_rows = []
    for seed in SEEDS:
        sd = out / f"seed{seed}"
        sd.mkdir(exist_ok=True)
        seed_csv = sd / "per_day_metrics.csv"
        if seed_csv.exists() and (out / "nested_origin_manifest.csv").exists():
            print(f"  pass3 seed {seed}: resumed from {seed_csv.name}", flush=True)
            continue
        bank, ctx = bank_for(seed)
        todo = [d for d in outer if d in bank and str(d) in selected]
        results = Parallel(n_jobs=args.n_workers, prefer="processes")(
            delayed(_pass3_origin)(bank, ctx, d, warmup, selected[str(d)], block_sec, seed,
                                   pred_dir, seed == SEEDS[0])
            for d in todo)
        seed_rows = []
        for d, day_rows, mrow in results:
            for r in day_rows:
                seed_rows.append({"day": int(d), **r})
            if mrow is not None:
                manifest_rows.append(mrow)
        pd.DataFrame(seed_rows).to_csv(seed_csv, index=False)
        del bank, ctx
    if manifest_rows:
        manifest_rows.sort(key=lambda r: r["origin"])
        pd.DataFrame(manifest_rows).to_csv(out / "nested_origin_manifest.csv", index=False)

    (out / "summary.json").write_text(json.dumps({
        "source": args.source, "code_commit": git_commit(), "seeds": list(SEEDS),
        "outer_days": outer, "inner_k": INNER_K, "n_features": args.n_features,
        "sample_frac": args.sample_frac, "block_sec": block_sec, "delay_sec": DELAY_SEC,
        "variants": ALL_VARIANTS, "runtime_s": time.time() - t0,
        "grids": {"arw_delta": ARW_DELTA_GRID, "adamoe_lambda": ADAMOE_LAMBDA_GRID,
                  "amgtp_rho": AMGTP_RHO_GRID, "ops_B": OPS_B_GRID, "ops_eta0": OPS_ETA0_GRID,
                  "ops_sched": OPS_SCHED_GRID, "ons_lam": ONS_LAM_GRID, "ons_eta": ONS_ETA_GRID,
                  "apops_lambda": APOPS_LAMBDA_GRID, "apops_tau": [tau_key(t) for t in APOPS_TAU_GRID],
                  "apops_eta_m": APOPS_ETA_M_GRID,
                  "anchor_ops": "per-origin, per-stream, from OPS_{B,eta0,sched}"},
    }, indent=2, default=float))
    print(f"\nnested TTAM run done in {time.time() - t0:.0f}s -> {out}/", flush=True)


if __name__ == "__main__":
    main()
