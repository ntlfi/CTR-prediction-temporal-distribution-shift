"""Corrected fully-nested rolling-origin evaluation of the nine TTAM
variants (plan sections 2-6).

Every data-tuned setting is chosen from data strictly before the
evaluated origin.  For each outer origin ``d`` (Criteo 16-30, Avazu 5-9;
zero-based):

  * inner validation = the three latest eligible days before ``d``;
  * one configuration is chosen per knob, shared by seeds 0/1/2, by
    **mean inner-validation log loss across the seeds** (plan section 3);
  * the chosen configuration's state is replayed over the whole prefix
    ``<= d`` to initialise it, then day ``d`` is scored exactly once;
  * day ``d`` influences only later origins.

Staged selection (plan section 6): the historical module is chosen first
on *uncalibrated* inner loss (pass 1), then the calibration settings on
the chosen module's causal inner prediction stream (pass 2); pass 3
replays the frozen winners and scores + stores the origin-day
predictions.  All three passes read the five-horizon bank and context
sketch from the disk cache (``final_experiments/ttam/build_banks.py``) so
the backbone is fitted once.

Maturity rule (plan section 3): a label is eligible only once its 30-min
delay has elapsed; this is enforced inside the online calibrators
(``apops`` absolute-time queue) and inside AMG-TP's period update
(``ttam.amgtp``).  Inner-validation days are wholly in the past of every
origin, so their labels are all matured for selection.

Output mirrors ``run_apops_nested.py`` / ``run_rolling.py``:
``seed{k}/per_day_metrics.csv``, ``nested_origin_manifest.csv``,
``selected_configs.json``, ``final_predictions/``, ``summary.json``.
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
from apops.method import build_ap_experts
from apops.aggregate import MetaConfig, aggregate

# The AP-OPS reset anchor R is the repo's verified Online Platt Scaling with
# its component-study setting, frozen (dataset-independent 2-parameter map);
# the standalone ``ops`` arm tunes OPS separately for its own fair best case.
ANCHOR_OPS_HP = {"B": 0.25, "eta0": 0.3, "schedule": "const", "a_bounds": (0.2, 5.0)}

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

OPS_B_GRID = [0.25, 0.5, 1.0]
OPS_ETA0_GRID = [0.03, 0.1, 0.3]
OPS_SCHED_GRID = ["const", "inv_sqrt"]

ONS_LAM_GRID = [0.1, 1.0]
ONS_ETA_GRID = [0.25, 1.0, 4.0]
APOPS_LAMBDA_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]
APOPS_TAU_GRID = [4.0, 16.0, float("inf")]
APOPS_ETA_M = 100.0                                 # fixed a priori (dataset-independent block-loss scale)

SEL_BOUNDS = {"a_bounds": [0.2, 5.0], "b_bounds": [-0.25, 0.25]}
OPS_HP_BASE = {"a_bounds": (0.2, 5.0)}
S_L_HALF_LIVES = (4.0, 16.0)
ALL_VARIANTS = ["expanding", "best_fixed_window", "arw", "adamoe", "ops",
                "ttam", "without_both", "amgtp_only", "apops_only"]


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


def iw_on(records, day_set):
    return impression_weighted_logloss([r for r in records if r["day"] in day_set])


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

    rec["bfw"] = {h: iw_on(V.q_records(bank, hist, {e: bank[e].preds[h] for e in hist}), iset)
                  for h in HORIZONS5}
    rec["arw"] = {}
    for delta in ARW_DELTA_GRID:
        recs, _ = V.arw(bank, hist, delta=delta)
        rec["arw"][f"{delta:g}"] = iw_on(recs, iset)
    rec["adamoe"] = {}
    for lam in ADAMOE_LAMBDA_GRID:
        recs, _ = V.adamoe(bank, hist, lam=lam)
        rec["adamoe"][f"{lam:g}"] = iw_on(recs, iset)
    rec["amgtp"] = {}
    for rho in AMGTP_RHO_GRID:
        qa, _ = run_amgtp5(bank, ctx, hist, amgtp_cfg(rho, DELAY_SEC), seed=seed)
        recs = V.q_records(bank, hist, qa)
        rec["amgtp"][f"{rho:g}"] = iw_on(recs, iset)
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
def _q_streams(bank, ctx, days, modsel, seed):
    w = np.asarray(modsel["simplex_w"], float)
    q_fixed = fixed_mixture_q(bank, days, w)
    qa, _ = run_amgtp5(bank, ctx, days, amgtp_cfg(modsel["amgtp_rho"], DELAY_SEC), seed=seed)
    return q_fixed, qa


def _ops_grid_inner(bank, days, q_by_day, iset, block_sec):
    rows = {}
    for B, eta0, sched in product(OPS_B_GRID, OPS_ETA0_GRID, OPS_SCHED_GRID):
        hp = {"B": B, "eta0": eta0, "schedule": sched, **OPS_HP_BASE}
        recs = V.ops(bank, days, q_by_day, hp, block_sec, DELAY_SEC)
        rows[f"B{B:g}_e{eta0:g}_{sched}"] = (iw_on(recs, iset), hp)
    return rows


def _apops_grid_inner(bank, days, q_by_day, iset, block_sec):
    """(ons_lam, ons_eta) fix the persistent experts; (lambda_AP, tau)
    are the meta knobs. Experts are built once per (lam, eta)."""
    rows = {}
    for lam, eta in product(ONS_LAM_GRID, ONS_ETA_GRID):
        base = apops_config(SEL_BOUNDS, block_sec, DELAY_SEC, ons_lam=lam, ons_eta=eta,
                            eta_m=APOPS_ETA_M, lambda_ap=0.0, tau_h=4.0,
                            persist_hl_h=S_L_HALF_LIVES[0])
        experts, _ = build_ap_experts(bank, days, q_by_day, base, ANCHOR_OPS_HP, learn_slope=True)
        for lap, tau in product(APOPS_LAMBDA_GRID, APOPS_TAU_GRID):
            mcfg = MetaConfig(eta_m=APOPS_ETA_M, switch_half_life_h=tau,
                              block_sec=block_sec, delay_sec=DELAY_SEC, lambda_ap=lap)
            recs, _ = aggregate(experts, bank, days, mcfg)
            key = f"lam{lam:g}_eta{eta:g}_lap{lap:g}_tau{tau_key(tau)}"
            rows[key] = (iw_on(recs, set(iset)),
                         {"ons_lam": lam, "ons_eta": eta, "lambda_ap": lap, "tau_h": tau})
    return rows


def _pass2_origin(bank, ctx, d, bank_days, warmup, modsel, block_sec, seed):
    """Calibration-grid inner loss for one outer origin, on the module
    stream chosen for it in pass 1. Independent across origins."""
    _limit_worker_threads()
    inner = inner_days_for(d, bank_days, warmup)
    iset = set(inner)
    hist = [e for e in bank_days if e < d]
    q_fixed, q_amgtp = _q_streams(bank, ctx, hist, modsel, seed)

    rec = {}
    rec["ops_fixed"] = {k: v[0] for k, v in _ops_grid_inner(bank, hist, q_fixed, iset, block_sec).items()}
    rec["apops_fixed"] = {k: v[0] for k, v in _apops_grid_inner(bank, hist, q_fixed, iset, block_sec).items()}
    rec["apops_amgtp"] = {k: v[0] for k, v in _apops_grid_inner(bank, hist, q_amgtp, iset, block_sec).items()}
    print(f"    pass2 origin {d}: seed {seed}", flush=True)
    return d, rec


def pass2_seed(bank, ctx, outer_days, warmup, modules, block_sec, seed, n_workers=1):
    bank_days = sorted(bank)
    todo = [d for d in outer_days if d in bank and str(d) in modules]
    results = Parallel(n_jobs=n_workers, prefer="processes")(
        delayed(_pass2_origin)(bank, ctx, d, bank_days, warmup, modules[str(d)], block_sec, seed)
        for d in todo)
    return dict(results)


def _decode_ops(key: str) -> dict:
    b, e, sched = key.split("_", 2)
    return {"B": float(b[1:]), "eta0": float(e[1:]), "schedule": sched, **OPS_HP_BASE}


def _decode_apops(key: str) -> dict:
    lam, eta, lap, tau = key.split("_")
    return {"ons_lam": float(lam[3:]), "ons_eta": float(eta[3:]), "lambda_ap": float(lap[3:]),
            "tau_h": (float("inf") if tau[3:] == "inf" else float(tau[3:]))}


def pick_calibration(pass2: dict, modules: dict, outer_days) -> dict:
    sel = {}
    for d in outer_days:
        per_seed = [pass2[str(s)][str(d)] for s in SEEDS if str(d) in pass2.get(str(s), {})]
        if not per_seed:
            continue

        def best(key):
            cands = per_seed[0][key].keys()
            means = {c: float(np.mean([ps[key][c] for ps in per_seed])) for c in cands}
            return min(means, key=means.get)

        cfg = dict(modules[str(d)])
        cfg["ops"] = _decode_ops(best("ops_fixed"))
        cfg["apops_fixed"] = _decode_apops(best("apops_fixed"))
        cfg["apops_amgtp"] = _decode_apops(best("apops_amgtp"))
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
    q_fixed = fixed_mixture_q(bank, prefix, w)
    q_amgtp, amg_trace = run_amgtp5(bank, ctx, prefix, amgtp_cfg(sel["amgtp_rho"], DELAY_SEC), seed=seed)

    ac_fixed = apops_config(SEL_BOUNDS, block_sec, DELAY_SEC,
                            ons_lam=sel["apops_fixed"]["ons_lam"], ons_eta=sel["apops_fixed"]["ons_eta"],
                            eta_m=APOPS_ETA_M, lambda_ap=sel["apops_fixed"]["lambda_ap"],
                            tau_h=sel["apops_fixed"]["tau_h"], persist_hl_h=S_L_HALF_LIVES[0])
    ac_amgtp = apops_config(SEL_BOUNDS, block_sec, DELAY_SEC,
                            ons_lam=sel["apops_amgtp"]["ons_lam"], ons_eta=sel["apops_amgtp"]["ons_eta"],
                            eta_m=APOPS_ETA_M, lambda_ap=sel["apops_amgtp"]["lambda_ap"],
                            tau_h=sel["apops_amgtp"]["tau_h"], persist_hl_h=S_L_HALF_LIVES[0])

    streams = {
        "expanding": V.expanding(bank, prefix),
        "best_fixed_window": V.best_fixed_window(bank, prefix, sel["best_fixed_window"]),
        "arw": V.arw(bank, prefix, delta=sel["arw_delta"])[0],
        "adamoe": V.adamoe(bank, prefix, lam=sel["adamoe_lambda"])[0],
        "ops": V.ops(bank, prefix, q_fixed, sel["ops"], block_sec, DELAY_SEC),
        "without_both": V.q_records(bank, prefix, q_fixed),
        "amgtp_only": V.q_records(bank, prefix, q_amgtp),
        "apops_only": apops_on(bank, prefix, q_fixed, ac_fixed, ANCHOR_OPS_HP)[0],
        "ttam": apops_on(bank, prefix, q_amgtp, ac_amgtp, ANCHOR_OPS_HP)[0],
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
            "ops": str(sc["ops"]), "apops_fixed": str(sc["apops_fixed"]),
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
            bank, ctx = bank_for(seed)
            pass1[str(seed)] = {str(k): v for k, v in
                                pass1_seed(bank, ctx, outer, warmup, seed,
                                           n_workers=args.n_workers).items()}
            del bank, ctx
        p1_path.write_text(json.dumps(pass1, indent=2, default=float))
    modules = pick_modules(pass1, outer)
    (out / "selected_modules.json").write_text(json.dumps(modules, indent=2, default=float))
    if args.only_pass == 1:
        print(f"pass 1 done ({time.time() - t0:.0f}s)", flush=True)
        return

    # ---- pass 2 --------------------------------------------------------
    p2_path = out / "pass2_calib_inner.json"
    if p2_path.exists():
        pass2 = json.loads(p2_path.read_text())
    else:
        pass2 = {}
        for seed in SEEDS:
            bank, ctx = bank_for(seed)
            pass2[str(seed)] = {str(k): v for k, v in
                                pass2_seed(bank, ctx, outer, warmup, modules, block_sec, seed,
                                           n_workers=args.n_workers).items()}
            del bank, ctx
        p2_path.write_text(json.dumps(pass2, indent=2, default=float))
    selected = pick_calibration(pass2, modules, outer)
    (out / "selected_configs.json").write_text(json.dumps(
        {"source": args.source, "code_commit": git_commit(), "outer_days": outer,
         "block_sec": block_sec, "delay_sec": DELAY_SEC, "eta_m": APOPS_ETA_M,
         "s_l_half_lives_h": list(S_L_HALF_LIVES), "per_origin": selected}, indent=2, default=float))
    if args.only_pass == 2:
        print(f"pass 2 done ({time.time() - t0:.0f}s)", flush=True)
        return

    # ---- pass 3 -------------------------------------------------------
    pred_dir = out / "final_predictions"
    pred_dir.mkdir(exist_ok=True)
    manifest_rows = []
    per_seed_day_rows = {s: [] for s in SEEDS}
    for seed in SEEDS:
        bank, ctx = bank_for(seed)
        todo = [d for d in outer if d in bank and str(d) in selected]
        results = Parallel(n_jobs=args.n_workers, prefer="processes")(
            delayed(_pass3_origin)(bank, ctx, d, warmup, selected[str(d)], block_sec, seed,
                                   pred_dir, seed == SEEDS[0])
            for d in todo)
        for d, day_rows, mrow in results:
            for r in day_rows:
                per_seed_day_rows[seed].append({"day": int(d), **r})
            if mrow is not None:
                manifest_rows.append(mrow)
        del bank, ctx
    manifest_rows.sort(key=lambda r: r["origin"])

    for seed in SEEDS:
        sd = out / f"seed{seed}"
        sd.mkdir(exist_ok=True)
        pd.DataFrame(per_seed_day_rows[seed]).to_csv(sd / "per_day_metrics.csv", index=False)
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
                  "eta_m_fixed": APOPS_ETA_M},
    }, indent=2, default=float))
    print(f"\nnested TTAM run done in {time.time() - t0:.0f}s -> {out}/", flush=True)


if __name__ == "__main__":
    main()
