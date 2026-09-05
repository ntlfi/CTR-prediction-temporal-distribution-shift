"""Day-level rolling-origin evaluation runner (rolling-protocol sections
2-3, 8). One dataset per invocation. Every choice this script hard-codes
(outer days, sketch dim, block/delay, calibrator config) is frozen in
``withinday_experiments/ROLLING_PROTOCOL_FREEZE.md`` -- written and
committed before this script was ever run.

Produces, under ``--out``:
  rolling_origin_manifest.csv   outer day, train range, inner-val days,
                                 chosen hyperparameters, code/config hash,
                                 data-sampling seed, model seed
  inner_grid_search.csv         every inner-CV (config, fold) score, for audit
  daily_metrics.csv             one row per day x method x seed
  aggregate_results.csv         day-level statistics vs each baseline
  plots/*.png                   the 4 required diagnostic plots
  summary.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from twoscale.calib import CalibConfig
from twoscale.data import load
from twoscale.longterm import adaptive_weights, build_bank, long_term_predictions
from twoscale.methods import build_suite
from twoscale.metrics import brier, day_logloss
from twoscale_run import DATA_PATHS

from withinday.blocks import summary_dim, token_dim
from withinday.cache import build_cache
from withinday.daystats import day_summary, impression_weighted_effect
from withinday.rolling import rolling_origin_v5

# frozen (ROLLING_PROTOCOL_FREEZE.md) -- do not change after seeing results
CALIB_CONFIG = dict(B=1.0, eta0=0.1, eta_schedule="inv_sqrt", update="block",
                    block_sec=1800, delay_sec=1800, eps=1e-5, init_b=0.0, carryover_rho=0.0)
FROZEN_M = 32
FROZEN_BLOCK_SEC = 900
FROZEN_DELAY_SEC = 1800
OUTER_DAYS = {"criteo": list(range(16, 22)), "avazu": list(range(5, 10))}

METHOD_P = {
    "long_only": lambda r: r.p_long_only,
    "online_platt": lambda r: r.p_online_platt,
    "v5_linear": lambda r: r.p_v5,
    "v5_no_history": lambda r: r.p_v5_no_history,
    "v5_shuffled_history": lambda r: r.p_v5_shuffled_history,
}


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=Path(__file__).resolve().parent).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["criteo", "avazu"], required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--sample-frac", type=float, default=None)
    ap.add_argument("--data-seed", type=int, default=0, help="data-sampling seed (frozen at 0)")
    ap.add_argument("--model-seed", type=int, default=0, help="V5 torch init seed")
    ap.add_argument("--n-features", type=int, default=2 ** 18)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sample_frac = args.sample_frac if args.sample_frac is not None else (1.0 if args.source == "criteo" else 0.2)
    out = Path(args.out)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    path = args.data or DATA_PATHS[args.source]
    print(f"loading {args.source} sample_frac={sample_frac} data_seed={args.data_seed} ...", flush=True)
    ds = load(args.source, path, n_features=args.n_features, sample_frac=sample_frac, seed=args.data_seed)
    print(f"  {len(ds.y):,} rows, {ds.n_days} days", flush=True)

    outer_days = OUTER_DAYS[args.source]
    eval_days = list(range(1, max(outer_days) + 1))   # day 0 never usable (no prior history)

    print(f"fitting long-term bank over days {eval_days[0]}-{eval_days[-1]} ...", flush=True)
    bank = build_bank(ds, eval_days, seed=args.data_seed, n_jobs=args.n_jobs)
    eval_days = sorted(bank)
    w = adaptive_weights(bank, eval_days)
    q_by_day = long_term_predictions(bank, eval_days, "adaptive", weights=w)

    calib_cfg = CalibConfig(**CALIB_CONFIG)
    methods, _, _ = build_suite(bank, eval_days, calib_cfg, include_platt=True)

    print(f"building causal cache (m={FROZEN_M}) ...", flush=True)
    cache = build_cache(ds, bank, q_by_day, eval_days, block_sec=FROZEN_BLOCK_SEC,
                        delay_sec=FROZEN_DELAY_SEC, m=FROZEN_M, seed=args.data_seed)

    a_dim, tok_dim, summ_dim = FROZEN_M + 2, token_dim(FROZEN_M), summary_dim(FROZEN_M)
    print(f"rolling-origin V5 over outer days {outer_days} ...", flush=True)
    results, inner_rows = rolling_origin_v5(cache, outer_days, methods["long_only"], methods["online_platt"],
                                            a_dim, tok_dim, summ_dim, FROZEN_M, seed=args.model_seed)
    code_hash = git_commit_hash()
    print(f"  done in {time.time() - t0:.1f}s", flush=True)

    # ---- rolling_origin_manifest.csv + inner_grid_search.csv -----------
    manifest_rows = [{
        "dataset": args.source, "outer_test_day": r.day,
        "train_day_range": f"{r.train_days[0]}-{r.train_days[-1]}",
        "n_train_days": len(r.train_days),
        "inner_validation_days": ";".join(map(str, r.inner_val_days)),
        "chosen_cross_dim": r.chosen_overlay.get("cross_dim"),
        "chosen_lr": r.chosen_overlay.get("lr"),
        "chosen_weight_decay": r.chosen_overlay.get("weight_decay"),
        "code_commit": code_hash,
        "data_sampling_seed": args.data_seed,
        "model_seed": args.model_seed,
    } for r in results]
    pd.DataFrame(manifest_rows).to_csv(out / "rolling_origin_manifest.csv", index=False)
    pd.DataFrame([{"dataset": args.source, **row} for row in inner_rows]).to_csv(
        out / "inner_grid_search.csv", index=False)

    # ---- daily_metrics.csv ----------------------------------------------
    daily_rows = []
    for r in results:
        ll = {name: day_logloss(r.y, fn(r)) for name, fn in METHOD_P.items()}
        for name, fn in METHOD_P.items():
            p = fn(r)
            daily_rows.append({
                "dataset": args.source, "day": r.day, "data_sampling_seed": args.data_seed,
                "model_seed": args.model_seed, "method": name,
                "log_loss": ll[name], "brier": brier(r.y, p), "n_impressions": r.n_impressions,
                "empirical_ctr": r.empirical_ctr,
                "delta_vs_long_only": ll[name] - ll["long_only"],
                "delta_vs_online_platt": ll[name] - ll["online_platt"],
            })
    pd.DataFrame(daily_rows).to_csv(out / "daily_metrics.csv", index=False)

    # ---- aggregate_results.csv -------------------------------------------
    n_imp = np.array([r.n_impressions for r in results])
    ll_v5 = np.array([r.ll_v5 for r in results])
    ll_long = np.array([r.ll_long_only for r in results])
    ll_ops = np.array([r.ll_online_platt for r in results])
    ll_nohist = np.array([r.ll_v5_no_history for r in results])
    ll_shuf = np.array([r.ll_v5_shuffled_history for r in results])

    agg_rows = []
    for name, base_ll in (("vs_long_only", ll_long), ("vs_online_platt", ll_ops)):
        deltas = ll_v5 - base_ll
        s = day_summary(deltas, seed=args.model_seed)
        agg_rows.append({
            "dataset": args.source, "comparison": name,
            "impression_weighted_effect": impression_weighted_effect(n_imp, ll_v5, base_ll),
            "equal_day_weighted_effect": s["mean_delta"], "median_delta": s["median_delta"],
            "n_days": s["n_days"], "n_days_won": s["n_days_won"], "frac_days_won": s["frac_days_won"],
            "ci95_lo": s["ci95_lo"], "ci95_hi": s["ci95_hi"], "sign_test_p": s["sign_test_p"],
            "worst_day_delta": s["worst_day_delta"],
            "loo_mean_min": s["loo_mean_min"], "loo_mean_max": s["loo_mean_max"],
            "loo_reverses_sign": s["loo_reverses_sign"],
            "moving_block_bootstrap": json.dumps(s["moving_block_bootstrap"]),
        })
    agg_rows.append({"dataset": args.source, "comparison": "frac_days_beats_both",
                     "equal_day_weighted_effect": float(np.mean((ll_v5 < ll_long) & (ll_v5 < ll_ops)))})
    agg_rows.append({"dataset": args.source, "comparison": "frac_days_beats_no_history_control",
                     "equal_day_weighted_effect": float(np.mean(ll_v5 < ll_nohist))})
    agg_rows.append({"dataset": args.source, "comparison": "frac_days_beats_shuffled_history_control",
                     "equal_day_weighted_effect": float(np.mean(ll_v5 < ll_shuf))})
    pd.DataFrame(agg_rows).to_csv(out / "aggregate_results.csv", index=False)

    # ---- plots ------------------------------------------------------------
    days = [r.day for r in results]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="gray", lw=1)
    ax.plot(days, ll_v5 - ll_long, "o-", label="V5 - long_only")
    ax.plot(days, ll_v5 - ll_ops, "s-", label="V5 - online_platt")
    ax.set_xlabel("calendar day"); ax.set_ylabel("daily log-loss difference"); ax.legend()
    ax.set_title(f"{args.source}: daily V5 vs baseline (negative favors V5)")
    fig.tight_layout(); fig.savefig(out / "plots" / "daily_diffs.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(days, np.cumsum(ll_v5 - ll_long), "o-", label="cum. V5 - long_only")
    ax.plot(days, np.cumsum(ll_v5 - ll_ops), "s-", label="cum. V5 - online_platt")
    ax.axhline(0, color="gray", lw=1)
    ax.set_xlabel("calendar day"); ax.set_ylabel("cumulative excess log loss"); ax.legend()
    ax.set_title(f"{args.source}: cumulative excess log loss over time")
    fig.tight_layout(); fig.savefig(out / "plots" / "cumulative_excess.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(days, [r.chosen_overlay.get("cross_dim") for r in results], "o-")
    ax.set_xlabel("outer test day"); ax.set_ylabel("selected cross_dim")
    ax.set_yticks([16, 32, 64])
    ax.set_title(f"{args.source}: selected V5 hyperparameter by outer day")
    fig.tight_layout(); fig.savefig(out / "plots" / "selected_hyperparams.png", dpi=120); plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(days, n_imp, alpha=0.5, color="steelblue")
    ax1.set_xlabel("calendar day"); ax1.set_ylabel("n impressions", color="steelblue")
    ax2 = ax1.twinx()
    ax2.plot(days, [r.empirical_ctr for r in results], "o-", color="firebrick")
    ax2.set_ylabel("empirical CTR", color="firebrick")
    ax1.set_title(f"{args.source}: daily row counts and CTR")
    fig.tight_layout(); fig.savefig(out / "plots" / "daily_rows_ctr.png", dpi=120); plt.close(fig)

    summary = {"dataset": args.source, "outer_days": outer_days, "code_commit": code_hash,
              "data_sampling_seed": args.data_seed, "model_seed": args.model_seed,
              "runtime_s": time.time() - t0}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n" + pd.DataFrame(agg_rows).to_string(index=False), flush=True)
    print(f"\nruntime {time.time() - t0:.1f}s -> {out}/", flush=True)


if __name__ == "__main__":
    main()
