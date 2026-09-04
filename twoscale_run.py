"""Two-timescale CTR forecasting -- one (dataset, seed) evaluation cell.

Runs the full protocol of ``CTR_Two_Timescale_Experiment_Plan.pdf``:

  1. load data with a within-day time axis (twoscale.data)
  2. scaled chronological split (twoscale.splits)
  3. fit the long-term candidate bank on days < d for every eval day
  4. feasibility diagnostics on the dev days (section 5)
  5. build the baseline / ablation suite (section 4)
  6. metrics, paired CIs, theory-aligned regret / captured gain (sections 7-8)
  7. ablations 1-7 (section 9)

Writes everything to ``--out`` as CSV + summary.json. Hyperparameters come
from ``--config`` (a frozen JSON from twoscale_hpo.py); without it the plan's
default grid centre is used.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from twoscale.calib import CalibConfig
from twoscale.data import load
from twoscale.diagnostics import (daily_oracle_improvement, early_late_gain,
                                  intraday_residual_structure)
from twoscale.longterm import build_bank
from twoscale.methods import build_suite
from twoscale.metrics import (bootstrap_paired_ci, days_won, early_mid_late,
                              impression_weighted_logloss, intraday_block_frame,
                              paired_day_diffs, per_day_frame,
                              regret_and_captured_gain, unweighted_daily_logloss)
from twoscale.splits import make_split

DEFAULT_CONFIG = dict(B=1.0, eta0=0.1, eta_schedule="inv_sqrt", update="block",
                      block_sec=1800, delay_sec=1800, eps=1e-5,
                      init_b=0.0, carryover_rho=0.0)
DATA_PATHS = {
    "criteo": "/insomnia001/home/tn2447/data/criteo/criteo_attribution_dataset.tsv.gz",
    "avazu": "/insomnia001/home/tn2447/data/avazu/Avazu_x4.zip",
}


def summarize_method(name, records, test_days):
    tr = [r for r in records if r["day"] in test_days]
    pdf = pd.DataFrame(per_day_frame(tr))
    return {
        "imp_weighted_log_loss": impression_weighted_logloss(tr),
        "daily_mean_log_loss": unweighted_daily_logloss(tr),
        "median_day_log_loss": float(pdf["log_loss"].median()) if len(pdf) else float("nan"),
        "worst_day_log_loss": float(pdf["log_loss"].max()) if len(pdf) else float("nan"),
        "brier": float(np.average(pdf["brier"], weights=pdf["n"])) if len(pdf) else float("nan"),
        "ece": float(np.average(pdf["ece"], weights=pdf["n"])) if len(pdf) else float("nan"),
        "n_test_days": int(len(pdf)),
        "early_mid_late": early_mid_late(tr),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["criteo", "avazu"], default="criteo")
    ap.add_argument("--data", default=None)
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-features", type=int, default=2 ** 18)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--config", default=None, help="frozen hyperparameter JSON")
    ap.add_argument("--mixture-eta", type=float, default=60.0)
    ap.add_argument("--mixture-halflife", type=float, default=5.0)
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    cfg_d = dict(DEFAULT_CONFIG)
    if args.config:
        cfg_d.update(json.loads(Path(args.config).read_text()))
    cfg = CalibConfig(**{k: v for k, v in cfg_d.items() if k in CalibConfig.__annotations__})
    mixture_eta = cfg_d.get("mixture_eta", args.mixture_eta)
    mixture_halflife = cfg_d.get("mixture_halflife", args.mixture_halflife)
    args.mixture_eta, args.mixture_halflife = mixture_eta, mixture_halflife

    path = args.data or DATA_PATHS[args.source]
    print(f"loading {args.source} from {path} (sample_frac={args.sample_frac}) ...", flush=True)
    ds = load(args.source, path, n_features=args.n_features,
              sample_frac=args.sample_frac, seed=args.seed)
    print(f"  {len(ds.y):,} rows, {ds.n_days} days, click rate {ds.y.mean():.4f}", flush=True)

    split = make_split(ds.n_days, warmup=args.warmup)
    test_days = set(int(d) for d in split.test_days)
    dev_days = set(int(d) for d in split.dev_days)
    print(f"  split: train {list(map(int, split.train_days))}  dev {list(map(int, split.dev_days))}  "
          f"test {list(map(int, split.test_days))}", flush=True)

    print("fitting long-term candidate bank ...", flush=True)
    bank = build_bank(ds, split.eval_days, seed=args.seed, n_jobs=args.n_jobs)
    eval_days = sorted(bank)

    t_suite = time.time()
    methods, q_sources, traces = build_suite(
        bank, eval_days, cfg, mixture_eta=args.mixture_eta,
        mixture_halflife=args.mixture_halflife)
    suite_time = time.time() - t_suite

    # ---- section 7.4: within-day calibration cost, isolated ------------
    from twoscale.methods import _calibrated
    n_imp = sum(len(bank[d].y) for d in eval_days)
    t_cal = time.time()
    _calibrated(q_sources["combined"], bank, eval_days, cfg)
    calib_sec_per_million = (time.time() - t_cal) / max(n_imp, 1) * 1e6

    # ---- feasibility diagnostics (dev days, section 5) -------------------
    q_adaptive = q_sources["long_only"]
    diag_oracle = daily_oracle_improvement(q_adaptive, bank, dev_days, B=cfg.B, eps=cfg.eps)
    resid_per_day, resid_runs = intraday_residual_structure(q_adaptive, bank, dev_days, cfg.block_sec)
    el_gain = early_late_gain(q_adaptive, methods["combined"], bank, dev_days)
    pd.DataFrame(diag_oracle).to_csv(out / "diag_daily_oracle.csv", index=False)
    pd.DataFrame(resid_per_day).to_csv(out / "diag_intraday_residual.csv", index=False)
    pd.DataFrame(resid_runs).to_csv(out / "diag_residual_runs.csv", index=False)

    # ---- per-day metrics for every method ------------------------------
    per_day_rows = []
    for name, recs in methods.items():
        for r in per_day_frame(recs):
            per_day_rows.append({"method": name, "is_test": r["day"] in test_days, **r})
    pd.DataFrame(per_day_rows).to_csv(out / "per_day_metrics.csv", index=False)

    # ---- intraday block behaviour (test days) --------------------------
    block_rows = []
    for name in ("expanding", "long_only", "short_only", "combined", "time_of_day"):
        tr = [r for r in methods[name] if r["day"] in test_days]
        for br in intraday_block_frame(tr, block_sec=3600):
            block_rows.append({"method": name, **br})
    pd.DataFrame(block_rows).to_csv(out / "intraday_blocks.csv", index=False)

    # ---- traces -------------------------------------------------------
    tr_rows = []
    for name, tl in traces.items():
        for t in tl:
            tr_rows.append({"method": name, "day": t["day"], "b_end": t["b_end"], "a_end": t["a_end"]})
    pd.DataFrame(tr_rows).to_csv(out / "calib_traces.csv", index=False)

    # ---- method summaries + decisive paired comparisons ---------------
    method_summ = {n: summarize_method(n, r, test_days) for n, r in methods.items()}

    def paired(a, b):
        _, deltas = paired_day_diffs([r for r in methods[a] if r["day"] in test_days],
                                     [r for r in methods[b] if r["day"] in test_days])
        m, lo, hi = bootstrap_paired_ci(deltas, seed=args.seed)
        frac, won, tot = days_won([r for r in methods[a] if r["day"] in test_days],
                                  [r for r in methods[b] if r["day"] in test_days])
        return {"mean_delta": m, "ci95": [lo, hi], "days_won": won, "days_total": tot,
                "days_won_frac": frac, "significant_below_zero": hi < 0}

    comparisons = {
        "combined_vs_long_only": paired("combined", "long_only"),
        "combined_vs_short_only": paired("combined", "short_only"),
        "combined_vs_expanding": paired("combined", "expanding"),
        "combined_vs_time_of_day": paired("combined", "time_of_day"),
        "short_only_vs_expanding": paired("short_only", "expanding"),
        "long_only_vs_expanding": paired("long_only", "expanding"),
        "online_platt_vs_combined": paired("online_platt", "combined"),
    }

    # ---- theory-aligned: regret + captured gain (section 8) ------------
    rcg = {}
    for name in ("short_only", "combined", "online_platt", "time_of_day"):
        rows = regret_and_captured_gain(
            [r for r in methods[name] if r["day"] in test_days],
            [r for r in methods["long_only" if name != "short_only" else "expanding"]
             if r["day"] in test_days],
            q_sources[name], B=cfg.B, eps=cfg.eps)
        pd.DataFrame(rows).to_csv(out / f"regret_{name}.csv", index=False)
        valid = [r for r in rows if not r["denom_negligible"]]
        rcg[name] = {
            "mean_regret": float(np.mean([r["regret"] for r in rows])) if rows else float("nan"),
            "mean_captured_gain": float(np.nanmean([r["captured_gain"] for r in valid])) if valid else float("nan"),
            "n_days_captured_gain": len(valid),
        }

    # ---- ablations 1-7 (section 9) -----------------------------------
    ablations = {}
    if not args.no_ablations:
        ablations = run_ablations(bank, eval_days, cfg, test_days, args)
        pd.DataFrame(ablations["table"]).to_csv(out / "ablations.csv", index=False)

    summary = {
        "config": {"source": args.source, "seed": args.seed, "sample_frac": args.sample_frac,
                   "n_rows": int(len(ds.y)), "n_days": ds.n_days,
                   "train_days": list(map(int, split.train_days)),
                   "dev_days": list(map(int, split.dev_days)),
                   "test_days": list(map(int, split.test_days)),
                   "calib": asdict(cfg),
                   "mixture_eta": args.mixture_eta, "mixture_halflife": args.mixture_halflife},
        "feasibility": {
            "dev_mean_oracle_improvement": float(np.mean([r["improvement"] for r in diag_oracle])) if diag_oracle else None,
            "dev_mean_rel_oracle_improvement": float(np.mean([r["rel_improvement"] for r in diag_oracle])) if diag_oracle else None,
            "dev_mean_longest_same_sign_run": float(np.mean([r["longest_same_sign_run"] for r in resid_runs])) if resid_runs else None,
            "dev_mean_residual_autocorr_lag1": float(np.nanmean([r["residual_autocorr_lag1"] for r in resid_runs])) if resid_runs else None,
            "early_late_gain": el_gain,
        },
        "methods": method_summ,
        "comparisons": comparisons,
        "regret_captured_gain": rcg,
        "ablations": ablations.get("summary", {}),
        "success_criteria": success_criteria(comparisons, method_summ, rcg),
        "cost": {"bank_fit_s": float(sum(bank[d].fit_time for d in eval_days)),
                 "suite_build_s": suite_time,
                 "calib_sec_per_million_impressions": calib_sec_per_million,
                 "n_impressions": int(n_imp)},
        "runtime_s": time.time() - t_start,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    tbl = pd.DataFrame([
        {"method": n, "imp_wt_ll": s["imp_weighted_log_loss"], "daily_ll": s["daily_mean_log_loss"],
         "worst_day": s["worst_day_log_loss"], "brier": s["brier"], "ece": s["ece"]}
        for n, s in method_summ.items()]).sort_values("imp_wt_ll")
    tbl.to_csv(out / "comparison_table.csv", index=False)
    print("\n=== locked-test comparison ===", flush=True)
    print(tbl.to_string(index=False), flush=True)
    print("\ncombined vs long_only :", comparisons["combined_vs_long_only"], flush=True)
    print("combined vs short_only:", comparisons["combined_vs_short_only"], flush=True)
    print(f"\nruntime {time.time() - t_start:.1f}s -> {out}/", flush=True)


def run_ablations(bank, eval_days, cfg, test_days, args):
    """Plan section 9: reset vs carry-over, feedback delay, update granularity,
    seasonality vs current-day, chronology placebo, long-term redundancy,
    calibration complexity."""
    from twoscale.metrics import impression_weighted_logloss as iwll
    from twoscale.methods import _calibrated, _uncalibrated
    from twoscale.longterm import adaptive_weights, long_term_predictions

    w = adaptive_weights(bank, eval_days, eta=args.mixture_eta, halflife=args.mixture_halflife)
    q_adaptive = long_term_predictions(bank, eval_days, "adaptive", weights=w)
    q_expanding = long_term_predictions(bank, eval_days, "expanding")
    q_equal = long_term_predictions(bank, eval_days, "equal")

    def ll(recs):
        return iwll([r for r in recs if r["day"] in test_days])

    rows = []

    # 1. daily reset vs carry-over
    for rho in (0.0, 0.5, 1.0):
        recs, _ = _calibrated(q_adaptive, bank, eval_days, replace(cfg, carryover_rho=rho))
        rows.append({"ablation": "carryover_rho", "setting": rho, "imp_wt_ll": ll(recs)})

    # 2. feedback delay
    for delay in (0, 900, 1800, 3600):
        recs, _ = _calibrated(q_adaptive, bank, eval_days, replace(cfg, delay_sec=delay))
        rows.append({"ablation": "delay_sec", "setting": delay, "imp_wt_ll": ll(recs)})

    # 3. update granularity. Per-impression eq (5) needs a decaying rate
    # (eta_i) rather than the block-tuned constant one, or a single noisy
    # label sends b bouncing across [-B, B]; use inv_sqrt for that arm.
    for upd, blk in (("impression", None), ("block", 900), ("block", 1800), ("block", 3600)):
        if blk is None:
            c = replace(cfg, update="impression", eta_schedule="inv_sqrt", eta0=0.03)
        else:
            c = replace(cfg, update="block", block_sec=blk)
        recs, _ = _calibrated(q_adaptive, bank, eval_days, c)
        rows.append({"ablation": "update_granularity",
                     "setting": upd if blk is None else f"block_{blk}", "imp_wt_ll": ll(recs)})

    # 5. chronology placebo (shuffle within-day order)
    recs_real, _ = _calibrated(q_adaptive, bank, eval_days, cfg)
    recs_shuf, _ = _calibrated(q_adaptive, bank, eval_days, cfg, shuffle_seed=args.seed + 1)
    rows.append({"ablation": "chronology", "setting": "real", "imp_wt_ll": ll(recs_real)})
    rows.append({"ablation": "chronology", "setting": "shuffled", "imp_wt_ll": ll(recs_shuf)})

    # 6. long-term redundancy (combined backbone = expanding / equal / adaptive)
    for bname, qsrc in (("expanding", q_expanding), ("equal", q_equal), ("adaptive", q_adaptive)):
        recs, _ = _calibrated(qsrc, bank, eval_days, cfg)
        rows.append({"ablation": "long_term_backbone", "setting": bname, "imp_wt_ll": ll(recs)})

    # 7. calibration complexity (intercept vs Platt)
    recs_i, _ = _calibrated(q_adaptive, bank, eval_days, cfg)
    recs_p, _ = _calibrated(q_adaptive, bank, eval_days, replace(cfg, platt=True))
    rows.append({"ablation": "calib_complexity", "setting": "intercept", "imp_wt_ll": ll(recs_i)})
    rows.append({"ablation": "calib_complexity", "setting": "platt", "imp_wt_ll": ll(recs_p)})

    summary = {
        "chronology_placebo_gain_lost": ll(recs_shuf) - ll(recs_real),
        "delay0_vs_delay1800": next(r["imp_wt_ll"] for r in rows if r["ablation"] == "delay_sec" and r["setting"] == 0)
        - next(r["imp_wt_ll"] for r in rows if r["ablation"] == "delay_sec" and r["setting"] == 1800),
    }
    return {"table": rows, "summary": summary}


def success_criteria(comparisons, method_summ, rcg):
    """Plan section 10 decision rules."""
    c = comparisons
    return {
        "1_beats_long_only_ci": c["combined_vs_long_only"]["significant_below_zero"],
        "2_beats_short_only": c["combined_vs_short_only"]["mean_delta"] < 0,
        "3_beats_time_of_day": c["combined_vs_time_of_day"]["mean_delta"] < 0,
        "5_captures_oracle_gain": rcg.get("combined", {}).get("mean_captured_gain", 0) > 0.1,
        "6_not_from_few_days": c["combined_vs_long_only"]["days_won_frac"] >= 0.5,
        "interpretation": interpret(c),
    }


def interpret(c):
    so = c["combined_vs_short_only"]["mean_delta"]
    lo = c["combined_vs_long_only"]["mean_delta"]
    if abs(so) < 1e-4:
        return "short_only ~= combined: adaptive long-term module adds little"
    if abs(lo) < 1e-4:
        return "long_only ~= combined: limited exploitable within-day calibration drift"
    if lo < 0 and so < 0:
        return "combined beats both: two-timescale framework supported"
    return "mixed / only-oracle: improve the online update policy"


if __name__ == "__main__":
    main()
