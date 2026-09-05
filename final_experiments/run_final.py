"""Section 12: the primary 3-seed locked-test comparison, all six headline
methods (Expanding, Best Fixed Window, ARW, AdaMoE, OPS, DualTime-CTR),
using the frozen ``selected_configs.json`` produced by ``run_hpo.py`` --
no hyperparameter is chosen or re-selected here.

For each seed the bank is fit over ``dev_days + test_days`` (not test_days
alone) so the day-by-day online methods (ARW's tournament, AdaMoE's weight
trace, OPS's/DualTime's calibrator state) carry the same history into the
test period a real deployment would have had -- test-day *metrics* are
still computed only over ``test_days``, matching ``twoscale_run.py``'s
existing eval_days-vs-test_days split.

Writes, per seed, ``final_experiments/<source>/final/seed<seed>/`` with
per_day_metrics.csv + comparisons + summary.json, then aggregates the 3
seeds into ``final_experiments/<source>/final/headline_results.csv`` (one
row per method: mean/day-level stats over the pooled (seed, test day)
log-loss deltas vs Expanding, via ``withinday.daystats.day_summary`` --
each (seed, day) pair is treated as one exchangeable replicate, which is
the same statistical unit ``run_hpo.py``'s "mean across seeds" selection
rule implicitly assumes; the original spec's own section 12/14 pooling
rule was not available when this was written, so treat the CI machinery
here as "the repo's standard day-level treatment applied straightforwardly
to 3 seeds," not a verbatim transcription of the spec -- flag to the user
if the two need reconciling against the source PDF/conversation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from twoscale.calib import CalibConfig
from twoscale.data import load
from twoscale.longterm import build_bank
from twoscale.metrics import (bootstrap_paired_ci, day_logloss, days_won,
                              impression_weighted_logloss, paired_day_diffs,
                              per_day_frame, unweighted_daily_logloss)
from twoscale.splits import make_split
from twoscale_run import DATA_PATHS

from dualtime.online import DualTimeConfig
from methods import adamoe_method, adaptive_q_by_day, arw_method, dualtime_method, expanding_method, ops_method
from withinday.daystats import day_summary

SEEDS = (0, 1, 2)
METHOD_ORDER = ["expanding", "best_fixed", "arw", "adamoe", "ops", "dualtime"]
METHOD_LABEL = {"expanding": "Expanding", "best_fixed": "Best Fixed Window", "arw": "ARW",
                "adamoe": "AdaMoE", "ops": "OPS", "dualtime": "DualTime-CTR"}


def build_headline_methods(ds, bank, eval_days, selected, seed):
    """One record list per headline method, built ONLY from frozen
    ``selected`` -- no per-seed re-tuning of any hyperparameter."""
    methods = {}
    methods["expanding"] = expanding_method(bank, eval_days)

    h_star = selected["best_fixed_window"]["h_star"]
    days = sorted(d for d in eval_days if d in bank)
    methods["best_fixed"] = [{"day": d, "y": bank[d].y, "p": bank[d].preds[h_star],
                              "sec_in_day": bank[d].sec_in_day} for d in days]

    arw_cfg = selected["arw"]
    recs, _ = arw_method(bank, eval_days, delta=arw_cfg["delta"], min_history=arw_cfg["min_history"])
    methods["arw"] = recs

    recs, _ = adamoe_method(bank, eval_days, lam=selected["adamoe"]["lambda"])
    methods["adamoe"] = recs

    mix = selected["shared_mixture"]
    q_by_day, _ = adaptive_q_by_day(bank, eval_days, eta=mix["eta"], halflife=mix["halflife"])

    ops_cfg = selected["ops"]
    cfg = CalibConfig(B=ops_cfg["B"], eta0=ops_cfg["eta0"], eta_schedule=ops_cfg["schedule"],
                      update="block", block_sec=selected["block_sec"], delay_sec=selected["delay_sec"],
                      platt=True)
    recs, _ = ops_method(bank, eval_days, q_by_day, cfg)
    methods["ops"] = recs

    dt_cfg_d = selected["dualtime"]
    dt_cfg = DualTimeConfig(block_sec=selected["block_sec"], delay_sec=selected["delay_sec"],
                            m=dt_cfg_d["m"], cross_dim=dt_cfg_d["cross_dim"], B_w=dt_cfg_d["B_w"])
    methods["dualtime"] = dualtime_method(ds, bank, eval_days, q_by_day, dt_cfg,
                                          sketch_seed=seed, hash_seed=seed)
    return methods


def summarize_method(records, test_days):
    tr = [r for r in records if r["day"] in test_days]
    pdf = pd.DataFrame(per_day_frame(tr))
    return {
        "imp_weighted_log_loss": impression_weighted_logloss(tr),
        "daily_mean_log_loss": unweighted_daily_logloss(tr),
        "worst_day_log_loss": float(pdf["log_loss"].max()) if len(pdf) else float("nan"),
        "n_test_days": int(len(pdf)),
    }, tr


def run_seed(source, data_path, sample_frac, n_features, warmup, n_jobs, seed, selected, out_dir):
    ds = load(source, data_path, n_features=n_features, sample_frac=sample_frac, seed=seed)
    split = make_split(ds.n_days, warmup=warmup)
    test_days = set(int(d) for d in split.test_days)
    print(f"  seed {seed}: {len(ds.y):,} rows, test days {sorted(test_days)}", flush=True)

    bank = build_bank(ds, split.eval_days, seed=seed, n_jobs=n_jobs)
    eval_days = sorted(bank)
    methods = build_headline_methods(ds, bank, eval_days, selected, seed)

    per_day_rows, test_records, method_summ = [], {}, {}
    for name, recs in methods.items():
        summ, tr = summarize_method(recs, test_days)
        method_summ[name] = summ
        test_records[name] = tr
        for r in per_day_frame(tr):
            per_day_rows.append({"method": name, **r})

    seed_out = out_dir / f"seed{seed}"
    seed_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_day_rows).to_csv(seed_out / "per_day_metrics.csv", index=False)
    (seed_out / "summary.json").write_text(json.dumps({
        "seed": seed, "n_rows": int(len(ds.y)), "n_days": ds.n_days,
        "test_days": sorted(test_days), "methods": method_summ,
    }, indent=2, default=float))
    del ds, bank
    return test_records, method_summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["criteo", "avazu"], required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--sample-frac", type=float, default=None)
    ap.add_argument("--n-features", type=int, default=2 ** 18)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--config", required=True, help="selected_configs.json from run_hpo.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    selected = json.loads(Path(args.config).read_text())
    assert selected["source"] == args.source, "selected_configs.json is for a different --source"

    sample_frac = args.sample_frac if args.sample_frac is not None else 1.0
    warmup = args.warmup if args.warmup is not None else (4 if args.source == "criteo" else 3)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    all_test_records = {m: [] for m in METHOD_ORDER}   # per seed, appended as (seed, records)
    per_seed_summ = []
    for seed in SEEDS:
        test_records, method_summ = run_seed(
            args.source, args.data or DATA_PATHS[args.source], sample_frac,
            args.n_features, warmup, args.n_jobs, seed, selected, out)
        per_seed_summ.append(method_summ)
        for m in METHOD_ORDER:
            all_test_records[m].append((seed, test_records[m]))

    # ---- aggregate across seeds: pool (seed, day) as the replicate unit ---
    headline_rows = []
    for m in METHOD_ORDER:
        mean_iwll = float(np.mean([s[m]["imp_weighted_log_loss"] for s in per_seed_summ]))
        std_iwll = float(np.std([s[m]["imp_weighted_log_loss"] for s in per_seed_summ]))
        if m == "expanding":
            headline_rows.append({"method": METHOD_LABEL[m], "mean_imp_wt_ll": mean_iwll,
                                  "std_across_seeds": std_iwll, "n_seeds": len(SEEDS)})
            continue
        deltas = []
        for (seed_e, exp_recs), (seed_m, m_recs) in zip(all_test_records["expanding"], all_test_records[m]):
            assert seed_e == seed_m
            _, d = paired_day_diffs(m_recs, exp_recs)
            deltas.extend(d.tolist())
        stats = day_summary(deltas, seed=0)
        headline_rows.append({"method": METHOD_LABEL[m], "mean_imp_wt_ll": mean_iwll,
                              "std_across_seeds": std_iwll, "n_seeds": len(SEEDS),
                              "mean_delta_vs_expanding": stats["mean_delta"],
                              "ci95_lo": stats["ci95_lo"], "ci95_hi": stats["ci95_hi"],
                              "significant_below_zero": stats["ci95_hi"] < 0,
                              "frac_seed_day_won": stats["frac_days_won"],
                              "n_seed_days": stats["n_days"]})

    headline = pd.DataFrame(headline_rows).sort_values("mean_imp_wt_ll")
    headline.to_csv(out / "headline_results.csv", index=False)
    headline.to_csv(out / "headline_results.tex", sep="&", index=False, lineterminator=" \\\\\n")

    (out / "summary.json").write_text(json.dumps({
        "source": args.source, "config_used": str(args.config), "seeds": list(SEEDS),
        "per_seed": per_seed_summ, "runtime_s": time.time() - t0,
    }, indent=2, default=float))

    print("\n=== headline results (3-seed locked test) ===", flush=True)
    print(headline.to_string(index=False), flush=True)
    print(f"\nruntime {time.time() - t0:.1f}s -> {out}/", flush=True)


if __name__ == "__main__":
    main()
