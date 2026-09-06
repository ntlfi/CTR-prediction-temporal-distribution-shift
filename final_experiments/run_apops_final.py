"""Plan section 3, "Fixed test" + "Golden check": evaluate the four AP-OPS
rows once on the existing locked stream (Criteo test days 22-30, Avazu
7-9), 3 seeds, all hyper-parameters frozen in ``apops_selected.json``.

Primary comparison: AP-OPS vs current OPS (paired calendar-day log-loss
difference).  Also emits the shift-sensitive temporal metrics the plan
asks for (pre-feedback loss, first-quarter-of-day loss, worst-day loss)
and the AP-OPS mechanism trace (expert-weight paths, memory-scale
dominance), and applies the pre-declared decision rule (section 4).

Golden check (hard-fails the run): the ``current_ops`` row must reproduce
this dataset's saved headline OPS number within tolerance, and an
anchor-only AP-OPS wrapper must be prediction-equivalent to it.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from twoscale.data import load
from twoscale.longterm import build_bank
from twoscale.metrics import (day_logloss, impression_weighted_logloss,
                              paired_day_diffs, per_day_frame, unweighted_daily_logloss)
from twoscale.splits import make_split
from twoscale_run import DATA_PATHS

from methods import adaptive_q_by_day
from apops.method import APOPSConfig, anchor_only_ap_ops, build_rows
from withinday.daystats import day_summary

SEEDS = (0, 1, 2)
ROW_ORDER = ["current_ops", "ap_ops", "single_memory", "no_slope"]
ROW_LABEL = {"current_ops": "Current OPS", "ap_ops": "AP-OPS",
             "single_memory": "Single-memory ablation", "no_slope": "No-slope ablation"}
SECONDS_PER_DAY = 86_400


def cfg_from_selected(sel) -> APOPSConfig:
    return APOPSConfig(block_sec=sel["block_sec"], delay_sec=sel["delay_sec"],
                       ons_lam=sel["ons_lam"], ons_eta=sel["ons_eta"], eta_m=sel["eta_m"],
                       switch_half_life_h=sel["switch_half_life_h"],
                       a_bounds=tuple(sel["a_bounds"]), b_bounds=tuple(sel["b_bounds"]),
                       single_memory_pick=sel["single_memory_pick"])


def temporal_metrics(records, delay_sec):
    """Pre-feedback / first-quarter / worst-day pooled log loss."""
    pre_n = pre_s = q1_n = q1_s = 0.0
    worst = -np.inf
    for r in records:
        y = np.asarray(r["y"], float)
        p = np.clip(np.asarray(r["p"], float), 1e-12, 1 - 1e-12)
        ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        sec = np.asarray(r["sec_in_day"], float)
        pm = sec < delay_sec
        qm = sec < SECONDS_PER_DAY / 4
        pre_s += ll[pm].sum(); pre_n += pm.sum()
        q1_s += ll[qm].sum(); q1_n += qm.sum()
        worst = max(worst, day_logloss(y, p))
    return {"pre_feedback_ll": pre_s / pre_n if pre_n else float("nan"),
            "first_quarter_ll": q1_s / q1_n if q1_n else float("nan"),
            "worst_day_ll": float(worst)}


def run_seed(source, data_path, sample_frac, n_features, warmup, n_jobs, seed, sel, out_dir):
    ds = load(source, data_path, n_features=n_features, sample_frac=sample_frac, seed=seed)
    split = make_split(ds.n_days, warmup=warmup)
    test_days = set(int(d) for d in split.test_days)
    bank = build_bank(ds, split.eval_days, seed=seed, n_jobs=n_jobs)
    eval_days = sorted(bank)
    print(f"  seed {seed}: {len(ds.y):,} rows, test days {sorted(test_days)}", flush=True)

    mix = sel["shared_mixture"]
    q_by_day, _ = adaptive_q_by_day(bank, eval_days, eta=mix["eta"], halflife=mix["halflife"])
    cfg = cfg_from_selected(sel)

    rows, info = build_rows(bank, eval_days, q_by_day, cfg, sel["ops"])

    # ---- golden check ---------------------------------------------------
    anchor = anchor_only_ap_ops(bank, eval_days, q_by_day, cfg, sel["ops"])
    def _cat(rs): return np.concatenate([np.asarray(r["p"], float)
                                         for r in sorted(rs, key=lambda x: x["day"])])
    assert np.array_equal(_cat(anchor), _cat(rows["current_ops"])), \
        "golden check FAILED: anchor-only AP-OPS != current OPS"

    per_day_rows, test_records, summ = [], {}, {}
    for name in ROW_ORDER:
        tr = [r for r in rows[name] if r["day"] in test_days]
        test_records[name] = tr
        pdf = pd.DataFrame(per_day_frame(tr))
        summ[name] = {
            "imp_weighted_log_loss": impression_weighted_logloss(tr),
            "daily_mean_log_loss": unweighted_daily_logloss(tr),
            **temporal_metrics(tr, sel["delay_sec"]),
            "n_test_days": int(len(pdf)),
        }
        for row in per_day_frame(tr):
            per_day_rows.append({"method": name, **row})

    seed_out = out_dir / f"seed{seed}"
    seed_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_day_rows).to_csv(seed_out / "per_day_metrics.csv", index=False)
    pd.DataFrame(info["ap_ops"]["weight_path"]).to_csv(seed_out / "apops_weight_path.csv", index=False)
    (seed_out / "mechanism.json").write_text(json.dumps({
        "ap_ops": {k: v for k, v in info["ap_ops"].items() if k != "weight_path"},
        "no_slope": {k: v for k, v in info["no_slope"].items() if k != "weight_path"},
        "single_memory_pick": info["single_memory_pick"],
        "expert_end_state": {n: info["expert_traces"][n][-1] if info["expert_traces"][n] else None
                             for n in ("R", "S", "L")},
    }, indent=2, default=float))
    (seed_out / "summary.json").write_text(json.dumps(
        {"seed": seed, "n_rows": int(len(ds.y)), "test_days": sorted(test_days), "methods": summ},
        indent=2, default=float))
    del ds, bank
    return test_records, summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["criteo", "avazu"], required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--n-features", type=int, default=2 ** 18)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--config", required=True, help="apops_selected.json from run_apops_hpo.py")
    ap.add_argument("--saved-ops-loss", type=float, default=None,
                    help="headline OPS mean_imp_wt_ll for the golden tolerance check")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sel = json.loads(Path(args.config).read_text())
    assert sel["source"] == args.source
    warmup = args.warmup if args.warmup is not None else (4 if args.source == "criteo" else 3)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    all_recs = {m: [] for m in ROW_ORDER}
    per_seed = []
    for seed in SEEDS:
        tr, summ = run_seed(args.source, args.data or DATA_PATHS[args.source], args.sample_frac,
                            args.n_features, warmup, args.n_jobs, seed, sel, out)
        per_seed.append(summ)
        for m in ROW_ORDER:
            all_recs[m].append((seed, tr[m]))

    # ---- aggregate: paired daily deltas vs current OPS -----------------
    headline = []
    for m in ROW_ORDER:
        mean_iwll = float(np.mean([s[m]["imp_weighted_log_loss"] for s in per_seed]))
        std_iwll = float(np.std([s[m]["imp_weighted_log_loss"] for s in per_seed]))
        row = {"method": ROW_LABEL[m], "mean_imp_wt_ll": mean_iwll, "std_across_seeds": std_iwll,
               "mean_pre_feedback_ll": float(np.mean([s[m]["pre_feedback_ll"] for s in per_seed])),
               "mean_first_quarter_ll": float(np.mean([s[m]["first_quarter_ll"] for s in per_seed])),
               "mean_worst_day_ll": float(np.mean([s[m]["worst_day_ll"] for s in per_seed]))}
        if m != "current_ops":
            deltas = []
            for (se, e_recs), (sm, m_recs) in zip(all_recs["current_ops"], all_recs[m]):
                assert se == sm
                _, d = paired_day_diffs(m_recs, e_recs)
                deltas.extend(d.tolist())
            st = day_summary(deltas, seed=0)
            row.update({"mean_delta_vs_ops": st["mean_delta"], "ci95_lo": st["ci95_lo"],
                        "ci95_hi": st["ci95_hi"], "sign_test_p": st["sign_test_p"],
                        "frac_seed_days_won": st["frac_days_won"], "n_seed_days": st["n_days"],
                        "worst_seed_day_delta": st["worst_day_delta"]})
        headline.append(row)

    hd = pd.DataFrame(headline)
    hd.to_csv(out / "apops_headline.csv", index=False)

    # ---- decision rule (section 4) -----------------------------------
    dni = sel["decision_rule"]["delta_NI"]
    ap_row = next(r for r in headline if r["method"] == "AP-OPS")
    non_inferior = ap_row["ci95_hi"] < dni
    improves = ap_row["ci95_hi"] < 0
    ap_iwll = ap_row["mean_imp_wt_ll"]
    ops_iwll = next(r for r in headline if r["method"] == "Current OPS")["mean_imp_wt_ll"]
    point_no_worse = (ap_iwll - ops_iwll) <= dni
    verdict = ("Improves" if improves else
               "Retains performance" if non_inferior else
               "Inconclusive / Failure")
    decision = {
        "delta_NI": dni, "ap_ops_mean_delta_vs_ops": ap_row["mean_delta_vs_ops"],
        "ap_ops_ci95": [ap_row["ci95_lo"], ap_row["ci95_hi"]],
        "point_estimate_no_worse_than_delta_NI": bool(point_no_worse),
        "ci_upper_below_delta_NI": bool(non_inferior),
        "ci_upper_below_zero": bool(improves), "verdict": verdict,
    }
    (out / "decision.json").write_text(json.dumps(decision, indent=2, default=float))

    if args.saved_ops_loss is not None:
        gap = abs(ops_iwll - args.saved_ops_loss)
        ok = gap < 5e-4
        (out / "golden_check.txt").write_text(
            f"current_ops mean_imp_wt_ll = {ops_iwll:.8f}\n"
            f"saved headline OPS          = {args.saved_ops_loss:.8f}\n"
            f"abs gap                     = {gap:.2e}   -> {'PASS' if ok else 'FAIL'} (tol 5e-4)\n")
        assert ok, f"golden check FAILED: current_ops {ops_iwll} vs saved {args.saved_ops_loss}"

    (out / "summary.json").write_text(json.dumps(
        {"source": args.source, "config": str(args.config), "seeds": list(SEEDS),
         "per_seed": per_seed, "decision": decision, "runtime_s": time.time() - t0},
        indent=2, default=float))

    print("\n=== AP-OPS fixed-test headline ===", flush=True)
    print(hd.to_string(index=False), flush=True)
    print(f"\ndecision: {verdict}  (delta_NI={dni:.6g})", flush=True)
    print(f"runtime {time.time() - t0:.1f}s -> {out}/", flush=True)


if __name__ == "__main__":
    main()
