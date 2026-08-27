"""AMG-TP experimental plan -- Stage 1 runner: one (regime, seed) cell of the
full benchmark of M5b-high-smooth against the baseline suite
(AMG-TP_Academic_LaTeX.pdf sections 5-7).

Runs, for a single data source + seed, the complete method list from the plan:

  Baselines (5.1)     expanding, rolling_1/3/7/14, validation_selected,
                      exponential forgetting (decay_hl1/3/7), Han ARW,
                      Differentiable Forgetting, AdaMoE, uniform-5 average,
                      M1 global mix, M2 context gate.
  M5b smoothness grid m5b_smooth{0,1e-3,1e-2,1e-1,3e-1} -- this is also the
                      ablation ladder A1 (no persistence, smooth=0),
                      A2 (fixed low persistence = M5b-default, 1e-3) and
                      A3 (fixed high persistence = M5b-high-smooth, 1e-1),
                      and supplies the oracle-fixed-smoothness diagnostic.
  Diagnostic ceiling  ensemble3 (M2 / M5b-default / M5b-high-smooth meta-gate).

Then computes every prediction metric (log loss, Brier, PR-AUC, ROC-AUC,
calibration error), the per-day / per-subgroup loss curves, the oracle
diagnostics (best fixed horizon, per-day and per-group oracle horizon,
per-day oracle persistence regime), the temporal-adaptation metrics
(recovery time, peak post-shift excess, cumulative post-shift regret,
stationary downside, gate movement, effective horizon), and day-level
bootstrap CIs, and writes them all to --out.

SFTL is run separately (run_sftl.py, slow neural method) and merged at
aggregation time if present.

Example:
    python amgtp_run.py --source synthetic --synthetic-days 120 \\
        --synthetic-rows-per-day 3000 --synthetic-drift abrupt \\
        --seed 0 --out amgtp_experiments/stage1_m5b_high_smooth/s1_abrupt/seed0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from adamoe import run_adamoe
from amgtp_eval import (adaptation_metrics, best_fixed_horizon, block_bootstrap_ci_df,
                        gate_dynamics, oracle_per_day_frame, weighted_mean_df)
from baselines import WINDOW_FAMILY, build_candidates, fit_predict
from candidate_bank import build_candidate_bank
import diff_forgetting
from data_source import add_data_source_args, load_data_with_context
from ensemble3 import EXPERTS as ENS3_EXPERTS, run_ensemble3
from han_arw import run_han_arw
from m1_global_mix import run_m1
from m2_context_gate import run_m2
from m5_multiscale_gate import run_m5
from metrics import day_metrics
from splits import compute_splits

M5B_SMOOTH_GRID = [0.0, 1e-3, 1e-2, 1e-1, 3e-1]
M5B_DEFAULT_SMOOTH = 1e-3
M5B_HIGH_SMOOTH = 1e-1
METHOD_UNDER_TEST = "m5b_smooth0.1"

# nominal-horizon fallback expert set for M2 (2-expert short/long gate)
M2_EXPERTS = ["rolling_3", "expanding"]


def smooth_label(s: float) -> str:
    return f"m5b_smooth{s:g}"


def _rows_from_bank(bank, name, days):
    return [{"day": t, "y_true": bank[name][t]["y_true"], "y_pred": bank[name][t]["y_pred"],
             "n_train": bank[name][t]["n_train"], "fit_time": bank[name][t]["fit_time"]}
            for t in days if t in bank[name]]


def _uniform5_rows(bank, days):
    rows = []
    for t in days:
        if not all(t in bank[n] for n in WINDOW_FAMILY):
            continue
        preds = np.stack([bank[n][t]["y_pred"] for n in WINDOW_FAMILY], axis=1)
        rows.append({"day": t, "y_true": bank[WINDOW_FAMILY[0]][t]["y_true"],
                     "y_pred": preds.mean(axis=1),
                     "n_train": int(np.mean([bank[n][t]["n_train"] for n in WINDOW_FAMILY])),
                     "fit_time": float(sum(bank[n][t]["fit_time"] for n in WINDOW_FAMILY))})
    return rows


def _decay_rows(X, y, day, days, candidates, alpha, seed, n_jobs):
    out = {}
    for hl_name in [c for c in candidates if c.startswith("decay_hl")]:
        jobs = Parallel(n_jobs=n_jobs)(
            delayed(fit_predict)(X, y, day, t, candidates[hl_name], alpha=alpha, seed=seed) for t in days)
        out[hl_name] = [{"day": t, "y_true": r["y_true"], "y_pred": r["y_pred"],
                         "n_train": r["n_train"], "fit_time": r["fit_time"]}
                        for t, r in zip(days, jobs) if r is not None]
    return out


def _diff_forgetting_rows(X, y, day, days, alpha, seed, n_jobs, maxiter):
    def _one(t):
        return t, diff_forgetting.fit_predict(X, y, day, t, alpha=alpha, seed=seed, maxiter=maxiter)
    res = Parallel(n_jobs=n_jobs)(delayed(_one)(t) for t in days)
    return [{"day": t, "y_true": r["y_true"], "y_pred": r["y_pred"],
             "n_train": r["n_train"], "fit_time": r["fit_time"]}
            for t, r in res if r is not None]


def _validation_selected_rows(bank, dev_days, test_and_dev_days):
    scores = {}
    for name in WINDOW_FAMILY:
        losses = [(len(bank[name][t]["y_true"]),
                   day_metrics(bank[name][t]["y_true"], bank[name][t]["y_pred"])["log_loss"])
                  for t in dev_days if t in bank[name]]
        if losses:
            w = np.array([l[0] for l in losses], dtype=float)
            v = np.array([l[1] for l in losses])
            scores[name] = float(np.average(v, weights=w))
    h = min(scores, key=scores.get)
    return h, _rows_from_bank(bank, h, test_and_dev_days)


def per_day_metric_rows(name, rows, test_days):
    out = []
    for r in rows:
        m = day_metrics(r["y_true"], r["y_pred"])
        out.append({"method": name, "day": r["day"], "is_test": r["day"] in test_days,
                    **{k: v for k, v in m.items()},
                    "n_train": r.get("n_train", np.nan), "fit_time": r.get("fit_time", np.nan)})
    return out


def group_metric_rows(name, rows, group, day):
    out = []
    for r in rows:
        t = r["day"]
        g = group[day == t]
        yt, yp = np.asarray(r["y_true"]), np.asarray(r["y_pred"])
        if len(g) != len(yt):
            continue
        for label, mask in (("A", g), ("B", ~g), ("overall", np.ones_like(g, dtype=bool))):
            if mask.sum() == 0:
                continue
            m = day_metrics(yt[mask], yp[mask])
            out.append({"method": name, "day": t, "group": label, **m})
    return out


def shift_days_for(drift, n_days, shift_day):
    if drift in ("abrupt", "local"):
        return [shift_day if shift_day is not None else n_days // 2]
    if drift == "opposing_local":
        return [n_days // 3, (2 * n_days) // 3]
    return []  # none / gradual / recurring / mixed: aggregate excess-over-oracle only


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_source_args(ap)
    ap.add_argument("--n-features", type=int, default=2**18)
    ap.add_argument("--warmup-days", type=int, default=3)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--diff-forgetting-maxiter", type=int, default=12)
    ap.add_argument("--no-diff-forgetting", action="store_true",
                    help="Skip Differentiable Forgetting (the slow per-day bilevel baseline).")
    ap.add_argument("--recovery-horizon", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    X, y, day, group, context = load_data_with_context(args, args.n_features, args.seed)
    T = int(day.max())
    print(f"Loaded {X.shape[0]} rows, {T + 1} days, click rate {y.mean():.4f}", flush=True)

    eligible_days, dev_days, test_days = compute_splits(day, args.warmup_days, args.test_frac)
    eligible_days = list(eligible_days)
    test_set = set(test_days)
    dev_test_days = sorted(set(eligible_days))
    print(f"{len(eligible_days)} eligible days, {len(dev_days)} dev, {len(test_days)} locked test", flush=True)

    candidates = build_candidates()
    print("Building WINDOW_FAMILY candidate bank ...", flush=True)
    t0 = time.time()
    bank = build_candidate_bank(X, y, day, eligible_days, alpha=args.alpha, seed=args.seed, n_jobs=args.n_jobs)
    print(f"  {time.time() - t0:.1f}s", flush=True)

    methods = {}   # name -> rows
    weights = {}   # name -> (weight_rows, expert_names)

    for name in WINDOW_FAMILY:
        methods[name] = _rows_from_bank(bank, name, eligible_days)
    methods["uniform5"] = _uniform5_rows(bank, eligible_days)

    h_val, val_rows = _validation_selected_rows(bank, set(dev_days), dev_test_days)
    methods[f"validation_selected(h={h_val})"] = val_rows

    print("Exponential forgetting (decay_hl) ...", flush=True)
    t0 = time.time()
    methods.update(_decay_rows(X, y, day, eligible_days, candidates, args.alpha, args.seed, args.n_jobs))
    print(f"  {time.time() - t0:.1f}s", flush=True)

    print("Han ARW ...", flush=True)
    t0 = time.time()
    han_rows = run_han_arw(bank, eligible_days, dev_days=set(dev_days))
    methods["han_arw"] = [{"day": r["day"], "y_true": r["y_true"], "y_pred": r["y_pred"],
                           "n_train": r["n_train"], "fit_time": r["fit_time"]} for r in han_rows]
    pd.DataFrame([{"day": r["day"], "selected_window": r["selected_window"]} for r in han_rows]) \
        .to_csv(out / "han_arw_selected_window.csv", index=False)
    print(f"  {time.time() - t0:.1f}s", flush=True)

    print("AdaMoE ...", flush=True)
    adamoe_rows = run_adamoe(bank, eligible_days)
    methods["adamoe"] = [{k: r[k] for k in ("day", "y_true", "y_pred", "n_train", "fit_time")} for r in adamoe_rows]
    weights["adamoe"] = ([{"day": r["day"], "mean_weights": r["weights"]} for r in adamoe_rows], list(WINDOW_FAMILY))

    if not args.no_diff_forgetting:
        print("Differentiable Forgetting ...", flush=True)
        t0 = time.time()
        methods["diff_forgetting"] = _diff_forgetting_rows(
            X, y, day, eligible_days, args.alpha, args.seed, args.n_jobs, args.diff_forgetting_maxiter)
        print(f"  {time.time() - t0:.1f}s", flush=True)

    print("M1 global mix ...", flush=True)
    m1_rows = run_m1(bank, eligible_days)
    methods["m1_global_mix"] = [{k: r[k] for k in ("day", "y_true", "y_pred", "n_train", "fit_time")} for r in m1_rows]

    print("M2 context gate ...", flush=True)
    m2_rows = run_m2(bank, eligible_days, T=T, context=context, day=day, seed=args.seed)
    methods["m2_context_gate"] = [{k: r[k] for k in ("day", "y_true", "y_pred", "n_train", "fit_time")} for r in m2_rows]

    m5_by_smooth = {}
    for s in M5B_SMOOTH_GRID:
        label = smooth_label(s)
        print(f"M5b {label} ...", flush=True)
        r = run_m5(bank, eligible_days, T=T, smooth_reg=s, context=context, day=day, seed=args.seed)
        m5_by_smooth[s] = r
        methods[label] = [{k: rr[k] for k in ("day", "y_true", "y_pred", "n_train", "fit_time")} for rr in r]
        weights[label] = ([{"day": rr["day"], "mean_weights": rr["mean_weights"]} for rr in r], list(WINDOW_FAMILY))

    print("ensemble3 (diagnostic ceiling) ...", flush=True)
    ens3_rows = run_ensemble3(m2_rows, m5_by_smooth[M5B_DEFAULT_SMOOTH], m5_by_smooth[M5B_HIGH_SMOOTH],
                              T=T, context=context, day=day, seed=args.seed)
    methods["ensemble3"] = [{k: r[k] for k in ("day", "y_true", "y_pred", "n_train", "fit_time")} for r in ens3_rows]
    weights["ensemble3"] = ([{"day": r["day"], "mean_weights": r["mean_weights"]} for r in ens3_rows], list(ENS3_EXPERTS))

    # ---- per-day metrics (computed once; all aggregation reads these) ----
    per_day = []
    for name, rows in methods.items():
        per_day += per_day_metric_rows(name, rows, test_set)
    per_day_df = pd.DataFrame(per_day)
    per_day_df.to_csv(out / "per_day_metrics.csv", index=False)
    test_df = per_day_df[per_day_df["is_test"]]

    grp_df = None
    if group is not None:
        grp = []
        for name, rows in methods.items():
            grp += group_metric_rows(name, rows, group, day)
        grp_df = pd.DataFrame(grp)
        grp_df.to_csv(out / "group_per_day_metrics.csv", index=False)

    # ---- gate dynamics -------------------------------------------------
    gd_frames = []
    for name, (wrows, experts) in weights.items():
        gdf = pd.DataFrame(gate_dynamics(wrows, experts))
        gdf.insert(0, "method", name)
        gd_frames.append(gdf)
    if gd_frames:
        pd.concat(gd_frames, ignore_index=True).to_csv(out / "gate_dynamics.csv", index=False)

    # ---- oracle diagnostics -----------------------------------------------
    oracle_rows = oracle_per_day_frame(
        bank, eligible_days, group=group, day=day,
        m5b_low=m5_by_smooth[M5B_DEFAULT_SMOOTH], m5b_high=m5_by_smooth[M5B_HIGH_SMOOTH])
    pd.DataFrame(oracle_rows).to_csv(out / "oracle_per_day.csv", index=False)
    oracle_test_rows = [r for r in oracle_rows if r["day"] in test_set]

    o1_name, o1_loss, o1_scores = best_fixed_horizon(bank, test_days)

    # ---- summary --------------------------------------------------------
    drift = args.synthetic_drift if args.source == "synthetic" else "criteo"
    sdays = shift_days_for(drift, args.synthetic_days, args.synthetic_shift_day)

    static_baselines = list(WINDOW_FAMILY) + [k for k in methods if k.startswith("decay_hl")]
    def wtm(name, key="log_loss"):
        return weighted_mean_df(test_df[test_df["method"] == name], key)
    best_static_loss = min(wtm(b) for b in static_baselines)

    grp_test = grp_df[grp_df["day"].isin(test_set)] if grp_df is not None else None

    method_summ = {}
    for name, rows in methods.items():
        mdf = test_df[test_df["method"] == name]
        mean, lo, hi = block_bootstrap_ci_df(mdf, "log_loss", seed=args.seed)
        adapt = adaptation_metrics(rows, oracle_test_rows, list(test_days), sdays,
                                   horizon=args.recovery_horizon)
        ll = weighted_mean_df(mdf, "log_loss")
        entry = {
            "log_loss": ll,
            "log_loss_ci95": [lo, hi],
            "brier": weighted_mean_df(mdf, "brier"),
            "pr_auc": weighted_mean_df(mdf, "pr_auc"),
            "roc_auc": weighted_mean_df(mdf, "roc_auc"),
            "ece": weighted_mean_df(mdf, "ece"),
            "n_test_days": int(len(mdf)),
            "stationary_downside": (ll - best_static_loss) if drift == "none" else None,
            **adapt,
        }
        if grp_test is not None:
            for lbl in ("A", "B"):
                sub = grp_test[(grp_test["method"] == name) & (grp_test["group"] == lbl)]
                if len(sub):
                    entry[f"log_loss_{lbl}"] = weighted_mean_df(sub, "log_loss")
        method_summ[name] = entry

    summary = {
        "config": {
            "source": args.source, "drift": drift, "seed": args.seed,
            "synthetic_days": args.synthetic_days, "synthetic_rows_per_day": args.synthetic_rows_per_day,
            "synthetic_period_days": args.synthetic_period_days,
            "synthetic_shift_day": args.synthetic_shift_day, "sample_frac": args.sample_frac,
            "warmup_days": args.warmup_days, "test_frac": args.test_frac,
            "n_eligible_days": len(eligible_days), "n_dev_days": len(dev_days), "n_test_days": len(test_days),
            "shift_days": sdays, "validation_selected_h": h_val,
            "method_under_test": METHOD_UNDER_TEST,
        },
        "oracle": {
            "best_fixed_horizon": o1_name, "best_fixed_horizon_loss": o1_loss,
            "fixed_horizon_scores": o1_scores,
            "oracle_fixed_smoothness": (
                "high" if wtm(smooth_label(M5B_HIGH_SMOOTH)) < wtm(smooth_label(M5B_DEFAULT_SMOOTH))
                else "low"),
            "m5b_low_test_loss": wtm(smooth_label(M5B_DEFAULT_SMOOTH)),
            "m5b_high_test_loss": wtm(smooth_label(M5B_HIGH_SMOOTH)),
            "oracle_persistence_switch_frac": float(np.mean(
                [1.0 if r.get("oracle_persistence") == "high" else 0.0 for r in oracle_test_rows]))
                if oracle_test_rows else None,
        },
        "methods": method_summ,
        "runtime_s": time.time() - t_start,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # quick human-readable table
    tbl = pd.DataFrame([
        {"method": k, "log_loss": v["log_loss"], "ci_lo": v["log_loss_ci95"][0], "ci_hi": v["log_loss_ci95"][1],
         "brier": v["brier"], "roc_auc": v["roc_auc"], "ece": v["ece"],
         "mean_excess_vs_oracle": v.get("mean_excess_over_oracle")}
        for k, v in method_summ.items()]).sort_values("log_loss").reset_index(drop=True)
    tbl.to_csv(out / "comparison_table.csv", index=False)
    print("\n=== locked-test comparison ===", flush=True)
    print(tbl.to_string(index=False), flush=True)
    print(f"\nmethod under test: {METHOD_UNDER_TEST}  |  best fixed horizon (O1): {o1_name} {o1_loss:.4f}", flush=True)
    print(f"total runtime {time.time() - t_start:.1f}s -> {out}/", flush=True)


if __name__ == "__main__":
    main()
