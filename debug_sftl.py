"""Implements sftl-debugging-plan.pdf: a staged failure analysis for SFTL,
distinguishing implementation bug / calibration mismatch / EMA timescale
mismatch / trajectory instability / genuine method limitation.

Does NOT modify sftl.py's core algorithm (per the plan: "do not silently
change SFTL and continue labeling it as the original baseline") -- this
script only instruments and re-runs it under controlled conditions.

Stage 1-3 (implementation invariants, ranking-vs-calibration split,
runaway-margin trace) run on the same 120-day/3000-rows-per-day/shift-at-95
config used for the reported production result (results_synthetic_abrupt),
so the diagnosis directly explains that number. Stage 4-6 (EMA half-life
sweep, trajectory ablation, gradient-contribution lambda tuning) use a
faster 60-day/1500-rows-per-day/shift-at-48 proxy config -- same
qualitative abrupt-shift setup, ~5x cheaper per run, since these stages
need 4-5 full runs each rather than one.

Outputs land in results/sftl_debug/.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from data import hash_indices
from sftl import SFTL, ema_half_life_to_alpha
from splits import compute_splits
from synthetic_data import generate_synthetic_raw

PROD_CFG = dict(n_days=120, rows_per_day=3000, shift_day=95, warmup_days=3, test_frac=0.3)
PROXY_CFG = dict(n_days=60, rows_per_day=1500, shift_day=48, warmup_days=3, test_frac=0.3)
SFTL_KW = dict(epochs_per_domain=5, batch_size=256, seed=0)


def load(cfg, seed=0):
    df, columns = generate_synthetic_raw(n_days=cfg["n_days"], rows_per_day=cfg["rows_per_day"],
                                          drift_mode="abrupt", shift_day=cfg["shift_day"], seed=seed)
    day = df["day"].to_numpy()
    y = df["click"].to_numpy()
    x_idx = hash_indices(df, columns=columns, vocab_size=2**16)
    eligible_days, dev_days, test_days = compute_splits(day, cfg["warmup_days"], cfg["test_frac"])
    return x_idx, y, day, len(columns), eligible_days, dev_days, test_days


def full_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p = np.clip(y_pred, 1e-12, 1 - 1e-12)
    if len(np.unique(y_true)) < 2:
        auc = pr_auc = float("nan")
    else:
        auc = roc_auc_score(y_true, y_pred)
        pr_auc = average_precision_score(y_true, y_pred)
    return {
        "auc": auc, "log_loss": log_loss(y_true, p, labels=[0, 1]), "brier": brier_score_loss(y_true, y_pred),
        "pr_auc": pr_auc, "mean_pred": float(y_pred.mean()), "true_rate": float(y_true.mean()),
        "calib_error": float(abs(y_pred.mean() - y_true.mean())),
    }


def run_instrumented(cfg, sftl_kwargs, collect_gradient_norms_at=None):
    """One full run, logging working/slow/fast metrics, margins, logit
    stats, and per-domain loss components for every eligible day. Returns
    (per_day_df, domain_loss_df, gradient_norm_records)."""
    x_idx, y, day, n_columns, eligible_days, dev_days, test_days = load(cfg)
    eligible = set(int(d) for d in eligible_days)
    model = SFTL(n_columns=n_columns, vocab_size=2**16, seed=sftl_kwargs.get("seed", 0),
                 **{k: v for k, v in sftl_kwargs.items() if k not in ("epochs_per_domain", "seed")})
    rng = np.random.default_rng(sftl_kwargs.get("seed", 0))
    epochs = sftl_kwargs.get("epochs_per_domain", 5)

    per_day_rows, domain_loss_rows, grad_records = [], [], []
    for t in range(int(day.min()), int(day.max()) + 1):
        mask = day == t
        if mask.sum() == 0:
            continue
        x_t, y_t = x_idx[mask], y[mask]

        if t in eligible:
            for name, net in (("working", model.working), ("slow", model.slow), ("fast", model.fast)):
                p = model._predict_net(net, x_t)
                margin, logits = model.margin_and_logits(net, x_t, y_t)
                m = full_metrics(y_t, p)
                per_day_rows.append({
                    "day": t, "learner": name, **m, "margin": margin,
                    "mean_abs_logit": float(np.abs(logits).mean()), "std_logit": float(logits.std()),
                    "max_abs_logit": float(np.abs(logits).max()),
                })

        if collect_gradient_norms_at is not None and t in collect_gradient_norms_at and model.domain_idx >= model.warmup_domains:
            grad_records.append({"day": t, "domain_idx": model.domain_idx, **model.measure_gradient_norms(x_t, y_t)})

        stats = model.train_domain(x_t, y_t, rng, epochs=epochs, collect_stats=True)
        domain_loss_rows.append({"day": t, "domain_idx": model.domain_idx - 1, **stats})

    return pd.DataFrame(per_day_rows), pd.DataFrame(domain_loss_rows), grad_records, test_days


def stage1_invariant(out_dir: Path):
    """H1: with lambda_s=lambda_f=0, the working learner must be numerically
    identical to a model trained on BCE alone (same seed, same data order).
    Since train_domain's loss reduces to plain BCE when both lambdas are 0
    (and use_trajectory-gated branches are simply skipped), this checks
    that reduction actually holds in the running code, not just on paper."""
    print("Stage 1: implementation invariant (lambda=0 vs ordinary BCE baseline)")
    x_idx, y, day, n_columns, eligible_days, dev_days, test_days = load(PROD_CFG)
    rng_a, rng_b = np.random.default_rng(0), np.random.default_rng(0)

    model_zero = SFTL(n_columns=n_columns, vocab_size=2**16, seed=0, lambda_slow=0.0, lambda_fast=0.0,
                       batch_size=SFTL_KW["batch_size"])
    model_plain = SFTL(n_columns=n_columns, vocab_size=2**16, seed=0, use_slow_trajectory=False,
                        use_fast_trajectory=False, batch_size=SFTL_KW["batch_size"])

    max_diff = 0.0
    for t in range(int(day.min()), int(day.min()) + 10):  # first 10 domains is enough to detect divergence
        mask = day == t
        if mask.sum() == 0:
            continue
        x_t, y_t = x_idx[mask], y[mask]
        pa, pb = model_zero.predict_working(x_t), model_plain.predict_working(x_t)
        max_diff = max(max_diff, float(np.abs(pa - pb).max()))
        model_zero.train_domain(x_t, y_t, rng_a, epochs=SFTL_KW["epochs_per_domain"])
        model_plain.train_domain(x_t, y_t, rng_b, epochs=SFTL_KW["epochs_per_domain"])

    passed = max_diff < 1e-6
    result = {"max_abs_prediction_diff": max_diff, "passed": passed}
    print(f"  max |prediction diff| over first 10 domains: {max_diff:.2e} -> {'PASS' if passed else 'FAIL'}")
    (out_dir / "stage1_invariant.json").write_text(json.dumps(result, indent=2))
    return passed


def stage2_3_instrumented(out_dir: Path):
    print("Stage 2/3: instrumented run on the production abrupt config (lambda=0.05, the reported result)...")
    t0 = time.time()
    per_day, domain_loss, _, test_days = run_instrumented(PROD_CFG, {**SFTL_KW, "lambda_slow": 0.05, "lambda_fast": 0.05})
    print(f"  done in {time.time() - t0:.1f}s")
    per_day.to_csv(out_dir / "stage2_per_day_metrics.csv", index=False)
    domain_loss.to_csv(out_dir / "stage2_domain_loss_components.csv", index=False)

    # Figure: daily AUC and log loss around the shift, working vs fast.
    shift = PROD_CFG["shift_day"]
    window = per_day[(per_day["day"] >= shift - 10) & (per_day["day"] <= shift + 20)]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    for name, g in window.groupby("learner"):
        g = g.sort_values("day")
        axes[0].plot(g["day"], g["auc"], marker="o", markersize=3, label=name)
        axes[1].plot(g["day"], g["log_loss"], marker="o", markersize=3, label=name)
    axes[0].axvline(shift, color="k", linestyle="--", alpha=0.5)
    axes[1].axvline(shift, color="k", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("AUC"); axes[0].legend(fontsize=8); axes[0].set_title("Daily AUC around the abrupt shift")
    axes[1].set_ylabel("log loss"); axes[1].set_xlabel("prediction day")
    axes[1].set_title("Daily log loss around the abrupt shift")
    fig.tight_layout()
    fig.savefig(out_dir / "stage2_abrupt_shift_auc_logloss.png", dpi=150)
    plt.close(fig)

    # Figure: margin / logit trajectories (Stage 3).
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    for name, g in per_day.groupby("learner"):
        g = g.sort_values("day")
        axes[0].plot(g["day"], g["margin"], marker="o", markersize=2, label=name)
        axes[1].plot(g["day"], g["max_abs_logit"], marker="o", markersize=2, label=name)
    axes[0].axvline(shift, color="k", linestyle="--", alpha=0.5)
    axes[1].axvline(shift, color="k", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("positive-negative margin"); axes[0].legend(fontsize=8)
    axes[0].set_title("Margin trajectories, full training run")
    axes[1].set_ylabel("max |logit|"); axes[1].set_xlabel("prediction day")
    axes[1].set_title("Confidence (max |logit|) trajectories, full training run")
    fig.tight_layout()
    fig.savefig(out_dir / "stage3_margin_logit_trajectories.png", dpi=150)
    plt.close(fig)

    return per_day, domain_loss, test_days


def stage4_ema_sweep(out_dir: Path, half_lives=(10, 30, 100, 1000)):
    print(f"Stage 4: EMA half-life sweep H={half_lives} (proxy config, lambda=0.05) ...")
    x_idx, y, day, n_columns, eligible_days, dev_days, test_days = load(PROXY_CFG)
    shift = PROXY_CFG["shift_day"]
    rows = []
    for H in half_lives:
        alpha = ema_half_life_to_alpha(H)
        t0 = time.time()
        per_day, _, _, _ = run_instrumented(PROXY_CFG, {**SFTL_KW, "lambda_slow": 0.05, "lambda_fast": 0.05,
                                                          "ema_alpha": alpha})
        fast = per_day[per_day["learner"] == "fast"].sort_values("day").set_index("day")
        pre_shift_ll = fast.loc[fast.index < shift, "log_loss"].mean()
        post = fast.loc[fast.index >= shift, "log_loss"]
        # Recovery time: first day post-shift where log loss returns within
        # 20% of the pre-shift baseline (or "never" within the test window).
        recovered = post[post <= 1.2 * pre_shift_ll]
        recovery_day = int(recovered.index[0] - shift) if len(recovered) else None
        overall_ll = fast["log_loss"].mean()
        rows.append({"half_life_steps": H, "alpha": alpha, "pre_shift_log_loss": pre_shift_ll,
                     "peak_post_shift_log_loss": float(post.max()), "recovery_days": recovery_day,
                     "overall_log_loss": overall_ll})
        print(f"  H={H} (alpha={alpha:.4f}): pre-shift={pre_shift_ll:.3f}, "
              f"peak={post.max():.3f}, recovery={recovery_day}d, overall={overall_ll:.3f}  [{time.time()-t0:.0f}s]")
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "stage4_ema_half_life_sweep.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = table.dropna(subset=["recovery_days"])
    ax.plot(plotted["half_life_steps"], plotted["recovery_days"], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("EMA half-life (optimizer steps)")
    ax.set_ylabel("days to recover within 20% of pre-shift log loss")
    ax.set_title("EMA timescale vs. post-shift recovery time")
    fig.tight_layout()
    fig.savefig(out_dir / "stage4_halflife_vs_recovery.png", dpi=150)
    plt.close(fig)
    return table


def stage5_ablation(out_dir: Path):
    print("Stage 5: ablation A(BCE-only) / B(slow-only) / C(fast-only) / D(full) (proxy config) ...")
    variants = {
        "A_bce_only": dict(use_slow_trajectory=False, use_fast_trajectory=False),
        "B_slow_only": dict(use_slow_trajectory=True, use_fast_trajectory=False),
        "C_fast_only": dict(use_slow_trajectory=False, use_fast_trajectory=True),
        "D_full_sftl": dict(use_slow_trajectory=True, use_fast_trajectory=True),
    }
    rows = []
    for name, kw in variants.items():
        t0 = time.time()
        per_day, _, _, test_days = run_instrumented(PROXY_CFG, {**SFTL_KW, "lambda_slow": 0.05,
                                                                  "lambda_fast": 0.05, **kw})
        test_mask = per_day["day"].isin(test_days)
        for learner in ("working", "fast"):
            sub = per_day[test_mask & (per_day["learner"] == learner)]
            rows.append({
                "variant": name, "learner": learner,
                "log_loss": np.average(sub["log_loss"], weights=None), "auc": sub["auc"].mean(),
                "brier": sub["brier"].mean(), "pr_auc": sub["pr_auc"].mean(),
                "mean_abs_logit": sub["mean_abs_logit"].mean(), "calib_error": sub["calib_error"].mean(),
            })
        print(f"  {name}: done in {time.time()-t0:.0f}s "
              f"(fast log_loss={rows[-1]['log_loss']:.3f}, auc={rows[-1]['auc']:.3f})")
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "stage5_ablation_table.csv", index=False)
    return table


def stage6_gradient_ratio(out_dir: Path, targets=(0.01, 0.05, 0.10, 0.25, 0.50)):
    print("Stage 6: measuring gradient contribution ratios (early vs. pre-shift), then tuning lambda ...")
    x_idx, y, day, n_columns, eligible_days, dev_days, test_days = load(PROXY_CFG)
    warmup_domains = SFTL_KW.get("warmup_domains", 3)
    shift = PROXY_CFG["shift_day"]
    model = SFTL(n_columns=n_columns, vocab_size=2**16, seed=0, lambda_slow=1.0, lambda_fast=1.0)
    rng = np.random.default_rng(0)

    # Two snapshots: right after warmup (initial contribution), and right
    # before the shift (does the ratio grow as margins escalate over
    # training, independent of any actual distribution change?).
    days_sorted = sorted(np.unique(day))
    snapshots = {}
    for t in days_sorted:
        mask = day == t
        x_t, y_t = x_idx[mask], y[mask]
        if model.domain_idx == warmup_domains:
            snapshots["early"] = {"day": int(t), **model.measure_gradient_norms(x_t, y_t)}
        if model.domain_idx == shift - 1:
            snapshots["pre_shift"] = {"day": int(t), **model.measure_gradient_norms(x_t, y_t)}
        model.train_domain(x_t, y_t, rng, epochs=SFTL_KW["epochs_per_domain"])
    (out_dir / "stage6_gradient_snapshot.json").write_text(json.dumps(snapshots, indent=2))
    for name, snap in snapshots.items():
        ratio_s = snap["traj_s_grad_norm"] / snap["bce_grad_norm"] if snap["bce_grad_norm"] else float("nan")
        ratio_f = snap["traj_f_grad_norm"] / snap["bce_grad_norm"] if snap["bce_grad_norm"] else float("nan")
        print(f"  [{name}, day {snap['day']}] bce_grad={snap['bce_grad_norm']:.4g}, "
              f"traj_s_grad={snap['traj_s_grad_norm']:.4g} (ratio at lambda=1: {ratio_s:.2f}x BCE), "
              f"traj_f_grad={snap['traj_f_grad_norm']:.4g} (ratio at lambda=1: {ratio_f:.2f}x BCE)")

    # Use the EARLY snapshot for the lambda-targeting sweep (Stage 6's own
    # instruction: "target approximate INITIAL contribution levels").
    probe = snapshots["early"]
    # lambda * ||grad_traj|| / ||grad_bce|| = target  =>  lambda = target * ||grad_bce|| / ||grad_traj||
    bce_norm = probe["bce_grad_norm"]
    traj_s_norm, traj_f_norm = probe["traj_s_grad_norm"], probe["traj_f_grad_norm"]
    rows = []
    for target in targets:
        lam_s = target * bce_norm / traj_s_norm if traj_s_norm > 0 else float("nan")
        lam_f = target * bce_norm / traj_f_norm if traj_f_norm > 0 else float("nan")
        t0 = time.time()
        per_day, _, _, test_days = run_instrumented(PROXY_CFG, {**SFTL_KW, "lambda_slow": lam_s, "lambda_fast": lam_f})
        test_mask = per_day["day"].isin(test_days) & (per_day["learner"] == "fast")
        sub = per_day[test_mask]
        rows.append({"target_contribution": target, "lambda_slow": lam_s, "lambda_fast": lam_f,
                     "log_loss": sub["log_loss"].mean(), "auc": sub["auc"].mean(),
                     "calib_error": sub["calib_error"].mean()})
        print(f"  target={target:.0%}: lambda_s={lam_s:.4f}, lambda_f={lam_f:.4f} -> "
              f"log_loss={rows[-1]['log_loss']:.3f}, auc={rows[-1]['auc']:.3f}  [{time.time()-t0:.0f}s]")
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "stage6_gradient_ratio_lambda_sweep.csv", index=False)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", default=["1", "2", "3", "4", "5", "6"],
                         help="Which stages to run (2 and 3 share one instrumented run).")
    parser.add_argument("--out", default="results/sftl_debug")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "1" in args.stages:
        stage1_invariant(out_dir)
    if "2" in args.stages or "3" in args.stages:
        stage2_3_instrumented(out_dir)
    if "4" in args.stages:
        stage4_ema_sweep(out_dir)
    if "5" in args.stages:
        stage5_ablation(out_dir)
    if "6" in args.stages:
        stage6_gradient_ratio(out_dir)

    print(f"\nAll requested stages done. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
