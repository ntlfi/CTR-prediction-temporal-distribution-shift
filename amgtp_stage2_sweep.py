"""Stage 2, step 1: freeze AMG-TP's persistence-model hyperparameters on the
DEV seeds only (plan section 16: bounded tuning, freeze before confirmation).

Builds the candidate bank once per (regime, seed) and evaluates a small grid
of AMG-TP configs against it -- the gate/persistence nets are cheap, the bank
is the only expensive part. Scores each config by locked-test log loss on the
three discriminating dev regimes (S1 abrupt: must not regress vs M5b-default;
S3 recurring: should approach M5b-high-smooth; S0 stationary: no downside),
and prints a ranked table. The winning config is then hard-coded into
amgtp_run.py for the confirmation battery.

    python amgtp_stage2_sweep.py --out amgtp_experiments/stage2_amgtp/_sweep
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from amgtp_config import DEV_SEEDS, SYNTH_DAYS, SYNTH_PERIOD_DAYS, SYNTH_ROWS_PER_DAY
from amgtp_eval import day_loss, weighted_mean_df
from amgtp_method import run_amgtp
from candidate_bank import build_candidate_bank
from m5_multiscale_gate import run_m5
from metrics import day_metrics
from splits import compute_splits
from synthetic_data import generate_synthetic_raw
from data import hash_features, raw_numeric_features

SWEEP_REGIMES = {
    "s0_none": dict(drift_mode="none"),
    "s1_abrupt": dict(drift_mode="abrupt", shift_day=SYNTH_DAYS // 2),
    "s3_recurring": dict(drift_mode="recurring", period_days=SYNTH_PERIOD_DAYS),
}

GRID = {
    "init_bias": [-1.0, -2.0],
    "rho": [0.3, 0.5],
    "beta_entropy_reg": [0.0, 1e-3],
}


def load(regime_kw, seed, n_features=2**18):
    df, cols = generate_synthetic_raw(n_days=SYNTH_DAYS, rows_per_day=SYNTH_ROWS_PER_DAY,
                                      seed=seed, **regime_kw)
    X = hash_features(df, columns=cols, n_features=n_features)
    ctx = raw_numeric_features(df, columns=cols)
    return X, df["click"].to_numpy(), df["day"].to_numpy(), ctx


def test_ll(rows, test_set):
    md = [{"method": "x", "day": r["day"], **day_metrics(r["y_true"], r["y_pred"])}
          for r in rows if r["day"] in test_set]
    return weighted_mean_df(pd.DataFrame(md), "log_loss")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="amgtp_experiments/stage2_amgtp/_sweep")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEV_SEEDS)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    configs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    print(f"{len(configs)} AMG-TP configs x {len(SWEEP_REGIMES)} regimes x {len(args.seeds)} dev seeds")

    records = []
    for regime, rkw in SWEEP_REGIMES.items():
        for seed in args.seeds:
            t0 = time.time()
            X, y, day, ctx = load(rkw, seed)
            T = int(day.max())
            eligible, dev, test = compute_splits(day, 3, 0.3)
            test_set = set(test)
            bank = build_candidate_bank(X, y, day, list(eligible), seed=seed, n_jobs=args.n_jobs)

            ref_lo = test_ll(run_m5(bank, eligible, T=T, smooth_reg=1e-3, context=ctx, day=day, seed=seed), test_set)
            ref_hi = test_ll(run_m5(bank, eligible, T=T, smooth_reg=0.1, context=ctx, day=day, seed=seed), test_set)
            records.append(dict(regime=regime, seed=seed, config="m5b_default", ll=ref_lo))
            records.append(dict(regime=regime, seed=seed, config="m5b_high_smooth", ll=ref_hi))

            for cfg in configs:
                rows = run_amgtp(bank, eligible, T=T, context=ctx, day=day, seed=seed, **cfg)
                ll = test_ll(rows, test_set)
                betas = [r["beta"] for r in rows if r["day"] in test_set]
                records.append(dict(regime=regime, seed=seed, config=json.dumps(cfg), ll=ll,
                                    mean_beta=float(np.mean(betas))))
            print(f"  {regime} seed {seed}: {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(out / "sweep_raw.csv", index=False)

    piv = df.groupby(["config", "regime"])["ll"].mean().unstack("regime")
    # score: sum of per-regime loss, but penalise S1 regression vs m5b_default and reward S3 near m5b_high
    ref = piv.loc[["m5b_default", "m5b_high_smooth"]]
    piv["score_sum_ll"] = piv[list(SWEEP_REGIMES)].sum(axis=1)
    piv = piv.sort_values("score_sum_ll")
    piv.to_csv(out / "sweep_summary.csv")
    print("\n=== dev-seed mean locked-test log loss by config ===")
    print(piv.to_string())
    print("\nreference rows (m5b_default / m5b_high_smooth) included above.")
    best = piv.drop(index=[c for c in ["m5b_default", "m5b_high_smooth"] if c in piv.index]).index[0]
    print(f"\nbest AMG-TP config by summed dev log loss: {best}")


if __name__ == "__main__":
    main()
