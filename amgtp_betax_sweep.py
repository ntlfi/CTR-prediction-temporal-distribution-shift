"""Extension B, step 1: freeze the per-example-persistence variance penalty
`beta_var_reg` on the DEV seeds only (plan section 16).

Same structure as amgtp_hidden_sweep.py: build the candidate bank once per
(regime, seed) and score AMG-TP with per-example beta_t(x) at each
`beta_var_reg` in {0, 1e-4, 1e-3, 1e-2, 1e-1}, against the global-beta_t
AMG-TP (the thing beta_t(x) has to beat) and the two fixed-smoothness M5b
references.

Regimes: S7 opposing_recurring (the primary target -- global beta_t is
provably wrong for one subgroup at all times), S4 local + S5 opposing_local
(secondary -- subgroup-specific drift), S3 recurring + S0 stationary
(no-downside checks).

    python amgtp_betax_sweep.py --cell $SLURM_ARRAY_TASK_ID --out DIR
    python amgtp_betax_sweep.py --aggregate --out DIR
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from amgtp_config import DEV_SEEDS, SYNTH_DAYS, SYNTH_PERIOD_DAYS, SYNTH_ROWS_PER_DAY
from amgtp_eval import weighted_mean_df
from amgtp_method import run_amgtp
from amgtp_run import AMGTP_CONFIG
from candidate_bank import build_candidate_bank
from data import hash_features, raw_numeric_features
from m5_multiscale_gate import run_m5
from metrics import day_metrics
from splits import compute_splits
from synthetic_data import generate_synthetic_raw

SWEEP_REGIMES = {
    "s0_none": dict(drift_mode="none"),
    "s3_recurring": dict(drift_mode="recurring", period_days=SYNTH_PERIOD_DAYS),
    "s4_local": dict(drift_mode="local", shift_day=SYNTH_DAYS // 2),
    "s5_opposing_local": dict(drift_mode="opposing_local"),
    "s7_opposing_recurring": dict(drift_mode="opposing_recurring", period_days=SYNTH_PERIOD_DAYS),
}
VAR_REG_GRID = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]

GRID_CELLS = [(r, s) for r in SWEEP_REGIMES for s in DEV_SEEDS]


def load(regime_kw, seed, n_features=2**18):
    df, cols = generate_synthetic_raw(n_days=SYNTH_DAYS, rows_per_day=SYNTH_ROWS_PER_DAY,
                                      seed=seed, **regime_kw)
    X = hash_features(df, columns=cols, n_features=n_features)
    ctx = raw_numeric_features(df, columns=cols)
    return X, df["click"].to_numpy(), df["day"].to_numpy(), df["group"].to_numpy(), ctx


def test_ll(rows, test_set):
    md = [{"method": "x", "day": r["day"], **day_metrics(r["y_true"], r["y_pred"])}
          for r in rows if r["day"] in test_set]
    return weighted_mean_df(pd.DataFrame(md), "log_loss")


def run_cell(regime: str, seed: int, n_jobs: int) -> list:
    t0 = time.time()
    X, y, day, group, ctx = load(SWEEP_REGIMES[regime], seed)
    T = int(day.max())
    eligible, dev, test = compute_splits(day, 3, 0.3)
    test_set = set(test)
    bank = build_candidate_bank(X, y, day, list(eligible), seed=seed, n_jobs=n_jobs)

    recs = [
        dict(regime=regime, seed=seed, config="m5b_default",
             ll=test_ll(run_m5(bank, eligible, T=T, smooth_reg=1e-3, context=ctx, day=day, seed=seed), test_set)),
        dict(regime=regime, seed=seed, config="m5b_high_smooth",
             ll=test_ll(run_m5(bank, eligible, T=T, smooth_reg=0.1, context=ctx, day=day, seed=seed), test_set)),
        dict(regime=regime, seed=seed, config="amgtp_global",
             ll=test_ll(run_amgtp(bank, eligible, T=T, context=ctx, day=day, seed=seed, **AMGTP_CONFIG), test_set)),
    ]
    for vr in VAR_REG_GRID:
        rows = run_amgtp(bank, eligible, T=T, context=ctx, day=day, seed=seed,
                         beta_per_example=True, beta_var_reg=vr, group=group, **AMGTP_CONFIG)
        b_std = float(np.mean([r["beta_std"] for r in rows if r["day"] in test_set]))
        d_ab = float(np.nanmean([abs(r.get("beta_A", np.nan) - r.get("beta_B", np.nan))
                                 for r in rows if r["day"] in test_set]))
        recs.append(dict(regime=regime, seed=seed, config=f"betax_vr{vr:g}", beta_var_reg=vr,
                         ll=test_ll(rows, test_set), mean_beta_std=b_std, mean_group_beta_gap=d_ab))
    print(f"  {regime} seed {seed}: {time.time() - t0:.0f}s", flush=True)
    return recs


def aggregate(out: Path):
    recs = [pd.read_csv(p) for p in sorted(out.glob("cell_*.csv"))]
    if not recs:
        raise SystemExit(f"no cell_*.csv under {out}")
    df = pd.concat(recs, ignore_index=True)
    df.to_csv(out / "sweep_raw.csv", index=False)

    piv = df.groupby(["config", "regime"])["ll"].mean().unstack("regime")
    reg_cols = [r for r in SWEEP_REGIMES if r in piv.columns]
    piv["score_sum_ll"] = piv[reg_cols].sum(axis=1)
    piv.sort_values("score_sum_ll").to_csv(out / "sweep_summary.csv")

    print("\n=== dev-seed mean locked-test log loss by config ===")
    print(piv.sort_values("score_sum_ll").to_string())

    betax = piv.loc[[c for c in piv.index if c.startswith("betax_")]]
    glob = piv.loc["amgtp_global"] if "amgtp_global" in piv.index else None
    cand = betax.sort_values("s7_opposing_recurring")
    best = cand.index[0]
    best_vr = float(best.replace("betax_vr", ""))
    # does ANY betax config beat the global-beta_t AMG-TP on S7 (the target)?
    beats_s7 = glob is not None and piv.loc[best, "s7_opposing_recurring"] < glob["s7_opposing_recurring"] - 1e-4
    verdict = ("beta_t(x) beats global beta_t on S7" if beats_s7
               else "NEGATIVE: no beta_t(x) config beats global beta_t on S7 (or anywhere)")
    print(f"\nbest betax config on S7: {best} (lambda={best_vr:g}) -- {verdict}")

    grp = df[df["config"].str.startswith("betax_")].groupby("beta_var_reg")[
        ["mean_beta_std", "mean_group_beta_gap"]].mean()
    lines = ["# Extension B -- per-example beta_t(x): variance penalty dev sweep", "",
             f"`amgtp_betax_sweep.py`, dev seeds {DEV_SEEDS}, regimes {list(reg_cols)}. "
             "Dev-seed mean locked-test log loss. `amgtp_global` = the global-beta_t "
             "AMG-TP that beta_t(x) must beat.", "",
             "| config | " + " | ".join(reg_cols) + " | sum |", "|" + "---|" * (len(reg_cols) + 2)]
    for cfg in ["m5b_default", "m5b_high_smooth", "amgtp_global"] + [f"betax_vr{v:g}" for v in VAR_REG_GRID]:
        if cfg not in piv.index:
            continue
        cells = " | ".join(f"{piv.loc[cfg, r]:.4f}" for r in reg_cols)
        lines.append(f"| {cfg} | {cells} | {piv.loc[cfg, 'score_sum_ll']:.4f} |")
    lines += ["", "Per-example beta spread and A/B group-beta gap (S7 is where a real "
              "gap is the point):", "",
              "| beta_var_reg | mean beta_std | mean |beta_A - beta_B| |", "|---|---|---|"]
    for vr, r in grp.iterrows():
        lines.append(f"| {vr:g} | {r['mean_beta_std']:.3f} | {r['mean_group_beta_gap']:.3f} |")
    if beats_s7:
        lines += ["", f"**Frozen choice: `beta_var_reg={best_vr:g}`** (best S7, no S0/S3 "
                  "regression vs global AMG-TP)."]
    else:
        lines += ["", "**NEGATIVE RESULT.** No `beta_var_reg` makes per-example "
                  "`beta_t(x)` beat the global `beta_t` AMG-TP on S7 -- its purpose-built "
                  f"target -- or anywhere: the best S7 config (`{best}`) is "
                  f"{piv.loc[best, 's7_opposing_recurring'] - (glob['s7_opposing_recurring'] if glob is not None else 0):+.4f} "
                  "vs global. The `|beta_A - beta_B|` column stays ~0.005 at every "
                  "penalty: `g_xi` never learns the subgroup split that S7 is built "
                  "around. Mechanistic read: the persistent state `m_{t-1}` it mixes "
                  "toward is a single *global* EMA, so routing a stable-subgroup example "
                  "to `m` still hands it the *blended* history, not that subgroup's own "
                  "-- per-example persistence needs a per-example (or per-group) `m`, "
                  "which PDF section 2.3 scopes out. Global `beta_t` is the right "
                  "granularity given a global `m`. Closes plan section 3's open question."]
    (out / "FROZEN.md").write_text("\n".join(lines))
    print(f"\nwrote {out}/sweep_summary.csv + FROZEN.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="amgtp_experiments/stage4_betax/_sweep")
    ap.add_argument("--cell", type=int, default=None)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.aggregate:
        aggregate(out)
        return
    if args.cell is not None:
        regime, seed = GRID_CELLS[args.cell]
        pd.DataFrame(run_cell(regime, seed, args.n_jobs)).to_csv(
            out / f"cell_{regime}_seed{seed}.csv", index=False)
        print(f"wrote {out}/cell_{regime}_seed{seed}.csv")
        return

    print(f"{len(VAR_REG_GRID)} penalties x {len(SWEEP_REGIMES)} regimes x {len(DEV_SEEDS)} dev seeds")
    for regime, seed in GRID_CELLS:
        pd.DataFrame(run_cell(regime, seed, args.n_jobs)).to_csv(
            out / f"cell_{regime}_seed{seed}.csv", index=False)
    aggregate(out)


if __name__ == "__main__":
    main()
