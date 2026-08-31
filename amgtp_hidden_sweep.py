"""Extension A, step 1: freeze the PersistenceNet hidden width on the DEV
seeds only (plan section 16 -- bounded tuning, freeze before confirmation).

Mirrors amgtp_stage2_sweep.py: build the candidate bank once per (regime,
seed) -- the only expensive part -- and evaluate AMG-TP with each
`persist_hidden` in {0, 4, 8, 16} against it, plus the two fixed-smoothness
M5b references. `persist_hidden=0` is the frozen Stage 2 linear net, so it
doubles as the "does a hidden layer help at all?" control.

Regimes: S0 (no downside), S1 abrupt (reaction speed -- a nonlinear
"drop beta on a loss jump" is the hidden layer's best case), S3 recurring
(persistence), S4 local, S5 opposing-local, S7 opposing-recurring (the new
per-example-persistence test bed -- here just a no-downside / any-gain check
for the global-beta hidden net).

SLURM-arrayable by cell:
    python amgtp_hidden_sweep.py --cell $SLURM_ARRAY_TASK_ID --out DIR   # one (regime, seed)
    python amgtp_hidden_sweep.py --aggregate --out DIR                    # combine -> summary + FROZEN stub
No --cell / --aggregate: run every cell in-process (small local runs only).
"""
import argparse
import json
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
    "s1_abrupt": dict(drift_mode="abrupt", shift_day=SYNTH_DAYS // 2),
    "s3_recurring": dict(drift_mode="recurring", period_days=SYNTH_PERIOD_DAYS),
    "s4_local": dict(drift_mode="local", shift_day=SYNTH_DAYS // 2),
    "s5_opposing_local": dict(drift_mode="opposing_local"),
    "s7_opposing_recurring": dict(drift_mode="opposing_recurring", period_days=SYNTH_PERIOD_DAYS),
}
HIDDEN_GRID = [0, 4, 8, 16]

GRID_CELLS = [(r, s) for r in SWEEP_REGIMES for s in DEV_SEEDS]


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


def run_cell(regime: str, seed: int, n_jobs: int) -> list:
    rkw = SWEEP_REGIMES[regime]
    t0 = time.time()
    X, y, day, ctx = load(rkw, seed)
    T = int(day.max())
    eligible, dev, test = compute_splits(day, 3, 0.3)
    test_set = set(test)
    bank = build_candidate_bank(X, y, day, list(eligible), seed=seed, n_jobs=n_jobs)

    recs = [
        dict(regime=regime, seed=seed, config="m5b_default",
             ll=test_ll(run_m5(bank, eligible, T=T, smooth_reg=1e-3, context=ctx, day=day, seed=seed), test_set)),
        dict(regime=regime, seed=seed, config="m5b_high_smooth",
             ll=test_ll(run_m5(bank, eligible, T=T, smooth_reg=0.1, context=ctx, day=day, seed=seed), test_set)),
    ]
    for h in HIDDEN_GRID:
        rows = run_amgtp(bank, eligible, T=T, context=ctx, day=day, seed=seed,
                         persist_hidden=h, **AMGTP_CONFIG)
        betas = [r["beta"] for r in rows if r["day"] in test_set]
        recs.append(dict(regime=regime, seed=seed, config=f"hidden{h}", persist_hidden=h,
                         ll=test_ll(rows, test_set), mean_beta=float(np.mean(betas))))
    print(f"  {regime} seed {seed}: {time.time() - t0:.0f}s", flush=True)
    return recs


def aggregate(out: Path):
    recs = []
    for p in sorted(out.glob("cell_*.csv")):
        recs.append(pd.read_csv(p))
    if not recs:
        raise SystemExit(f"no cell_*.csv under {out}")
    df = pd.concat(recs, ignore_index=True)
    df.to_csv(out / "sweep_raw.csv", index=False)

    piv = df.groupby(["config", "regime"])["ll"].mean().unstack("regime")
    reg_cols = [r for r in SWEEP_REGIMES if r in piv.columns]
    piv["score_sum_ll"] = piv[reg_cols].sum(axis=1)
    hid = piv.loc[[c for c in piv.index if c.startswith("hidden")]].sort_values("score_sum_ll")
    piv.sort_values("score_sum_ll").to_csv(out / "sweep_summary.csv")

    print("\n=== dev-seed mean locked-test log loss by config ===")
    print(piv.sort_values("score_sum_ll").to_string())
    best = hid.index[0]
    best_h = int(best.replace("hidden", ""))
    print(f"\nbest hidden width by summed dev log loss: {best} (h={best_h})")

    # per-regime: does any hidden>0 beat hidden0, and by how much?
    h0 = piv.loc["hidden0", reg_cols]
    lines = ["# PersistenceNet hidden width -- dev-seed sweep (Extension A)", "",
             f"`amgtp_hidden_sweep.py`, dev seeds {DEV_SEEDS}, regimes "
             f"{list(reg_cols)}. Dev-seed mean locked-test log loss; "
             "`hidden0` = the frozen Stage 2 linear persistence net.", "",
             "| config | " + " | ".join(reg_cols) + " | sum |",
             "|" + "---|" * (len(reg_cols) + 2)]
    for cfg in ["m5b_default", "m5b_high_smooth"] + [f"hidden{h}" for h in HIDDEN_GRID]:
        if cfg not in piv.index:
            continue
        cells = " | ".join(f"{piv.loc[cfg, r]:.4f}" for r in reg_cols)
        lines.append(f"| {cfg} | {cells} | {piv.loc[cfg, 'score_sum_ll']:.4f} |")
    lines += ["", "Delta vs `hidden0` (negative = the hidden layer helps):", "",
              "| config | " + " | ".join(reg_cols) + " |", "|" + "---|" * (len(reg_cols) + 1)]
    for h in HIDDEN_GRID[1:]:
        cfg = f"hidden{h}"
        if cfg not in piv.index:
            continue
        d = piv.loc[cfg, reg_cols] - h0
        lines.append(f"| {cfg} | " + " | ".join(f"{v:+.4f}" for v in d) + " |")
    lines += ["", f"**Frozen choice: `persist_hidden={best_h}`** "
              f"({'linear net retained -- no hidden layer helps on dev' if best_h == 0 else 'nonlinear net'}).",
              "", "`amgtp_hidden8` / `amgtp_hidden16` still run in the Stage 4 battery "
              "(ablation A10) so the negative result is confirmed on the disjoint seeds; "
              "the deployed `amgtp` keeps `persist_hidden=0`."]
    (out / "FROZEN.md").write_text("\n".join(lines))
    print(f"\nwrote {out}/sweep_summary.csv + FROZEN.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="amgtp_experiments/stage4_hidden/_sweep")
    ap.add_argument("--cell", type=int, default=None, help="run only GRID_CELLS[cell] and write cell_*.csv")
    ap.add_argument("--aggregate", action="store_true", help="combine cell_*.csv -> summary + FROZEN.md")
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.aggregate:
        aggregate(out)
        return

    if args.cell is not None:
        regime, seed = GRID_CELLS[args.cell]
        recs = run_cell(regime, seed, args.n_jobs)
        pd.DataFrame(recs).to_csv(out / f"cell_{regime}_seed{seed}.csv", index=False)
        print(f"wrote {out}/cell_{regime}_seed{seed}.csv")
        return

    print(f"{len(HIDDEN_GRID)} hidden widths x {len(SWEEP_REGIMES)} regimes x {len(DEV_SEEDS)} dev seeds")
    for regime, seed in GRID_CELLS:
        recs = run_cell(regime, seed, args.n_jobs)
        pd.DataFrame(recs).to_csv(out / f"cell_{regime}_seed{seed}.csv", index=False)
    aggregate(out)


if __name__ == "__main__":
    main()
