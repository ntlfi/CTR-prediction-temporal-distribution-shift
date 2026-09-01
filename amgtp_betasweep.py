"""Package 1 -- comprehensive fixed-beta sweep.

Holds the context gate and the 5-expert bank fixed, and evaluates
pi_t(x) = (1 - beta) q_t(x) + beta m_{t-1} at a dense grid
beta in {0, 0.05, ..., 0.95, 1.0} on every synthetic regime, alongside the
adaptive-beta AMG-TP and ensemble3. The question this answers: does adaptive
beta_t actually improve over the single best fixed beta chosen in hindsight
(per regime), or only interpolate between the two previously tested
low/high configs?

Per (regime, seed): build the bank once, then run
  - run_amgtp(adaptive_beta=False, fixed_beta=b) for every b in BETA_GRID
  - run_amgtp(adaptive_beta=True)            (AMG-TP)
  - ensemble3
  - m5b_smooth0.001 / 0.1                    (context for the reader)
and record locked-test log loss for each.

    python amgtp_betasweep.py --cell $SLURM_ARRAY_TASK_ID --out DIR
    python amgtp_betasweep.py --aggregate --out DIR
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from amgtp_config import ALL_SEEDS, SYNTH_DAYS, SYNTH_PERIOD_DAYS, SYNTH_REGIMES, SYNTH_ROWS_PER_DAY
from amgtp_eval import weighted_mean_df
from amgtp_method import run_amgtp
from amgtp_run import AMGTP_CONFIG
from candidate_bank import build_candidate_bank
from data import hash_features, raw_numeric_features
from ensemble3 import run_ensemble3
from m5_multiscale_gate import run_m5
from metrics import day_metrics
from splits import compute_splits
from synthetic_data import generate_synthetic_raw

BETA_GRID = [round(b, 2) for b in np.arange(0.0, 1.0001, 0.05)]
GRID_CELLS = [(rk, s) for rk in SYNTH_REGIMES for s in ALL_SEEDS]


def _regime_kw(rk):
    r = SYNTH_REGIMES[rk]
    kw = dict(drift_mode=r["drift"], period_days=r.get("period", SYNTH_PERIOD_DAYS))
    if "shift_day" in r:
        kw["shift_day"] = r["shift_day"]
    return kw


def load(rk, seed, n_features=2**18):
    df, cols = generate_synthetic_raw(n_days=SYNTH_DAYS, rows_per_day=SYNTH_ROWS_PER_DAY,
                                      seed=seed, **_regime_kw(rk))
    X = hash_features(df, columns=cols, n_features=n_features)
    ctx = raw_numeric_features(df, columns=cols)
    return X, df["click"].to_numpy(), df["day"].to_numpy(), df["group"].to_numpy(), ctx


def test_ll(rows, test_set):
    md = [{"day": r["day"], **day_metrics(r["y_true"], r["y_pred"])}
          for r in rows if r["day"] in test_set]
    return weighted_mean_df(pd.DataFrame(md), "log_loss")


def run_cell(rk, seed, n_jobs):
    import time
    t0 = time.time()
    X, y, day, grp, ctx = load(rk, seed)
    T = int(day.max())
    elig, dev, test = compute_splits(day, 3, 0.3)
    ts = set(test)
    bank = build_candidate_bank(X, y, day, list(elig), seed=seed, n_jobs=n_jobs)

    recs = []
    for b in BETA_GRID:
        r = run_amgtp(bank, elig, T=T, context=ctx, day=day, seed=seed,
                      adaptive_beta=False, fixed_beta=b, **AMGTP_CONFIG)
        recs.append(dict(regime=rk, seed=seed, config=f"fixed_beta_{b:.2f}", beta=b, ll=test_ll(r, ts)))
    r_ad = run_amgtp(bank, elig, T=T, context=ctx, day=day, seed=seed, **AMGTP_CONFIG)
    recs.append(dict(regime=rk, seed=seed, config="amgtp_adaptive", beta=np.nan, ll=test_ll(r_ad, ts)))

    m5_lo = run_m5(bank, elig, T=T, smooth_reg=1e-3, context=ctx, day=day, seed=seed)
    m5_hi = run_m5(bank, elig, T=T, smooth_reg=0.1, context=ctx, day=day, seed=seed)
    from m2_context_gate import run_m2
    m2 = run_m2(bank, elig, T=T, context=ctx, day=day, seed=seed)
    ens = run_ensemble3(m2, m5_lo, m5_hi, T=T, context=ctx, day=day, seed=seed)
    for nm, rr in (("m5b_smooth0.001", m5_lo), ("m5b_smooth0.1", m5_hi), ("ensemble3", ens)):
        recs.append(dict(regime=rk, seed=seed, config=nm, beta=np.nan, ll=test_ll(rr, ts)))
    print(f"  {rk} seed {seed}: {time.time() - t0:.0f}s", flush=True)
    return recs


def aggregate(out: Path):
    recs = [pd.read_csv(p) for p in sorted(out.glob("cell_*.csv"))]
    if not recs:
        raise SystemExit(f"no cell_*.csv under {out}")
    df = pd.concat(recs, ignore_index=True)
    df.to_csv(out / "betasweep_raw.csv", index=False)

    regimes = [r for r in SYNTH_REGIMES if r in df["regime"].unique()]
    seeds = sorted(df["seed"].unique())
    fb = [f"fixed_beta_{b:.2f}" for b in BETA_GRID]

    # per (regime, config) seed-mean log loss
    piv = df.groupby(["regime", "config"])["ll"].mean().unstack("config")

    # per-regime fixed-beta oracle (best fixed beta in hindsight, on seed-mean)
    oracle = piv[fb].min(axis=1)
    oracle_beta = piv[fb].idxmin(axis=1).str.replace("fixed_beta_", "").astype(float)

    # globally best fixed beta = argmin over b of mean-across-regimes seed-mean loss
    glob_b_scores = piv[fb].mean(axis=0)
    glob_best = glob_b_scores.idxmin()
    glob_best_beta = float(glob_best.replace("fixed_beta_", ""))

    # seed-level excess vs the per-regime oracle beta (paired), for CIs
    def seed_excess(config_col):
        rows = []
        for rk in regimes:
            ob = f"fixed_beta_{oracle_beta[rk]:.2f}" if config_col == "ORACLE" else None
            for s in seeds:
                m = df[(df.regime == rk) & (df.seed == s)]
                if config_col == "ORACLE":
                    val = m[m.config == ob]["ll"]
                else:
                    val = m[m.config == config_col]["ll"]
                oref = piv.loc[rk, f"fixed_beta_{oracle_beta[rk]:.2f}"]
                if len(val):
                    rows.append(dict(regime=rk, seed=s, excess=float(val.iloc[0]) - oref))
        return pd.DataFrame(rows)

    compare = {"beta=0": "fixed_beta_0.00",
               f"global fixed beta={glob_best_beta:g}": glob_best,
               "ensemble3": "ensemble3",
               "AMG-TP (adaptive)": "amgtp_adaptive"}

    def boot_ci(vals, n=5000, seed=0):
        rng = np.random.default_rng(seed)
        v = np.asarray(vals, float)
        bs = [rng.choice(v, len(v), replace=True).mean() for _ in range(n)]
        return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    lines = ["# Package 1 -- comprehensive fixed-beta sweep", "",
             f"beta grid {BETA_GRID[0]}..{BETA_GRID[-1]} step 0.05, regimes {regimes}, "
             f"seeds {seeds} ({len(seeds)}). Context gate + 5-expert bank held fixed; "
             "`fixed_beta_b` = pi = (1-b) q + b m. Excess = log loss minus the per-regime "
             "best fixed beta chosen in hindsight (so the oracle row is 0 by construction).", "",
             "## Per-regime best fixed beta (hindsight) and its log loss", "",
             "| regime | oracle beta | oracle log loss | AMG-TP | AMG-TP excess | beta=0 excess | "
             f"global-beta({glob_best_beta:g}) excess | ensemble3 excess |",
             "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for rk in regimes:
        ob = oracle_beta[rk]
        ol = oracle[rk]
        amg = piv.loc[rk, "amgtp_adaptive"]
        b0 = piv.loc[rk, "fixed_beta_0.00"]
        gb = piv.loc[rk, glob_best]
        e3 = piv.loc[rk, "ensemble3"]
        lines.append(f"| {rk} | {ob:.2f} | {ol:.4f} | {amg:.4f} | {amg-ol:+.4f} | {b0-ol:+.4f} | "
                     f"{gb-ol:+.4f} | {e3-ol:+.4f} |")
    lines += ["", "## Headline: excess log loss vs the per-regime fixed-beta oracle",
              "Paired bootstrap 95% CI over seed-level excess (pooled across regimes for the "
              "mean row; the max is the worst single regime's seed-mean excess).", "",
              "| method | mean excess [95% CI] | worst-regime excess |", "|---|---|--:|"]
    for label, col in compare.items():
        se = seed_excess(col)
        mean, lo, hi = boot_ci(se["excess"])
        worst = se.groupby("regime")["excess"].mean().max()
        lines.append(f"| {label} | {mean:+.4f} [{lo:+.4f}, {hi:+.4f}] | {worst:+.4f} |")
    lines += ["", "## Interpretation",
              "- If AMG-TP's **worst-regime excess** is below the global-fixed-beta row's, "
              "adaptive persistence buys real robustness (its stated revised claim).",
              "- If AMG-TP's **mean excess** is ~0 it matches the per-regime oracle on average "
              "-- i.e. it is not merely interpolating between the two previously tested configs.",
              "- ensemble3 is the strongest simple robustness alternative; AMG-TP should be "
              "close to it at a fraction of the inference cost (one gate vs three)."]
    (out / "REPORT.md").write_text("\n".join(lines))
    piv.to_csv(out / "betasweep_regime_config.csv")
    print("\n".join(lines))
    print(f"\nwrote {out}/REPORT.md + betasweep_regime_config.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="amgtp_experiments/stage4_betasweep")
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
        rk, s = GRID_CELLS[args.cell]
        pd.DataFrame(run_cell(rk, s, args.n_jobs)).to_csv(out / f"cell_{rk}_seed{s}.csv", index=False)
        print(f"wrote {out}/cell_{rk}_seed{s}.csv")
        return
    for rk, s in GRID_CELLS:
        pd.DataFrame(run_cell(rk, s, args.n_jobs)).to_csv(out / f"cell_{rk}_seed{s}.csv", index=False)
    aggregate(out)


if __name__ == "__main__":
    main()
