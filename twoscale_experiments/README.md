# Two-timescale CTR forecasting

Self-contained implementation of `CTR_Two_Timescale_Experiment_Plan.pdf`:
a long-term cross-day predictor plus a lightweight **within-day online scalar
calibration** correction, updated causally from labels observed earlier in
the current day.

This project shares **no code** with the repo's earlier adaptive-training /
AMG-TP experiments — only the raw Criteo / Avazu data files on disk. All
logic lives in the `twoscale/` package.

## Layout

```
twoscale/
  data.py         Criteo + Avazu loaders with an explicit within-day time axis
  splits.py       scaled 52/18/30 chronological split (plan section 3)
  longterm.py     rolling / expanding base models + a fresh exp-weights adaptive mixture
  calib.py        online scalar intercept (+ optional Platt slope), causal replay,
                  oracle intercept, time-of-day intercepts
  methods.py      the baseline / ablation suite of plan section 4
  metrics.py      log loss (imp-weighted + daily-mean), Brier, ECE, days-won,
                  intraday-block residuals, early/mid/late, paired bootstrap CI,
                  regret + captured-gain (plan section 8)
  diagnostics.py  section 5 feasibility diagnostics (dev days only)

twoscale_run.py        one (dataset, seed) locked-test cell -> CSV + summary.json
twoscale_hpo.py        section 6 dev-only grid search -> FROZEN.json
twoscale_aggregate.py  cells -> tables/ figures/ REPORT.md
twoscale_tests.py      leakage / causality / identity / reproducibility (15 checks)
twoscale_{criteo,avazu}.slurm
```

## Run

```bash
.venv/bin/python twoscale_tests.py

# one cell locally on a subsample
.venv/bin/python twoscale_hpo.py --source criteo --sample-frac 0.1 --seed 0 --out twoscale_experiments/dev/_hpo
.venv/bin/python twoscale_run.py --source criteo --sample-frac 0.1 --seed 0 \
    --config twoscale_experiments/dev/_hpo/FROZEN.json --out twoscale_experiments/dev

# full battery
sbatch twoscale_criteo.slurm      # 3 seeds, full data
sbatch twoscale_avazu.slurm       # 8 seeds, disjoint 20% subsamples
.venv/bin/python twoscale_aggregate.py --stage twoscale_experiments/criteo
```

## Method suite (plan section 4)

| method | long-term | within-day | tests |
|---|---|---|---|
| `expanding` / `rolling_3` / `rolling_7` | fixed window | none | H1 references |
| `equal_ensemble` | uniform avg of the 3 | none | multi-timescale w/o learned adaptation |
| `long_only` | exp-weights adaptive mixture | none | value of cross-day adaptation |
| `short_only` | expanding | online intercept | value of within-day calibration |
| `combined` | adaptive mixture | online intercept | **complementarity (H3)** |
| `time_of_day` | adaptive mixture | fixed hourly intercept from prior days | seasonality control |
| `oracle_intercept` | adaptive mixture | best hindsight daily b* | non-deployable ceiling |
| `online_platt` | adaptive mixture | online slope + intercept | calibration-complexity control |

Decisive comparisons: `combined` vs `long_only` and `combined` vs `short_only`.
