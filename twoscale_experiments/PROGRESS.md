# Two-timescale progress log

Newest entry on top.

---

## 2026-09-04 — package built, tests green, battery submitted

New self-contained package `twoscale/` implementing
`CTR_Two_Timescale_Experiment_Plan.pdf`. No imports from the AMG-TP / M5b
code (user steer: "treat as a new project plan, do not mix up with the old
code").

### What is implemented
- **data.py**: Criteo (`sec_in_day = timestamp % 86400`, real second
  resolution) and Avazu (`sec_in_day = hour_of_day * 3600`) loaders, rows
  sorted by (day, sec_in_day).
- **splits.py**: 60/21/35-of-116 proportions scaled to the dataset. Criteo
  31 days -> train 0-15 / dev 16-21 / test 22-30. Avazu 10 days ->
  train 0-4 / dev 5-6 / test 7-9.
- **longterm.py**: fresh L2-logistic base models on rolling-3 / rolling-7 /
  expanding history (fit on days < d); adaptive mixture = exponential weights
  over the 3, driven by each candidate's discounted past day-loss.
- **calib.py**: causal within-day replay of `b <- Proj[-B,B](b - eta_k g_k)`
  with feedback maturation delay `Delta` (a label at within-day time tau
  enters `g` only at tau+Delta) and block/per-impression update; optional
  online Platt slope; oracle intercept (Newton, eq 10); prior-days time-of-day
  intercepts.
- **methods.py**: the full section-4 suite (expanding/rolling/equal/long_only/
  short_only/combined/time_of_day/oracle_intercept/online_platt).
- **metrics.py / diagnostics.py**: section 5 feasibility (daily oracle
  improvement, intraday residual runs, early/late), section 7 metrics,
  section 8 regret + captured gain.
- **twoscale_run.py**: one cell -> per_day_metrics / intraday_blocks /
  calib_traces / diag_* / ablations CSVs + summary.json with the section-10
  success-criteria scorecard. Ablations 1-7 (section 9) run inline.
- **twoscale_hpo.py**: section-6 grid (B, eta0, schedule, update granularity)
  scored on dev days only -> FROZEN.json, stability tie-break.
- **twoscale_tests.py**: 15 checks (block + per-impression causality,
  chronology placebo bites, zero-lr identity, projection, oracle argmin,
  determinism, mixture simplex/causality, split proportions). **All pass.**

### Smoke (Criteo 2% subsample, single seed)
Pipeline runs end to end in ~50s. Feasibility on dev days:
`mean daily oracle-intercept improvement 0.00036 log loss (0.06% relative)`,
lag-1 residual autocorr 0.08 — i.e. **very little within-day scalar-calibration
headroom on Criteo**, consistent with every prior real-data finding in this
repo (shallow 31-day drift). `combined` beat `long_only` (its own backbone,
9/9 days) but `short_only` (expanding + calibration) was best because the
adaptive mixture underperformed plain expanding at 2%. Real signal awaits the
full run.

### Running
- `sbatch twoscale_criteo.slurm` (array 0-2, full data) — SUBMITTED <fill jobid>
- `sbatch twoscale_avazu.slurm` (array 0-7, 20% subsamples) — SUBMITTED <fill jobid>
Then `twoscale_aggregate.py --stage twoscale_experiments/{criteo,avazu}`.

### Not yet done
- Downstream matched-budget autobidding eval (plan section 7.3 / 11 step 9) —
  gated on a positive main result, will reuse a fresh counterfactual replay.
- Feedback-delay sensitivity is in the inline ablations (Delta in {0,900,
  1800,3600}); a dedicated sweep only if the main result is positive.
