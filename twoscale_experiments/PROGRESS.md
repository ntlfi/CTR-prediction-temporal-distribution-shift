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

### Smoke (Criteo 15% subsample, single seed, frozen-config path)
Full pipeline (HPO -> run -> aggregate) end to end in ~140s. HPO picked
mixture eta=150 halflife=10, calibrator B=0.25 eta0=0.3 const block-900s.

Locked-test imp-weighted log loss (9 test days): online_platt 0.60776 <
oracle_intercept 0.60778 < **combined 0.60785** < time_of_day 0.60801 <
long_only 0.60812 < equal 0.60818 < short_only 0.60837 < expanding 0.60878.
- combined vs long_only: Delta -0.00028, CI [-0.00051,-0.00008], **7/9 days, CI<0**
- combined vs short_only: Delta -0.00052, CI [-0.00066,-0.00040], **9/9, CI<0**
- long_only now beats expanding (the higher mixture eta fixed H1)
- combined ~ oracle_intercept: captures essentially all the scalar headroom
- **chronology placebo: shuffled ~= real (Delta -3.6e-5)** -> the gain is
  correcting a slowly-varying global bias, NOT exploiting within-day order;
  Criteo has ~no intraday chronological signal. Feasibility agrees: dev mean
  oracle-intercept improvement 0.00045 (0.07% rel).
- delay 0/900/1800/3600s: monotone tiny degradation (~5e-5 per step).
- calib cost: 0.10 s per million impressions (section 7.4, negligible).

So at 15% the two-timescale ordering holds directionally with clean day-block
CIs, but the absolute effect is ~0.03-0.05% relative and the placebo says it
is bias-correction not chronology. Full-data run is the real test of whether
the CI stays below zero and whether short_only alone captures it.

Bug fixes found in smoke: per-impression ablation needs its own decaying LR
(inv_sqrt, eta0=0.03) or a single noisy label swings b across [-B,B];
captured-gain denom floor raised to 1e-4 (Criteo's oracle gap is below that
most days, so captured-gain is legitimately mostly undefined here).

### RESULTS (2026-09-04, jobs 12482301 Criteo 3 seeds / 12482302 Avazu 8 seeds, all COMPLETED)

Full write-up: `FINDINGS.md`. Stage REPORTs: `criteo/REPORT.md`, `avazu/REPORT.md`.

**Headline: the within-day online scalar calibration does not earn its place.**
On neither dataset does `combined` beat *both* `long_only` and `short_only`
(H3 / complementarity — not supported).

- **Criteo (full data):** `combined` ≈ `long_only` (Δ −0.00006, 0/3 seeds
  CI<0, sign-p 0.125). The *long-term adaptive mixture* is the useful
  component (−0.15% vs plain expanding, 3/3 seeds, once its exp-weights η is
  tuned to ~150). Plan §10 outcome = **"Long-only ≈ Combined: limited
  exploitable within-day calibration drift."**
- **Avazu (10 days, thin):** the mixture *underperforms* expanding (0/8
  seeds) — too few days. `short_only` / `equal_ensemble` are best; `combined`
  loses to `short_only` (Δ +0.00046, 0/8). Within-day calibration helps a
  fixed backbone a little (short_only −0.02% vs expanding, 8/8) and beats
  `time_of_day` (8/8) — real but small.
- **Chronology placebo ≈ real on both** (Δ < 3e-5): the small calibration
  effect is slow-bias / seasonality correction, NOT exploitation of
  within-day arrival order. On Criteo `time_of_day` (fixed hourly intercepts)
  matches the online calibrator.
- **Feedback delay** 0→3600s: ≤ 1e-4 change. **online_platt** (slope+intercept)
  is marginally best-scoring on both, negative regret vs the intercept oracle,
  but within 1 sd of `long_only` on Criteo.
- Feasibility was right: dev oracle-intercept headroom 0.037% (Criteo) /
  0.023% (Avazu). Causal signature clean (calibrated ≈ long-term early in the
  day). Calib cost 0.06–0.10 s / M impressions.

**Downstream autobidding (step 9): NOT run** — gated on a positive prediction
result (plan §11), which neither dataset provides.

Committed: branch `twoscale-experiment`, commits 8af5211, 23a1806, <results commit>.
Not on main, not pushed (user to decide).

### Not yet done
- Downstream matched-budget autobidding eval (plan section 7.3 / 11 step 9) —
  gated on a positive main result, will reuse a fresh counterfactual replay.
- Feedback-delay sensitivity is in the inline ablations (Delta in {0,900,
  1800,3600}); a dedicated sweep only if the main result is positive.
