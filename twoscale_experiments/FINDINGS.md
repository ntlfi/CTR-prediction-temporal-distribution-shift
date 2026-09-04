# Two-timescale CTR forecasting — findings

`CTR_Two_Timescale_Experiment_Plan.pdf`, locked-test battery.
Criteo Attribution (full data, 3 seeds, 9 test days) and Avazu (20% subsamples,
8 seeds, 3 test days). Frozen calibrator config per dataset from
`twoscale_hpo.py` (dev days only).

## Headline

**The within-day online scalar calibration does not earn its place.** On
neither real dataset does the combined method beat *both* the long-only and
the short-only ablation — the complementarity hypothesis (H3) is not
supported.

| | Criteo | Avazu (10 days, thin) |
|---|---|---|
| long-term adaptation (H1) | **helps** — adaptive mixture −0.15% vs expanding, 3/3 seeds | **hurts** — mixture worse than expanding, 0/8 seeds (too few days to weight on loss) |
| within-day calibration (H2) | ~0 on top of the mixture (combined ≈ long_only, CI incl. 0, 0/3 seeds) | small but real on a fixed backbone (short_only −0.02% vs expanding, 8/8) |
| complementarity (H3) | **no** — combined ties long_only | **no** — short_only beats combined |
| plan §10 outcome | "Long-only ≈ Combined: limited exploitable within-day calibration drift" | "Short-only ≈/> Combined: adaptive long-term module adds little" |

## Locked-test impression-weighted log loss

**Criteo** (mean ± sd over 3 seeds):

```
online_platt      0.60700 ± 0.00007     <- marginally best, within noise of long_only
oracle_intercept  0.60707 ± 0.00005     <- non-deployable ceiling
time_of_day       0.60710 ± 0.00007
combined          0.60711 ± 0.00006
long_only         0.60718 ± 0.00006
equal_ensemble    0.60716 ± 0.00004
rolling_7         0.60729 ± 0.00005
rolling_3         0.60741 ± 0.00011
short_only        0.60788 ± 0.00001     <- expanding backbone + calib
expanding         0.60807 ± 0.00001
```

**Avazu** (mean ± sd over 8 seeds):

```
short_only        0.38777 ± 0.00045     <- best
equal_ensemble    0.38779 ± 0.00043
online_platt      0.38792 ± 0.00074
oracle_intercept  0.38795 ± 0.00042
expanding         0.38798 ± 0.00045
rolling_7         0.38802 ± 0.00049
combined          0.38834 ± 0.00058
long_only         0.38861 ± 0.00047     <- adaptive mixture underperforms here
time_of_day       0.38869 ± 0.00060
```

## Decisive comparisons

| comparison | Criteo (3 seeds) | Avazu (8 seeds) |
|---|---|---|
| combined − long_only | −0.00006, **0/3 CI<0**, sign-p 0.125 | −0.00023, 6/8 CI<0, sign-p 0.004 |
| combined − short_only | −0.00076, 3/3 CI<0 *(but short_only backbone = expanding, the worst method)* | **+0.00046**, 0/8 — combined worse |
| combined − time_of_day | +0.00003 — no gain over static seasonality | −0.00029, sign-p 0.004 |
| online_platt − combined | −0.00011, 3/3 CI<0 | −0.00040, 8/8 CI<0 |

## Why the within-day component is inert

1. **Feasibility (plan §5).** The dev-day best hindsight fixed intercept improves
   log loss by only **0.037%** (Criteo) / **0.023%** (Avazu). There is almost
   nothing for a scalar calibrator to capture. `combined` captures 83–86% of
   that Criteo ceiling and ~52% of Avazu's — a large fraction of nearly zero.

2. **Chronology placebo (plan §9.5).** Shuffling within-day impression order
   before calibrating changes the locked-test loss by **< 3e-5 on both
   datasets** — the calibrator is not exploiting arrival order, it is
   correcting a slowly-varying daily bias. Criteo's lag-1 intraday residual
   autocorrelation is 0.52 and Avazu's 0.80, but the *net* daily correction
   the intercept converges to is ≈ 0, so the structure is real yet unusable by
   a single scalar.

3. **Static seasonality explains it (plan §9.4).** On Criteo `time_of_day` — a
   fixed per-hour intercept learned from prior days — matches or slightly
   beats the online calibrator. Whatever the calibrator learns within a day is
   already knowable from the daily cycle.

4. **Feedback delay barely matters.** Δ ∈ {0, 900, 1800, 3600}s moves the
   locked-test loss by ≤ 1e-4 (Criteo) / ≤ 1e-5 (Avazu). H5 (causal
   robustness) holds trivially because the effect being probed is tiny.

5. **Causal signature is clean (plan §5.3).** `combined` matches the long-term
   model over the first third of each day and diverges only slightly later —
   no leakage, no early-day artifact. It just has little to add.

## What did work

- **online Platt scaling** (slope + intercept, plan eq 9) is the single
  best-scoring method on both datasets, with *negative* mean regret against
  the fixed-intercept oracle — but on Criteo it sits within one sd of
  `long_only`, so the practical gain is ~0.02%.
- **The long-term adaptive mixture** is a clear, reproducible win over plain
  expanding history on Criteo (−0.15%, 3/3 seeds), once its exponential-weights
  rate is tuned (η ≈ 150; the log-loss gaps between window candidates are
  O(0.01)). On Avazu's 10 days it has too little history to help.
- **Cost (plan §7.4):** the calibration update is 0.06–0.10 s per million
  impressions — negligible next to the ~40–110 s of base-model fitting.

## Verdict

Plan §10: **"If Long-only ≈ Combined, the dataset has limited exploitable
within-day calibration drift or labels arrive too late."** That is the Criteo
result. Avazu adds "the adaptive long-term module adds little" in a 10-day
regime. In neither case is the two-timescale framework supported as a new
algorithmic result on real data — consistent with this repo's standing finding
that real CTR datasets show only shallow temporal drift.

**Downstream autobidding (plan §11 step 9) was not run** — it is explicitly
gated on a positive prediction result.

### If revisited

- A dataset with genuine intraday non-stationarity (breaking-news publishers,
  live-event advertising) is the missing ingredient; the machinery
  (`twoscale/`) is ready for it — point `twoscale_run.py` at it.
- `online_platt` is the variant worth carrying forward; intercept-only leaves a
  little on the table.
- On short horizons (Avazu) skip the mixture and calibrate a fixed
  expanding/equal-ensemble backbone directly.
