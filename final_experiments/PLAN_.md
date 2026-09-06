# Experiment Plan: Generalized Online Scaling Variants

## 1. Goal

Test whether a generalized form of Online Platt Scaling can improve the current best calibration result, or retain its performance while providing a cleaner adaptive-regret guarantee.

The main candidate is **Adaptive-Persistence Online Platt Scaling (AP-OPS)**: a small causal bank of full slope-and-intercept online calibrators with different cross-day memory scales, combined by a delayed-feedback fixed-share aggregator. The exact current OPS implementation remains an anchor expert.

This plan is designed to answer four questions:

1. Is the learned slope \(a\), rather than intercept-only correction, essential?
2. Does ONS match or improve the current projected-gradient implementation?
3. Does retaining information across day boundaries improve early-day performance, especially on Avazu?
4. Can an adaptive mixture of memory scales match the current OPS everywhere while improving under temporal shifts?

## 2. Frozen experimental context

All variants must use the same upstream predictions and chronology as the existing final experiments.

| Item | Criteo | Avazu |
|---|---:|---:|
| Calibration block duration | 900 s | 3600 s |
| Label delay | 1800 s | 1800 s |
| Seeds | 0, 1, 2 | 0, 1, 2 |
| Primary metric | Impression-weighted log loss | Impression-weighted log loss |

Additional controls:

- Use the existing chronological train/dev/test split and exclusions in `final_experiments`.
- Train the cross-day predictor once per seed and cache its base probability \(q_i\). Every calibrator must receive the identical \(q_i\), labels, timestamps, and block boundaries.
- Keep the selected cross-day strategy fixed: the existing adaptive choice over roll3, roll7, and expanding-history predictions.
- Tune and select variants on development data only.
- Freeze the selected configuration, code commit, and configuration hash before the final rerun.
- Make predictions before incorporating any label whose delay has not matured.
- Average losses over impressions, not blocks, for the headline dataset-level metric.
- Preserve the existing probability clipping during the empirical comparison: \(\epsilon=10^{-5}\).
- Use full slope-and-intercept scaling unless a row is explicitly an intercept-only ablation.

Because the existing fixed-test results informed this plan, the strongest confirmation should come from the rolling-origin evaluation and, if feasible, an untouched final time slice or additional dataset. The fixed test remains useful as a directly comparable ablation benchmark.

## 3. Common prediction map

For base probability \(q_i\),

\[
z_i = \operatorname{logit}\!\left(\operatorname{clip}(q_i,\epsilon,1-\epsilon)\right),
\]

and the full online scaling map is

\[
\hat p_i = \sigma(a_t z_i+b_t), \qquad \theta_t=(a_t,b_t).
\]

Use the compact parameter set

\[
a_t\in[0.2,5.0], \qquad b_t\in[-0.25,0.25],
\]

for the first comparison so that the new variants differ from the current method only in their update and memory strategy. Any later change to these bounds is a separate dev-only ablation.

For block \(r\), define the mean log loss

\[
L_r(\theta)=
\frac{1}{n_r}\sum_{i\in r}
-\left[y_i\log \hat p_i+(1-y_i)\log(1-\hat p_i)\right].
\]

Updates occur only when all labels used in \(L_r\) have matured under the configured delay.

## 4. Methods to test

### 4.1 Required controls and ablations

| ID | Optimizer | Day-start state | Purpose |
|---|---|---|---|
| RAW | None | None | Fixed cross-day probability with no within-day calibration |
| OPS-INT-R | Current projected gradient | \(a=1,b=0\); reset daily; hold \(a=1\) | Measure the value of intercept-only correction |
| OPS-OGD-R | Current projected gradient | \(a=1,b=0\); reset daily | Exact current OPS and golden-reference anchor |
| OPS-OGD-P\(\rho\) | Current projected gradient | Partial parameter carryover | Isolate persistence without changing optimizer |
| OPS-ONS-R | Online Newton Step | \(a=1,b=0\); reset daily | Test a theory-friendly second-order update |
| OPS-ONS-P\(\rho\) | Online Newton Step | Partial parameter carryover | Test ONS with cross-day parameter persistence |
| OPS-DONS-\(h\) | Discounted ONS | Continuous discounted state | Test several physical memory scales |
| AP-OPS | Delayed fixed-share aggregation | Bank containing the current anchor and ONS variants | Main proposed generalized method |
| OBS-ONS-R | Online Beta Scaling with ONS | Reset daily | Secondary literature control for a richer monotone map |

The exact current implementation, OPS-OGD-R, must use its locked settings:

- \(B=0.25\);
- \(\eta_0=0.3\);
- constant learning rate;
- \(a\in[0.2,5.0]\);
- daily reset to \((1,0)\);
- the existing block and delay logic.

### 4.2 Partial carryover

Let \(\theta_{\mathrm{id}}=(1,0)\). For day \(d\), initialize a persistence expert by

\[
\theta_{d,1}^{(\rho)}
=
\theta_{\mathrm{id}}
+
\rho\left(
\theta_{d-1,\mathrm{end}}^{(\rho)}
-
\theta_{\mathrm{id}}
\right).
\]

Test

\[
\rho\in\{0.5,0.9,1.0\}.
\]

The reset case \(\rho=0\) is already represented by OPS-OGD-R or OPS-ONS-R. Reset the ONS curvature matrix at the start of each day in this experiment so that it isolates parameter carryover.

### 4.3 Discounted ONS

Discounted ONS carries both the parameter and second-order information through the chronological stream while gradually forgetting old gradients. With block duration \(\Delta t\), convert a physical half-life \(h\) to

\[
\gamma(h)=2^{-\Delta t/h}.
\]

Test

\[
h\in\{1,4,16,\infty\}\text{ hours}.
\]

A stable implementation is

\[
A_r =
\gamma A_{r-1}
+
g_r g_r^\top
+
(1-\gamma)\lambda I,
\]

\[
\theta_{r+1}
=
\Pi_{\Theta}^{A_r}
\left(
\theta_r-\eta_{\mathrm{ons}}A_r^{-1}g_r
\right),
\qquad
g_r=\nabla_\theta L_r(\theta_r).
\]

For \(h=\infty\), use \(\gamma=1\). Add standard numerical damping before the \(2\times2\) solve and record every projection event.

### 4.4 Adaptive-Persistence OPS

AP-OPS combines expert probabilities, rather than averaging their parameters:

\[
\hat p_{i,r}^{\mathrm{AP}}
=
\sum_{k=1}^{K}w_{r,k}\hat p_{i,r}^{(k)}.
\]

The initial expert bank is:

1. OPS-OGD-R, the exact current method and anchor;
2. OPS-ONS-R;
3. the dev-selected OPS-ONS-P expert;
4. the dev-selected short-memory DONS expert;
5. the dev-selected long-memory DONS expert.

Do not add multiple nearly identical experts merely because they were evaluated. Select the two DONS half-lives on dev before constructing the final bank.

Use an anchor-biased prior:

\[
w_{1,\mathrm{anchor}}=0.5,
\]

with the remaining probability divided uniformly among the other experts. This gives the current method explicit protection while still allowing adaptation.

When block \(r\)'s labels mature, update provisional weights using the experts' mean block losses:

\[
\widetilde w_{r+1,k}
\propto
w_{r,k}\exp(-\eta_m L_{r,k}),
\]

then apply fixed share:

\[
w_{r+1,k}
=
(1-\alpha)\widetilde w_{r+1,k}
+
\frac{\alpha}{K}.
\]

No weight may depend on an unmatured label. The predictions already emitted for a block are immutable.

Tune the meta learning rate on dev using

\[
\eta_m\in\{1,10,100\}.
\]

Express the fixed-share rate through a physical switching half-life,

\[
\alpha(\tau)=1-2^{-\Delta t/\tau},
\]

and test

\[
\tau\in\{4,16,64,\infty\}\text{ hours},
\]

where \(\tau=\infty\) means \(\alpha=0\).

If the anchor-only bank does not reproduce OPS-OGD-R bit-for-bit, stop before running AP-OPS.

## 5. ONS implementation and tuning

The ONS implementation should follow the standard projected second-order update on the compact two-dimensional parameter set. Keep the implementation shared across OPS-ONS, DONS, and OBS wherever possible.

Use a deliberately small dev grid:

| Parameter | Values |
|---|---|
| Initial curvature \(\lambda\) | \(10^{-3},10^{-2},10^{-1},1\) |
| ONS scale multiplier | \(0.25,1,4\) |
| Parameter bounds | Fixed at current bounds for the primary comparison |
| Probability clipping | Fixed at \(10^{-5}\) |

Select one ONS configuration per dataset using mean dev log loss across the three seeds. Break ties by, in order:

1. lower worst-day loss;
2. lower early-day loss;
3. fewer projection events;
4. the simpler/default scale.

Do not tune separately on individual test days.

## 6. Staged experiment sequence

### Stage 0: Cache and manifest the shared stream

For every dataset and seed, save:

- impression identifier or stable row index;
- timestamp and day;
- block identifier;
- label-availability timestamp;
- label;
- base probability \(q_i\);
- the upstream model/configuration hash.

Verify that every method reads the same ordered cache.

### Stage 1: Correctness and exact reproduction

Run a small chronological slice and require:

- OPS-OGD-R matches the current saved predictions and aggregate loss within numerical tolerance;
- the anchor-only AP-OPS bank matches OPS-OGD-R exactly;
- changing future labels cannot affect earlier predictions, parameters, or weights;
- changing labels that have not yet matured cannot affect the current update;
- all probabilities are finite and in \([\epsilon,1-\epsilon]\);
- all parameters remain within bounds;
- expert weights are nonnegative and sum to one;
- rerunning a seed is deterministic.

Then reproduce the locked full-data OPS-OGD-R number before comparing new variants.

### Stage 2: Single-mechanism development screening

Run the following on dev for all three seeds:

1. OPS-INT-R;
2. OPS-OGD-R;
3. OPS-OGD-P\(\rho\);
4. OPS-ONS-R;
5. OPS-ONS-P\(\rho\);
6. OPS-DONS-\(h\);
7. OBS-ONS-R.

This stage identifies whether performance changes come from:

- slope flexibility;
- optimizer choice;
- parameter carryover;
- curvature carryover/forgetting;
- a richer scaling family.

Do not construct AP-OPS until this stage is complete.

### Stage 3: Construct and select AP-OPS on dev

Build the five-expert bank specified above, using only the Stage 2 dev ranking. Tune \(\eta_m\) and \(\tau\) on dev.

Required ablations:

- uniform prior versus anchor-biased prior;
- no fixed share versus fixed share;
- anchor plus best single new expert;
- full selected bank;
- parameter averaging versus probability averaging, reported only as a diagnostic.

Freeze one AP-OPS configuration per dataset. Also freeze a single shared configuration across both datasets; this is preferred for the main paper result if it is non-inferior to the per-dataset choices.

### Stage 4: Locked fixed-test evaluation

Evaluate every required control plus the frozen AP-OPS configuration exactly once on the fixed test stream.

The main comparison is AP-OPS versus OPS-OGD-R. Other variants explain the result but should not be searched again after test inspection.

### Stage 5: Rolling-origin evaluation

Run the frozen methods at every valid chronological origin. At each origin:

- fit or update the upstream method using only the allowed history;
- begin calibrator state according to the method definition;
- preserve the exact label delay;
- report both the full evaluation day and the early-day windows.

Rolling-origin results provide the main evidence about adaptation across changing regimes.

### Stage 6: Fresh confirmation

If data permits, reserve the latest untouched days or use an additional CTR dataset. Apply the already frozen shared AP-OPS configuration with no retuning. This is the cleanest confirmatory result for the final paper.

## 7. Metrics and diagnostics

### 7.1 Primary results

Report:

1. impression-weighted log loss over the complete evaluation stream;
2. paired daily log-loss difference against OPS-OGD-R;
3. paired daily difference against Expanding;
4. win/loss/tie counts by day.

For paired daily inference, first average the three seeds within each calendar day. Calendar day, not impression or seed, is the main resampling unit.

Report a 95% paired bootstrap interval over days. For short streams, also show every daily difference and an exact paired sign/randomization result; do not rely only on an asymptotic standard error.

### 7.2 Shift-sensitive diagnostics

Report:

- loss before the first delayed feedback becomes available each day;
- loss in the first quarter of each day;
- remaining-day loss;
- worst-day loss;
- 90th percentile daily loss;
- performance around the largest base-rate changes;
- parameter paths \(a_t,b_t\);
- DONS effective memory;
- AP-OPS expert weights over time;
- number of bound projections;
- runtime and peak memory.

These diagnostics directly test whether cross-day persistence helps early on and whether the aggregator returns to the reset expert after a regime change.

### 7.3 Secondary calibration metrics

Report Brier score and one reliability summary such as adaptive-bin ECE. These are secondary; variant selection remains based on log loss.

## 8. Selection and decision rules

Before the final rerun, define a dataset-specific non-inferiority margin on dev:

\[
\delta_{\mathrm{NI}}
=
0.10\times
\left|
L_{\mathrm{OPS\text{-}OGD\text{-}R,dev}
-
L_{\mathrm{RAW,dev}}
\right|.
\]

Record the numerical margins in the frozen run manifest.

AP-OPS advances as the main method if all of the following hold:

1. Its test point estimate is no worse than OPS-OGD-R by more than \(\delta_{\mathrm{NI}}\) on either dataset.
2. The upper end of the paired 95% interval for AP-OPS minus OPS-OGD-R is below \(\delta_{\mathrm{NI}}\) on both datasets.
3. It improves the point estimate on at least one dataset.
4. It does not introduce a material worst-day or early-day regression.
5. Its result is stable across seeds and rolling origins.

Use stronger language only when supported:

- **Improves:** the paired interval for AP-OPS minus OPS-OGD-R lies below zero.
- **Retains performance:** the interval crosses zero but its upper end is below \(\delta_{\mathrm{NI}}\).
- **Inconclusive:** the interval includes the non-inferiority margin.
- **Worse:** the point estimate and interval show a practically meaningful regression.

If AP-OPS is non-inferior but not significantly better, its paper value rests on adaptive behavior and the stronger comparator guarantee. If it fails non-inferiority, retain OPS-OGD-R as the empirical method and report the generalizations as ablations rather than promoting a new name.

## 9. Theory targets

The theoretical analysis should use block rounds and the same delayed-information filtration as the implementation.

### 9.1 Base calibrator

On a compact \(\Theta\), clipped base probabilities, and binary log loss:

- establish bounded gradients and exp-concavity for the two-parameter Platt family;
- give the standard ONS static-regret bound against the best fixed \((a,b)\);
- state explicitly how constants depend on clipping and parameter bounds;
- show that \(a>0\) preserves ranking monotonicity.

### 9.2 Adaptive memory

For the generalized method, target a comparator result of the form

\[
R_T
\le
R_T^{(k)}
+
R_T^{\mathrm{meta}}
+
R_T^{\mathrm{delay}},
\]

for every expert \(k\), including the exact current OPS anchor.

The preferred result is one of:

1. strongly adaptive interval regret against the best calibrator on every interval;
2. switching regret against a piecewise-stationary sequence with \(S\) changes;
3. a path-length dynamic-regret bound for a drifting comparator.

For the finite AP-OPS bank, prove a fixed-share expert bound that scales with \(\log K\), the number of switches, and the feedback delay. State the bound for the actual mean-block-loss update used in code.

A useful minimum guarantee is: AP-OPS competes with the current OPS anchor up to a sublinear/additive meta-regret term, while also competing with persistent calibrators when they are better.

### 9.3 Theory-to-code checks

The proof and implementation must agree on:

- whether loss is summed or averaged within a block;
- the exact delay convention;
- projection geometry;
- probability clipping;
- parameter bounds;
- curvature discounting;
- day-boundary initialization;
- fixed-share timing.

## 10. Required statistical tables and figures

### Main tables

1. Fixed-test log loss for RAW, OPS-OGD-R, OPS-ONS-R, best persistence variant, and AP-OPS.
2. Rolling-origin mean, worst day, win/loss/tie count, and paired interval.
3. Early-day versus remaining-day log loss.
4. Runtime and memory overhead.

### Ablation table

Include:

- intercept-only versus full \((a,b)\);
- OGD versus ONS under daily reset;
- reset versus parameter carryover under the same optimizer;
- parameter carryover versus discounted curvature;
- best single expert versus AP-OPS;
- no-share versus fixed-share aggregation;
- Platt scaling versus OBS control.

### Figures

1. Daily paired log-loss differences relative to OPS-OGD-R.
2. A representative sequence of \(a_t,b_t\) and base rate.
3. AP-OPS expert weights across the same sequence.
4. Early-day loss by persistence mode.

Choose representative days by a rule fixed before plotting, such as largest absolute day-to-day base-rate change, rather than visual appeal.

## 11. Output artifacts

Place implementation and outputs under `final_experiments/ops_variants/`:

- `config/base_stream.yaml`
- `config/ons_grid.yaml`
- `config/ap_ops_grid.yaml`
- `methods.py`
- `run_dev.py`
- `run_fixed_test.py`
- `run_rolling.py`
- `tests/test_causality.py`
- `tests/test_anchor_equivalence.py`
- `results/manifests/`
- `results/dev/`
- `results/fixed_test/`
- `results/rolling/`
- `OPS_VARIANTS_FINDINGS.md`
- `OPS_VARIANTS_DAY_LEVEL_STATS.md`

Each run manifest must contain:

- git commit;
- dataset fingerprint;
- exact chronological ranges;
- seed;
- all hyperparameters;
- block and delay definitions;
- upstream prediction hash;
- command line;
- start/end time;
- software environment;
- output checksums.

## 12. Execution priority

Run in this order:

1. Cache the shared base predictions.
2. Add causality and anchor-equivalence tests.
3. Reproduce OPS-OGD-R.
4. Run the slope ablation.
5. Compare reset OGD with reset ONS.
6. Test parameter carryover.
7. Test discounted ONS half-lives.
8. Select the compact expert bank.
9. Tune and freeze AP-OPS.
10. Run fixed-test and rolling-origin evaluation.
11. Complete statistical analysis.
12. Attempt untouched-data confirmation.
13. Write the theorem against the exact implemented algorithm.

Most calibration variants should be replayable from cached base predictions, so screening should not require retraining the CTR model.

## 13. Paper-facing outcome

If the decision rules pass, frame the method as:

> We introduce Adaptive-Persistence Online Platt Scaling, a delayed-feedback calibration strategy that combines reset and persistent full Platt calibrators across multiple memory scales. It preserves the strong empirical anchor of online scaling while adapting its effective memory to temporal shifts.

Introduce standard OPS once in the background/related-work discussion, cite it there, and then use the concise names OPS for the baseline and AP-OPS for the extension. The empirical section should focus on what the adaptive extension changes and what the controlled ablations establish.

If AP-OPS only retains performance, the contribution is still meaningful if the adaptive/switching guarantee is materially stronger and the rolling analysis verifies the intended behavior. If it improves, lead with the empirical improvement and use the theorem as supporting structure.
