# AP-OPS — Adaptive-Persistence Online Platt Scaling: findings

Implementation of `final_experiments/AP_OPS_Minimal_Experiment_Plan.pdf`
(the "minimal" method-and-experiment note, Sept 2026). This document is
updated as the SLURM pipeline lands; see the **Status** section at the
bottom for what is done vs running.

## 1. What AP-OPS is

The completed DualTime-CTR experiment (`FINDINGS.md`) gave a sharp design
signal: full Online Platt Scaling (OPS) — a learned two-parameter map
`p = σ(a·logit(q) + b)`, updated online by projected gradient with a
**daily reset** — is the strongest calibrator on Criteo (0.606958 vs
0.607070 for DualTime-CTR) and close to the best on Avazu (0.387443 vs
0.387402 for AdaMoE). Avazu diagnostics hinted that carrying calibration
state across the day boundary might help early-day; Criteo prefers
resetting.

AP-OPS keeps the full two-parameter map and adds exactly one thing:
**adaptive temporal memory**. Three experts run in parallel on the
identical frozen cross-day probability `q_{d,i}` (the shared adaptive
roll3/roll7/expanding mixture, reused unchanged):

| expert | state across days | update | role |
|---|---|---|---|
| **R** reset anchor | reset `(a,b)=(1,0)` each day | current projected gradient — literally `twoscale.calib.replay_day` | protects the verified OPS behaviour |
| **S** short memory | carried continuously | discounted ONS, `h = 4 h` | tracks fast within-day / day-boundary shifts |
| **L** long memory | carried continuously | discounted ONS, `h = 16 h` | retains stable calibration across days |

Discounted Online Newton Step (per matured block `r`, `Δt` = block width):

```
γ      = 2^(-Δt / h)
g_r    = ∇_θ  L_r(θ_r)                         # gradient of mean block log loss
A_r    = γ A_{r-1} + g_r g_rᵀ + (1-γ) λ I
θ_{r+1}= Proj_Θ( θ_r − η_ons A_r⁻¹ g_r ),      Θ = {a∈[0.2,5], b∈[-0.25,0.25]}
```

The three expert probabilities are mixed `p^AP = Σ_k w_k p^(k)`; weights
start at `(0.50, 0.25, 0.25)` for `(R,S,L)`, persist across days, and are
updated by **delayed fixed-share** once a block's labels have all matured:

```
w̃_k ∝ w_k exp(−η_m L_{r,k});   w_k = (1−α) w̃_k + α/3
α   = 1 − 2^(−Δt / τ)                          # τ = switching half-life
```

Because expert R **is** the current OPS implementation, the case "all
weight on R" recovers the empirical anchor exactly (checked bit-for-bit,
see §4).

## 2. Code

`final_experiments/apops/` — `experts.py` (`replay_ons_stream`),
`aggregate.py` (`aggregate`, absolute-time maturation queue so a block
near midnight is scored at the right wall-clock moment), `method.py`
(`build_rows` → the plan's four rows). Runners:
`run_apops_hpo.py` / `run_apops_final.py` / `run_apops_rolling.py`.
Tests: `apops_tests.py`. SLURM: `apops_{hpo,final,rolling}_{criteo,avazu}.slurm`.

Nothing upstream of calibration changed: the shared adaptive cross-day
mixture and the locked OPS `(B, η0, schedule)` come straight from each
dataset's existing `final_experiments/<ds>/hpo/selected_configs.json`.

## 3. Protocol (plan section 3)

| phase | what |
|---|---|
| Development | established dev days, seeds 0/1/2. Tune **only** `ons_lam ∈ {1e-3,1e-2,1e-1,1}`, `ons_eta ∈ {0.25,1,4}`, `η_m ∈ {1,10,100}`, `τ ∈ {4,16,64,∞} h`. Select the single config with the lowest mean dev-day impression-weighted log loss across seeds; freeze into `apops_selected.json`. |
| Golden check | R alone must reproduce the saved headline OPS number within `5e-4`; anchor-only AP-OPS (weights pinned to `(1,0,0)`) must be prediction-equivalent to it. Hard-fails the run otherwise. |
| Fixed test | the four rows once on the existing locked stream (Criteo test days 22–30, Avazu 7–9), 3 seeds. |
| Rolling origin | the frozen rows at every established origin (Criteo 16–30, Avazu 5–9), 3 seeds — the primary temporal-adaptation evidence. |

**Four evaluated rows:** `current_ops` (golden reference), `ap_ops`,
`single_memory` (the better of S/L on dev, no meta mixture),
`no_slope` (AP-OPS with `a` fixed at 1 everywhere — is the slope DOF
essential?).

**Metrics:** primary = impression-weighted log loss and paired
calendar-day difference vs current OPS (seeds averaged within a day
first, then bootstrap / sign-test across days). Temporal = pre-feedback
loss, first-quarter-of-day loss, worst-day loss. Mechanism = expert-weight
paths, per-memory-scale block-dominance counts.

**Decision rule (plan section 4), pre-declared on dev:**
`δ_NI = 0.10 · |L_OPS,dev − L_base-q,dev|` (base-q = uncalibrated adaptive
mixture).

- **Improves** — paired daily CI vs OPS lies below 0 → advance AP-OPS, calibrate the claim to the interval.
- **Retains performance** — CI crosses 0 but upper end `< δ_NI`, weights adapt as intended → lead with adaptive memory + the stronger switching-comparator guarantee.
- **Failure** — CI upper end `≥ δ_NI`, or a systematic early-day / worst-day regression → keep current OPS, report AP-OPS only as an ablation.

## 4. Golden / leakage checks (`apops_tests.py`, 7/7 on a 3 % smoke)

1. anchor-only AP-OPS `(1,0,0)` == current OPS, bit-for-bit
2. `current_ops` row == the standalone `methods.ops_method` call the headline table used
3. flipping labels after time *t* leaves every prediction before *t − delay* unchanged; every earlier day byte-identical
4. meta weights nonnegative and sum to 1 at every update
5. persistent-expert `(a,b)` stay inside `[0.2,5] × [-0.25,0.25]`
6. identical inputs reproduce identical outputs

## 5. Results

### Criteo — dev tuning

_pending — `apops_hpo_criteo.slurm`, job 12513590_

### Criteo — fixed test (test days 22–30, 3 seeds)

_pending — `apops_final_criteo.slurm`_

### Criteo — rolling origin (origins 16–30, 3 seeds)

_pending — `apops_rolling_criteo.slurm`_

### Avazu — dev tuning

_pending — `apops_hpo_avazu.slurm`, job 12513591_

### Avazu — fixed test (test days 7–9, 3 seeds)

_pending — `apops_final_avazu.slurm`_

### Avazu — rolling origin (origins 5–9, 3 seeds)

_pending — `apops_rolling_avazu.slurm`_

### Decision

_pending_

## Status

| step | state |
|---|---|
| implementation + unit/leakage tests | **done** (commit `9b93394`, branch `apops-experiment`) |
| end-to-end smoke on 3 % Criteo | **done** — full pipeline runs, golden check + decision rule fire correctly |
| Criteo dev HPO | **running** — job 12513590 |
| Avazu dev HPO | **running** — job 12513591 |
| fixed test (both) | queued behind HPO |
| rolling origin (both) | queued behind fixed test |
| this document's Results section | filled as jobs land |

**Prior on the outcome.** Every line in this repo — AMG-TP, twoscale
(`combined ≈ long_only`), the capacity-ladder V5 (sub-materiality on
Criteo), and DualTime-CTR itself (OPS beats it on Criteo, no reproducible
edge on Avazu) — has found only shallow exploitable intraday / cross-day
calibration drift on these public datasets. The realistic expectation is
**"Retains performance"**: AP-OPS non-inferior to OPS, its value resting
on the adaptive-memory mechanism and the switching-comparator guarantee
(`S = 0` in the theory target immediately gives the OPS-anchor bound)
rather than a headline log-loss win. The rolling-origin weight paths are
designed to be informative either way — they show whether the aggregator
actually returns to R after a regime change and whether S/L ever earn
their weight.
