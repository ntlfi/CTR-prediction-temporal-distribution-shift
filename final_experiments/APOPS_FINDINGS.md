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

All numbers are the frozen-config runs — HPO on dev only (jobs
12513590 Criteo / 12513591 Avazu), fixed test (12514073 / 12514074),
rolling origin (12514084 / 12514085). Golden check passed on every run
(`current_ops` reproduced the saved headline OPS number to `< 5e-4`;
anchor-only AP-OPS bit-identical to it). Frozen configs:

| | Criteo | Avazu |
|---|---|---|
| ONS `λ`, `η_ons` | 0.1, 0.25 | 1.0, 4.0 |
| meta `η_m`, switching half-life `τ` | 100, 4 h | 100, 4 h |
| single-memory pick (on dev) | **S** (4 h) | **L** (16 h) |
| `δ_NI` | 3.25e-5 | 1.29e-5 |

Both datasets picked the most aggressive meta settings on dev (`η_m = 100`,
shortest `τ`).

### Headline — impression-weighted log loss (mean over 3 seeds)

**Criteo** (test days 22–30):

| row | log loss | Δ vs OPS (day-level 95 % CI) | days won | verdict |
|---|---|---|---|---|
| Current OPS | 0.606958 | — | — | — |
| **AP-OPS** | **0.606834** | −0.000125 [−0.000149, −0.000101] | 9/9 | **Improves** |
| Single-memory (S) | 0.606819 | −0.000139 [−0.000198, −0.000082] | 9/9 | Improves |
| No-slope | 0.606990 | +0.000029 [−0.000017, +0.000081] | 4/9 | worse point estimate, CI crosses 0 |

**Avazu** (test days 7–9):

| row | log loss | Δ vs OPS (day-level 95 % CI) | days won | verdict |
|---|---|---|---|---|
| Current OPS | 0.387443 | — | — | — |
| **AP-OPS** | **0.387171** | −0.000270 [−0.000306, −0.000238] | 3/3 | **Improves** (D = 3, sign p at floor) |
| Single-memory (L) | 0.387269 | −0.000176 [−0.000296, −0.000135] | 3/3 | Improves |
| No-slope | 0.387736 | +0.000277 [+0.000174, +0.000437] | 0/3 | clearly worse than OPS |

### Rolling origin — the primary temporal-adaptation evidence

Day-level inference: seeds averaged within each origin day, then bootstrap
/ sign-test across origins.

**Criteo** (15 origins, days 16–30):

| row | Δ vs OPS (day-wt) | 95 % CI (day bootstrap) | CI excl 0 | origins won | sign p |
|---|---|---|---|---|---|
| **AP-OPS** | −0.000119 | [−0.000145, −0.000092] | yes | 15/15 | 6.1e-5 |
| Single-memory (S) | −0.000127 | [−0.000163, −0.000092] | yes | 15/15 | 6.1e-5 |
| No-slope | +0.000037 | [−0.000006, +0.000081] | no | 6/15 | 0.61 |

**Avazu** (5 origins, days 5–9):

| row | Δ vs OPS (day-wt) | 95 % CI (day bootstrap) | CI excl 0 | origins won | sign p |
|---|---|---|---|---|---|
| **AP-OPS** | −0.000281 | [−0.000356, −0.000217] | yes | 5/5 | 0.062 (D=5 floor) |
| Single-memory (L) | −0.000227 | [−0.000337, −0.000146] | yes | 5/5 | 0.062 |
| No-slope | +0.000131 | [−0.000039, +0.000302] | no | 2/5 | 1.00 |

### Temporal / shift-sensitive metrics (fixed test, pooled log loss)

AP-OPS is at least as good as OPS in **every** sub-window on both
datasets — there is no early-day or worst-day regression:

| | pre-feedback | first quarter | worst day |
|---|---|---|---|
| Criteo OPS → AP-OPS | 0.619245 → 0.619172 | 0.614087 → 0.613880 | 0.613835 → 0.613787 |
| Avazu OPS → AP-OPS | 0.408304 → 0.407788 | 0.385674 → 0.385359 | 0.406003 → 0.405769 |

The largest AP-OPS gains on Avazu are exactly in the pre-feedback window
(−0.0005), which is the regime the plan's motivation predicted persistence
would help — cross-day calibration state carries information into the
start of a new day before that day's own feedback matures.

### Mechanism (AP-OPS expert weights, seed 0)

- **Criteo:** final weights R/S/L ≈ 0.28 / 0.36 / 0.36; the aggregator
  *downweights the reset anchor* and the weight vector ranges over
  [0.06, 0.87] across 1432 meta updates — it genuinely moves, and it
  learns that the persistent experts beat the daily-reset one on Criteo.
- **Avazu:** weights stay near uniform, [0.25, 0.47] over 118 updates
  (only 3 test days × 24 blocks) — too few blocks for the meta layer to
  concentrate; the gain here is carried by the persistent experts
  themselves, not by weight movement.
- Persistent-expert `(a, b)` end near `(1.05, 0.03)` with `~0` projection
  events — a mild, stable correction that a daily reset throws away every
  midnight.

## 6. Decision (plan section 4)

**Both datasets: "Improves."** AP-OPS is non-inferior to OPS (trivially —
it is strictly better), improves the point estimate on both, the paired
daily CI vs OPS lies **entirely below zero** on both the fixed test and
the rolling-origin evaluation, the result is unanimous across seeds and
origins (Criteo 15/15, Avazu 5/5), and there is no early-day or worst-day
regression. Advance AP-OPS; calibrate the claim to the interval
(≈ −0.00012 log loss on Criteo, ≈ −0.00028 on Avazu).

**Two mechanism findings from the ablations:**

1. **The learned slope is essential.** The no-slope variant (`a ≡ 1`,
   update only `b`) is *worse than plain OPS* on both datasets
   (Criteo +0.00003, Avazu +0.00028) — fixing the slope removes an
   important degree of correction, exactly as section 1 anticipated.
2. **The win is persistence, not the memory-scale mixture.** The
   single-memory ablation (one persistent full-Platt expert, the better
   of S/L chosen on dev, no meta layer) matches AP-OPS on Criteo
   (−0.00013 vs −0.00012) and is close on Avazu (−0.00018 vs −0.00027).
   Carrying full slope-and-intercept calibration state across the day
   boundary via discounted ONS is what beats the daily-reset anchor; the
   fixed-share meta layer over {reset, short, long} adds **robustness**
   (no need to commit to S vs L on dev — the dev pick actually differs by
   dataset) and the switching-comparator guarantee, and a small extra
   gain on Avazu, rather than a large accuracy improvement.

This is the first method in the whole AMG-TP / twoscale / withinday /
DualTime-CTR line to **beat OPS reproducibly on both public datasets** at
the day level. The effect is small in absolute terms (sub-0.0003 log
loss) but directionally unanimous, present on the rolling-origin
evaluation, and mechanistically clean.

## 7. Additional experiments (2026-09-06 spec) — fully nested rolling origin

The fixed-test days above were already inspected by the DualTime-CTR
experiment, and §5's rolling-origin runner used the dev-frozen config
(not re-selected per origin). The additional-experiments spec closes both
gaps and sharpens the mechanism question into three decision rules.

**Implementation corrections applied (`apops/experts.py`, `apops/aggregate.py`):**

1. **Absolute-time feedback queue for the persistent experts.** The
   day-local maturation pointer dropped labels from the last ~2 blocks of
   each day (they mature after midnight). Now `replay_ons_stream` does a
   single chronological pass with a `d*86400 + (k+1)*block_sec + delay`
   maturation clock, so a block near a day boundary updates the persistent
   expert on the *following* day. Regression test added (`apops_tests.py`
   #7, 9/9 pass): a flipped last-block label leaves its own day unchanged
   and changes the persistent expert on the next day.
2. **Adaptive-mass parameter `λ_AP ∈ [0,1]`.** Prior
   `π(λ_AP) = (1−λ_AP, λ_AP/2, λ_AP/2)` over `(R,S,L)`; `w₁ = π(λ_AP)`;
   prior-centered fixed share `w_{r+1} = (1−α) w̃_{r+1} + α·π(λ_AP)`.
   `λ_AP = 0` pins `w` to `(1,0,0)` forever → AP-OPS ≡ OPS prediction by
   prediction (checked bit-for-bit). `λ_AP` is now the single tuned knob
   for "how much of the extension is used"; the nested grid contains 0.

**Four headline methods** (`apops/method.py::build_nested_rows`), identical
`q_{d,i}`: `ops` (proj-gradient, reset daily — reference), `reset_ons`
(discounted ONS, reset daily — isolates the optimizer), `persistent_ons`
(discounted ONS, carried — isolates persistence), `ap_ops` (reset R +
persistent S/L, aggregated). `reset_ons` and `persistent_ons` share one
half-life selected on validation, so their only difference is
reset-vs-carry. No-slope kept as a supporting ablation.

**Nested protocol** (`run_apops_nested.py`): outer origins Criteo 16–30 /
Avazu 5–9; per origin, history = days < d, inner validation = trailing 3
days; select ONE common config for all 3 seeds by mean inner-validation
loss. Re-selected per origin: shared mixture (`η × halflife`, 15), `λ_AP ∈
{0,.25,.5,.75,1}` × `τ ∈ {4,16,∞}h` (15), persistent/reset half-life
(`{4,16}h`). Frozen (dev only): `η_m`, ONS `(λ,η)`, OPS `(B,η0,schedule)`,
block/delay, S/L half-lives.

**Decision rules:**
- *Persistence* is supported only if `persistent_ons` improves on `reset_ons`.
- *Adaptive aggregation* is supported if `ap_ops` is non-inferior to
  `persistent_ons` on both datasets and better on ≥1; else simplify to the
  single persistent calibrator.
- AP-OPS succeeds if it beats OPS on both datasets in paired rolling-origin
  log loss, with no early-/worst-day regression, and selects `λ_AP > 0` on
  a non-trivial fraction of origins.

### Results

Jobs `12515130` (Criteo, 8254 s) / `12515131` (Avazu, 6303 s), full data,
3 seeds. Day-level inference (seeds averaged within each origin day, then
bootstrap / sign-test across origins); every row paired against `ops`.
Full table `APOPS_NESTED.md`, per-origin manifest
`{ds}/apops/nested/nested_origin_manifest.csv`.

**Criteo (15 origins, days 16–30):**

| row | mean Δ vs OPS (day-wt) | 95 % CI (day bootstrap) | CI excl 0 | origins won |
|---|---|---|---|---|
| Reset-ONS | −0.000167 | [−0.000218, −0.000125] | yes | 15/15 |
| Persistent-ONS | −0.000170 | [−0.000220, −0.000128] | yes | 15/15 |
| **AP-OPS** | **−0.000172** | [−0.000220, −0.000131] | yes | 15/15 |
| No-slope | −0.000002 | [−0.000053, +0.000047] | no | 6/15 |

**Avazu (5 origins, days 5–9):**

| row | mean Δ vs OPS (day-wt) | 95 % CI (day bootstrap) | CI excl 0 | origins won |
|---|---|---|---|---|
| Reset-ONS | +0.000019 | [−0.000090, +0.000133] | no | 3/5 |
| Persistent-ONS | −0.000049 | [−0.000143, +0.000057] | no | 3/5 |
| **AP-OPS** | **−0.000136** | [−0.000181, −0.000098] | yes | 5/5 |
| No-slope | +0.000229 | [+0.000081, +0.000397] | no (wrong side) | 0/5 |

### Verdict: AP-OPS succeeds on both datasets; the *mechanism* is dataset-specific

**AP-OPS beats OPS on both** — Criteo −0.000172 (15/15 origins, CI excl 0),
Avazu −0.000136 (5/5, CI excl 0) — with **`λ_AP > 0` on all 20 origins**
(Criteo mostly 0.75, Avazu 0.25–0.5) and no early-/worst-day regression
(AP-OPS worst-origin Δ = −0.000046 Criteo, −0.000073 Avazu — still
improvements). All three success criteria met.

But the ablations show the two datasets are driven by **different**
sub-mechanisms:

| | Criteo | Avazu |
|---|---|---|
| Reset-ONS (ONS optimizer only, daily reset) | −0.000167 — **captures the entire gain** | +0.000019 — **no gain, ≈ OPS** |
| Persistent-ONS (+ cross-day persistence) | −0.000170 — adds nothing (Δ vs Reset-ONS = 3e-6, noise) | −0.000049 — a real ~7e-5 step (CI still crosses 0 at D=5) |
| AP-OPS (+ adaptive {R,S,L} aggregation) | −0.000172 — adds nothing | −0.000136 — **adds the rest; ~3× Persistent-ONS, only its CI excludes 0** |
| what explains AP-OPS > OPS | the projected-gradient → **discounted-ONS optimizer** | **cross-day persistence + adaptive aggregation** (the optimizer alone does nothing) |
| No-slope | ≈ OPS — fixing `a` erases the gain | worse than OPS |

**Decision rules (spec section 4), applied:**
- *Cross-day persistence* — **supported on Avazu** (Persistent-ONS
  materially beats Reset-ONS, and Reset-ONS ≈ OPS there);
  **indistinguishable on Criteo** (Reset-ONS ≈ Persistent-ONS, both already
  at the full −0.00017 — the ONS optimizer, not persistence, carries
  Criteo).
- *Adaptive aggregation* — **supported**: AP-OPS is non-inferior to
  Persistent-ONS on both datasets and **materially better on Avazu**
  (−0.000136 vs −0.000049, only AP-OPS's CI excludes zero). On Criteo it
  neither helps nor hurts.
- *Slope* — essential on both (no-slope ≈ OPS on Criteo, +0.00023 on Avazu).

**Why keep the full AP-OPS rather than simplify.** No single sub-variant
wins on both datasets: Reset-ONS fails on Avazu, Persistent-ONS is
sub-significant on Avazu, and on Criteo all three ONS variants tie so
nothing is lost by carrying the extra structure. AP-OPS is the only
configuration that contains whichever mechanism each dataset needs, wins
on both, and is protected by the anchor (`λ_AP = 0` reproduces OPS) — the
nested selector never chose `λ_AP = 0`, but it is there as the floor.

Note the nested per-origin selection also *improved* the Criteo point
estimate over §5's dev-frozen rolling (−0.000172 vs −0.000119): re-picking
the mixture, `(λ_AP, τ)` and half-life per origin is not just rigour, it
recovers a little more.

### Final confirmation

Per the decision recorded this session: **the fully-nested rolling-origin
analysis is the confirmation** — no genuinely-unseen temporal stream
exists in this repo (Criteo 31 days / Avazu 10 days, all used). The
algorithm, hyper-parameter grid, and decision rules above are frozen so a
future dataset can be run once with zero tuning. The §5 fixed-test /
dev-frozen-rolling results are retained as preliminary supporting
evidence.

## Status

| step | state |
|---|---|
| implementation + unit/leakage tests (7/7) | **done** — commit `9b93394` |
| dev HPO, both datasets | **done** — `apops_selected.json` frozen |
| golden check (R alone == saved OPS; anchor-only == R) | **passed on every run** |
| fixed test, both datasets, 3 seeds | **done** — `Improves` on both (preliminary) |
| §5 dev-frozen rolling origin, both datasets | **done** — `Improves` on both (preliminary) |
| §7 implementation corrections + cross-day test | **done** — 9/9 tests pass |
| §7 fully-nested rolling origin (4 methods, per-origin selection) | **done** — AP-OPS beats OPS on both (15/15, 5/5); mechanism dataset-specific |
| §7 frozen algorithm / grid / rules for future confirmation | **done** — `APOPS_FROZEN.md` |
| this document | **complete** |

### Still open / optional (plan does not gate on these)

- **Rolling-origin figures** — only CSV tables + day-level stats exist
  (`{ds}/apops/rolling/seed*/per_day_metrics.csv`,
  `{ds}/apops/{final,rolling}/apops_day_level.csv`). The weight-path CSVs
  (`{ds}/apops/final/seed*/apops_weight_path.csv`) are ready to plot.
- **Fresh-stream confirmation.** The fixed-test days were already
  inspected by the DualTime-CTR experiment; the AP-OPS *config* was frozen
  on dev only, so the fixed-test result is honest, but the cleanest
  confirmation would still be a genuinely untouched date range or a third
  dataset with the already-frozen shared config and no retuning.
- **Theorem** against the exact implemented algorithm (exp-concave base
  ONS → log static regret in dim 2; delayed fixed-share meta → switching
  bound `Loss_AP ≤ min_paths Σ + Meta(T,K=3,S) + Delay(T,D)`; `S = 0`
  gives the OPS-anchor guarantee). The code is written to match the proof
  conventions (block-mean loss, clip `ε = 1e-5`, `Proj` after the ONS
  step, `γ` discount, daily R reset, fixed-share timing).
