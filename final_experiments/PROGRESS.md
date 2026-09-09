# DualTime-CTR final experiment: progress log

Resumption document for the "Experimental Plan to Complete All TBD Results
in DualTime-CTR" spec (6-method headline table: Expanding, Best Fixed
Window, ARW, AdaMoE, OPS, DualTime-CTR; Criteo + Avazu; seeds 0,1,2). If
this session ends before the plan is finished, **read this file first**,
then the spec (in the conversation that requested it) for exact formulas.

## TTAM revised Section 6 -- Criteo + Avazu DONE; ablation revised to 2x2 (2026-09-09) (branch `ttam-additional-experiments`, started 2026-09-08)

New plan `final_experiments/TTAM_Additional_Experiments_Plan.pdf` (8 Sep
2026): evaluate the integrated **AMG-TP + AP-OPS** pipeline ("TTAM") under
one *corrected* fully-nested rolling-origin protocol -- every data-tuned
setting reselected from data strictly before each evaluated origin, no
inheritance from a frozen file whose dev window overlaps early outer days.
Six-method main table + four-variant ablation + Criteo bidding replay.
**This document specifies work to run; it does not report new results.**

**Done (commits 5761624, 754bfec):**
- `final_experiments/ttam/` package: `bank.py` (shared 5-horizon expert
  bank {1,3,7,14,expanding}, tie-aliased), `fixed_mix.py` (constant
  validation-fitted simplex mixture = no-AMG-TP control), `amgtp.py`
  (causal AMG-TP on the 5-horizon bank: sample gate + learned persistence
  + deployed-weight memory, maturation-aware), `method.py` (nine variants:
  expanding / best_fixed_window / arw / adamoe / ops / ttam / without_both
  / amgtp_only / apops_only), `build_banks.py` (bank + context-sketch disk
  cache).
- `run_ttam_nested.py` -- 3-pass nested runner (module selection on
  uncalibrated inner loss -> calibration selection on the chosen module's
  causal stream -> replay winners + score origin once). **Origins within
  each pass run in parallel** (`--n-workers`, deterministic, joblib
  processes, results-identical to serial). `ttam_stats.py` (paired
  day-level gain, 95% paired day-bootstrap + block-2 MBB, seed 20260908,
  seeds averaged first), `ttam_figure.py` (Section 6.2 two-panel figure),
  `run_ttam_bidding.py` (Criteo bidding replay, recorded display cost as
  price proxy, changes only the prediction input).
- `TTAM_FROZEN.md` -- frozen grids / staged-selection rule / statistics +
  bidding protocol / reuse audit, fixed before any TTAM result.
- `ttam_tests.py` 10/10: lambda_AP=0 => AP-OPS==OPS; causal invariance;
  cross-midnight maturation; simplex/param bounds; deployed-weight memory
  update; determinism.
- Criteo 3-seed expert banks cached (`_bankcache/`, gitignored). Full
  pipeline (nested runner -> stats -> figure) validated on a 3% Criteo
  smoke (193s, all 9 variants, 15 origins x 3 seeds). Bidding replay
  asserts row alignment which only holds at `sample_frac=1.0` -- it
  errors on the subsampled smoke by design; verified `load_criteo`
  (twoscale) and the bidding cost loader do the identical
  `(day, sec_in_day)` stable sort with no row filtering at full data.

### CRITEO NESTED RUN: DONE (2026-09-08, commit 45e50a6, local run ~1h)

`final_experiments/ttam/criteo/nested/` (`run_ttam_nested.py --n-workers 4`;
had to drop from 6 -> 4 after loky worker deaths under memory pressure --
a root `java` job appeared using 26 GB; per-seed checkpointing added so
the restart resumed at pass 2). Section 6: `TTAM_SECTION6_FINDINGS.md`,
`TTAM_SECTION6_STATS.md`, `ttam/section6_figure.png`,
`ttam/criteo/{nested,bidding}/`.

**Prediction (equal-day mean log loss, D=15):** TTAM 0.608029 < OPS
0.608201 < AdaMoE 0.608410 < ARW 0.608631 < BestFixedWindow 0.608744 <
Expanding 0.609475. TTAM beats every baseline 15/15 origins, all CIs
exclude 0; margin over OPS = 1.7e-4 (small, directionally unanimous).

**Downstream bidding (Criteo, matched spend):** TTAM wins FEWER clicks
than every baseline (-0.02% vs OPS to -0.07% vs AdaMoE), all CIs below
zero. The log-loss edge does not convert to bidding value.

### AVAZU NESTED RUN: DONE (2026-09-09, commit 220a4d6, local ~77 min)

Needed `twoscale.data.load_avazu` made chunk-hashed (commit c1b4c9e) --
the old path materialised the full 40M-row string frame (~50GB peak,
needed a 150-230GB node); now each read chunk is hashed to a sparse block
and the strings dropped, bit-identical output, ~25GB peak. Banks built
one seed per process (`--n-jobs 1`, ~24 min each) then
`run_ttam_nested.py --source avazu --n-workers 3`.

Prediction (equal-day mean log loss, D=5): TTAM 0.400934 < OPS 0.401147
< AdaMoE 0.401538 < Expanding 0.402094 < BestFixedWindow 0.402213 < ARW
0.402359. TTAM beats every baseline (5/5 vs Expanding/AdaMoE/OPS, 4/5 vs
BFW/ARW); margin over OPS +2.1e-4 (Criteo +1.7e-4).

### ABLATION REVISED TO 2x2 + BOTH DATASETS RE-RUN (2026-09-09, commit 4a65e91 + rerun)

User revised the ablation to a clean **2x2: historical predictor
{AMG-TP, expanding-history} x calibration {AP-OPS, none}**. Window
baseline = expanding history, pre-registered. Dropped the two
fixed-simplex-mixture arms (`without_both`, `apops_only`), added
`expanding_apops`. `ttam_stats.py::factorial_2x2` reports the four
marginal effects + interaction with paired day-bootstrap CIs
(`section6_factorial.json`, "6.3 Ablation -- 2x2" in
`TTAM_SECTION6_STATS.md`). Both datasets re-run (pass 1 unchanged ->
resumed from checkpoint); the main table (6 methods) is byte-identical.

**This changed the ablation conclusion.** Against a plain
expanding-history baseline (not the fitted mixture the old arms used --
itself an adaptive multi-horizon combination), **BOTH components
contribute significantly on both datasets**:

| | Criteo A_m | Avazu A_m |
|---|---|---|
| expanding | 0.609475 | 0.402094 |
| expanding + AP-OPS | 0.608807 | 0.401737 |
| AMG-TP (amgtp_only) | 0.608401 | 0.401514 |
| AMG-TP + AP-OPS (TTAM) | 0.608029 | 0.400934 |

All 8 marginal-effect CIs clear zero, 15/15 & 5/5 origins: AMG-TP
-0.5..1.1e-3, AP-OPS -0.4..0.6e-3. **Interaction disagrees by dataset:**
Criteo +3.0e-4 [+1.5e-4,+4.6e-4] = *substitutes* (each recovers most of
the gain alone; stacking gives less than the sum); Avazu -2.2e-4
[-3.3e-4,-1.1e-4] = *synergy*. The earlier "AMG-TP adds nothing" reading
was an artefact of ablating against the fitted mixture.

**Verdict:** TTAM (both timescales) is best on both datasets; both the
AMG-TP historical model and the AP-OPS calibrator carry real weight
against a naive baseline; they are non-additive with opposite sign
across the two datasets. Full write-up `TTAM_SECTION6_FINDINGS.md` §6.3;
two-panel figure `ttam/section6_figure.png`. `final_predictions/`
(Criteo 0.6GB + Avazu 1.6GB) deleted after -- gitignored, regenerable.

**Not done:**
- Avazu downstream bidding -- needs a frozen simulated-price spec (plan
  section 10); Criteo replay is the primary downstream result.
- Manuscript placeholders (plan section 6.x) -- no paper source file in
  this repo (same standing note as below); the `.md` + figure are the
  deliverables produced here.

---

## IMPORTANT: DualTime-CTR is not the capacity-ladder V5 (user clarification, 2026-09-05)

DualTime-CTR's within-day residual model uses an ONLINE-updated `w`:
`w_{d,i+1} = Pi_W(w_{d,i} - eta_i * grad(l_i(w_{d,i})))` (`dualtime/online.py`
`replay_day`, block-cadence discretization, `eta_k=B_w/sqrt(k)`, `w` resets
to 0 every day). This is DIFFERENT from the older capacity-ladder V5
(`withinday/adapters.py` `V5Linear`, `withinday_experiments/`), whose `w`
is trained OFFLINE on historical days and FROZEN during test -- only the
history features change causally there, not the weights. **V5's result
motivated DualTime-CTR's `phi(x,h)` architecture (the hashed
context-history bilinear interaction); it is not itself an implementation
of DualTime-CTR and must not be reported as if it were.**

Verified (2026-09-05): `dualtime/online.py::replay_day`, used by
`final_experiments/methods.py::dualtime_method`, which is what
`run_final.py` calls for the "DualTime-CTR" row of the headline table --
**already implements the online version**, not V5's offline/frozen one.
No code change was needed for this; the module docstring was updated to
state the distinction explicitly per the user's request. Every
`final_experiments/` result recorded below (both datasets done) is
therefore the correct final online DualTime-CTR, not V5. The separate
`run_diagnostic.py` arm `frozen_v5` is the deliberate V5 comparison
(review comment 3) and is labelled as such everywhere.

## AP-OPS follow-on: DONE and merged to main (2026-09-06, merge commit cdb9d14)

New plan `final_experiments/AP_OPS_Minimal_Experiment_Plan.pdf` (uploaded
to origin/main commit 3680a26; a longer `PLAN_.md` uploaded then deleted
-- **the minimal PDF is the spec**). Follow-on to the DualTime result
below: since full OPS was the strongest calibrator everywhere, AP-OPS
("Adaptive-Persistence OPS") keeps the 2-parameter Platt map but makes its
*temporal memory* adaptive -- one reset anchor R (= exact current OPS,
`twoscale.calib.replay_day`) + two persistent full-Platt experts S/L
(discounted **Online Newton Step**, 4h / 16h half-life, state carried
across days) combined by a **delayed fixed-share** meta rule over matured
block losses. Only the calibration layer is replayed; the frozen shared
cross-day mixture + locked OPS hyperparams are reused from each dataset's
`selected_configs.json` (nothing upstream of calibration re-tuned).

**Code:** `final_experiments/apops/{experts,aggregate,method}.py` +
`run_apops_{hpo,final,rolling,analysis}.py` + `apops_tests.py` (7/7:
anchor-equivalence, future/unmatured-label leakage, weight simplex, param
bounds, determinism) + `apops_{hpo,final,rolling}_{criteo,avazu}.slurm`.
Write-up `APOPS_FINDINGS.md`; day-level tables `APOPS_DAY_LEVEL.md`;
per-run `{criteo,avazu}/apops/{hpo-outputs, final/, rolling/}`.

**Golden check passed on every run:** R alone reproduced the saved
headline OPS `mean_imp_wt_ll` to `<5e-4`; anchor-only AP-OPS (weights
pinned to (1,0,0)) is bit-identical to R.

**Jobs (all DONE):** HPO 12513590 (criteo) / 12513591 (avazu); fixed test
12514073 / 12514074; rolling origin 12514084 / 12514085. Frozen configs:
Criteo ONS lam=0.1/eta=0.25, Avazu lam=1.0/eta=4.0; both eta_m=100, tau=4h
(most aggressive on dev); single-memory dev pick S (criteo) / L (avazu).
delta_NI = 3.25e-5 (criteo) / 1.29e-5 (avazu) -- tight, because OPS barely
beats the raw mixture on dev.

### AP-OPS RESULT: decision-rule verdict "Improves" on BOTH datasets

First method in the whole AMG-TP / twoscale / withinday / DualTime line to
**beat current OPS reproducibly on both public datasets** at the day level.

| dataset | current OPS | AP-OPS | day-level delta vs OPS (95% CI) | fixed | rolling |
|---|---|---|---|---|---|
| Criteo | 0.606958 | **0.606834** | -0.000125 [-0.000149, -0.000101] | 9/9 days | 15/15 origins, sign p 6e-5 |
| Avazu  | 0.387443 | **0.387171** | -0.000270 [-0.000306, -0.000238] | 3/3 days | 5/5 origins, sign p 0.062 (D-floor) |

CI entirely below zero on both the fixed test and rolling origin; unanimous
across seeds and origins; no early-day or worst-day regression (Avazu's
biggest AP-OPS gain, -0.0005, is in the pre-feedback window -- the
persistence-helps-early-day story the plan's motivation predicted).

**Two ablation findings (plan's two mechanism claims):**
1. **No-slope (a fixed at 1, update only b) is WORSE than plain OPS** on
   both datasets (criteo +0.00003, avazu +0.00028) -> the learned slope
   DOF is essential.
2. **Single persistent expert (no meta-mix) ~= AP-OPS**: criteo -0.00013
   vs -0.00012, avazu -0.00018 vs -0.00027. -> **the win is cross-day
   persistence** (carrying full slope+intercept calibration state across
   the day boundary via discounted ONS), NOT the memory-scale mixture. The
   fixed-share meta layer buys robustness (dev-optimal single expert
   differs by dataset) + the switching-comparator guarantee + a small
   extra avazu gain, not a large accuracy improvement.

Mechanism: Criteo meta weights move a lot (R/S/L final ~0.28/0.36/0.36,
range [0.06,0.87] over 1432 updates) and downweight the reset anchor;
Avazu weights stay near uniform (only 3 test days = too few blocks).

### AP-OPS additional experiments: DONE (2026-09-07, merge e3f52b6..8e76107)

User's "Minimal Additional Experiments" spec: implementation corrections
(absolute-time feedback queue; `lambda_AP in [0,1]` adaptive mass with
`lambda_AP=0` == OPS) + a fully **nested** rolling-origin comparison of
4 methods (`ops` / `reset_ons` / `persistent_ons` / `ap_ops`) to split the
gain into optimizer vs persistence vs aggregation. Code
`apops/{experts,aggregate,method}.py` + `run_apops_nested.py`; jobs
12515130 (Criteo) / 12515131 (Avazu). Write-up `APOPS_FINDINGS.md` §7,
`APOPS_NESTED.md`, frozen spec `APOPS_FROZEN.md`, resumption doc
`APOPS_ADDITIONAL_PROGRESS.md`.

**AP-OPS beats OPS on both** (nested): Criteo -0.000172 [-0.000220,
-0.000131] 15/15 origins; Avazu -0.000136 [-0.000181, -0.000098] 5/5;
`lambda_AP > 0` on all 20 origins. **Mechanism is dataset-specific:**
Criteo = the discounted-ONS optimizer alone (`reset_ons` -0.000167 gets
the full gain); Avazu = persistence + adaptive aggregation (`reset_ons`
~= OPS there; only `ap_ops`'s CI excludes 0). Slope essential on both.
Keep the full AP-OPS -- no single sub-variant wins on both datasets.
Final confirmation = the nested analysis itself (user decision; no
untouched stream exists). `APOPS_FROZEN.md` records the frozen
algo+grid+rules for a future dataset.

**Still open (nothing gates on these):** rolling-origin / weight-path
figures (all CSVs exist, no plots); the switching-regret theorem against
the exact implemented algorithm (code written to match proof conventions
-- block-mean loss, clip eps=1e-5, Proj after ONS step, gamma discount,
daily R reset, fixed-share timing).

Consistent with the repo's "shallow real drift" standing finding: the
effect is small (<3e-4 log loss) -- but unlike every prior method it is
directionally unanimous, present on rolling origin, and mechanistically
clean.

---

## Current status (read this first) -- updated 2026-09-05, commit d5f2f21

**All computation is done. Nothing is running.** (`squeue -u $USER` shows
only an unrelated interactive `bash` job.) The pipeline through the primary
3-seed locked test AND all three pre-decision review follow-ups are
finished, committed and pushed to `main`.

### Done and committed

| step | artifacts | commit |
|---|---|---|
| HPO (both datasets, 3 seeds, dev-only) | `{criteo,avazu}/hpo/selected_configs.json` + full grid CSVs | c3077b4, 8286b85 |
| Primary 3-seed locked test, 6 methods | `{criteo,avazu}/final/headline_results.csv` | f011eab |
| Consolidated 12-cell headline | `FINDINGS.md` | fab81be, 8883ead |
| **Review 1: day-level stats** | `day_level_stats.py`, `DAY_LEVEL_STATS.md` | d5f2f21 |
| **Review 1: 6-method rolling-origin** | `run_rolling.py`, `{criteo,avazu}/rolling/`, `DAY_LEVEL_STATS_ROLLING.md` | d5f2f21 |
| **Review 2: eta=1e6 = follow-the-leader** | corrected `FINDINGS.md` / `PROGRESS.md` | d5f2f21 |
| **Review 3: frozen-V5-vs-online diagnostic** | `run_diagnostic.py`, `{criteo,avazu}/diagnostic/`, `DAY_LEVEL_STATS_DIAGNOSTIC.md` | d5f2f21 |
| Overall response + recommendation | `final_experiments/REVIEW_RESPONSE.md` | d5f2f21 |

### Headline numbers to carry forward

- **Criteo:** every adaptive method beats Expanding at the day level --
  9/9 fixed-origin, 15/15 rolling-origin, bootstrap CI excl. 0, sign-test
  p at the D-floor. Order by log loss: OPS < DualTime-CTR < AdaMoE ~=
  long_only < ARW < BestFixed. **OPS beats online DualTime-CTR at the day
  level** (rolling: -0.000257 vs -0.000140 vs long_only; DualTime 0/9
  fixed-origin days head-to-head vs OPS).
- **Avazu:** underpowered -- D=3 fixed / D=5 rolling, sign-test floor
  p >= 0.0625. AdaMoE is the cleanest (5/5 rolling origins, CI excl. 0).
  DualTime-CTR shows no reproducible edge; its rolling Expanding CI
  crosses 0.
- **Diagnostic:** Criteo `L(frozen V5) ~= L(online DualTime)`, both < OPS
  -> no within-day contextual signal for either. Avazu `L(frozen V5) <
  L(online DualTime)` but only ties OPS, not significant.

### The open decision (why this doc still exists)

**Keep online DualTime-CTR as the headline method, or not?** On this
evidence it never beats OPS. `REVIEW_RESPONSE.md` lays out two honest
write-ups:
  (a) report it as a negative result (theory-friendly online contextual
      model doesn't beat a 2-parameter online scalar), or
  (b) build the **warm-start refinement** -- `w_{d,1} = w_historical`
      (offline-learned residual model from previous days) then projected
      OGD within the current day, instead of `w_{d,0}=0` every morning.
      The OGD regret bound still holds with non-zero init (constant
      becomes the initial distance to the comparator). Only Avazu hints
      it helps, and that hint is NOT significant.

### If you pick (b) -- next concrete steps for the new session

1. Implement warm-started DualTime: extend `dualtime/online.py::replay_day`
   to take `w_init` (offline-fit `w` over `phi` on the historical days,
   e.g. logistic regression / the frozen `withinday` V5 weights), keep the
   projection `||w|| <= B_w` and `eta_k = B_w/sqrt(k)`. Add it as a 5th
   arm in `run_diagnostic.py`.
2. **Do NOT re-score it on Criteo days 22-30 / Avazu days 7-9 and call it
   a locked test** -- those days are now inspected. Use the diagnostic
   result as development evidence only.
3. Confirmation must come from a genuinely untouched chronological stream:
   either a later date range of the same logs if more data exists, or a
   third dataset. Freeze the protocol before looking (see
   `withinday_experiments/ROLLING_PROTOCOL_FREEZE.md` for the pattern).

### Still not done (independent of the decision)

- Rolling-origin **figures** (only CSV tables + day-level stats exist).
- Paper text -- no paper source file has been located in this repo; ask
  the user where it lives.

## What's done

### 1. Citation verification (done, via WebSearch/WebFetch)

- **ARW = Han, Huang & Wang (2024)**, *"Model Assessment and Selection
  under Temporal Distribution Shift"*, ICML 2024, arXiv:2402.08672. "ARW"
  = Adaptive Rolling Window. Algorithm reconstructed from a WebFetch
  summary of the paper (bias-proxy + empirical-Bernstein variance-proxy
  window selection over nested windows, pairwise comparison via the
  loss-difference sequence, single-elimination tournament for >2
  candidates) -- **not verified against the authors' source code or a
  full reading of the paper**. Implemented in `dualtime/arw.py`.
- **AdaMoE = Liu et al. (2022)**, *"On the Adaptation to Concept Drift for
  CTR Prediction"*, arXiv:2204.05101 (first author Congcong Liu).
  Confirmed real; exact update formula not independently verified, so
  `dualtime/adamoe.py` implements the spec's own precise recipe (uniform
  init, EMA of instantaneous inverse-loss softmax, momentum lambda)
  rather than guessing the paper's exact equations.
- **OPS = Gupta & Ramdas**, Online Platt Scaling. Already implemented in
  this repo: `twoscale/calib.py` (`CalibConfig(platt=True)`, block-based
  causal replay). No new code needed for OPS itself, only its grid/wiring
  into the final comparison.

### 2. What's already reusable, unchanged (confirmed to already match the spec)

- `twoscale/data.py`, `twoscale/splits.py` -- loaders and the exact
  16/6/9 (Criteo) and 5/2/3 (Avazu) day splits the spec asks for.
- `twoscale/longterm.py` -- the shared three-expert bank is *already*
  `HORIZONS = ("roll3", "roll7", "expanding")`, no five-expert bank
  anywhere in this repo. `SGDClassifier(loss="log_loss", penalty="l2",
  alpha=1e-4, random_state=seed)` already matches section 4 (sklearn's
  own defaults already give max_iter=1000, tol=1e-3). The adaptive
  cross-day mixture (`adaptive_weights`, eta/halflife) already uses
  exactly the eta in {10,30,60,150,1e6} / halflife in {3,5,10} grid
  section 8 specifies (see `twoscale_hpo.py::MIX_GRID`).
- `twoscale/calib.py` -- OPS itself, and its grid (B in {0.25,0.5,1,2},
  eta0 in {0.01,0.03,0.1,0.3}, schedule in {const, inv_sqrt}) already
  matches section 10 exactly (see `twoscale_hpo.py::GRID`). Block-based
  update with a settable `block_sec` already supports per-dataset block
  widths (Criteo 900s / Avazu 3600s -- Avazu's needs to actually be
  changed to 3600s for this experiment; every prior run in this repo used
  900s for both datasets, matching the *old* within-day plan, not this
  one).
- `withinday/blocks.py`, `withinday/contextsketch.py` -- the frozen
  feature architecture (context sketch dim m=32, block token eq 5,
  deterministic summary eq 11 with EWMA half-lives [1,4,16]) the spec's
  section 11 explicitly says to reuse rather than re-derive. Confirmed:
  using `block_sec=3600` for Avazu (matching its native hourly
  resolution) automatically gives every same-hour impression the same
  `k_avail` / matured-history state, satisfying section 9.1's Avazu
  requirement with no special-casing needed.

### 3. New code written and unit-tested this session

`dualtime/` package (commit `3eb8f37`), `dualtime_tests.py` (26/26 pass):

- `dualtime/arw.py` -- `select_expert(loss_history, delta, min_history,
  fallback)`: causal tournament over the 3 experts' past per-day losses.
  `pairwise_prefers_first`, `_select_window`, `_bernstein_halfwidth` are
  the building blocks.
- `dualtime/adamoe.py` -- `initial_weights()`, `next_weights(weights,
  day_losses, lam)`, `mixture_prediction(weights, preds)`.
- `dualtime/online.py` -- DualTime-CTR's within-day module.
  `DualTimeConfig(block_sec, delay_sec, m, cross_dim, B_w, eps)`,
  `build_hash_projection`, `build_phi` (norm-bounded to <=1),
  `replay_day(q, y, sec_in_day, X_day, R_sketch, Ra, Rs, cfg)` -- w resets
  to 0 daily, block-cadence projected online GD, `eta_k = B_w/sqrt(k)`,
  mirrors `twoscale.calib.replay_day`'s structure generalized to a vector
  `w` over `phi` instead of a scalar `(a,b)` over `logit(q)`.

Tests confirm: ARW only ever uses the arrays passed to it and picks the
genuinely-better expert; AdaMoE weights always sum to 1, lambda=0/1
extremes behave as expected; DualTime's `p_hat == q` identity holds
within the first block, the `B_w` projection is never violated, and a
future-label-perturbation test confirms bitwise-unchanged predictions
before the causal horizon.

## What's NOT done yet (in the spec's section 19 order)

1. `final_experiments/manifest.json` -- not written.
2. Full-data feature matrices -- Criteo already loadable as-is
   (`sample_frac=1.0`); Avazu needs a decision: the spec asks for the
   *full* ~40M-row stream (`sample_frac=1.0`) rather than the 20% sample
   used everywhere else in this repo so far -- this is a bigger load than
   anything run yet and should be tried on real hardware before assuming
   it works interactively; use `.slurm`, not an interactive session (past
   OOM-kills on this login node: manifest-building jobs already had to
   move to sbatch for full Criteo).
3. Shared three-expert bank for seeds 0,1,2 -- trivial once done for one
   seed (`twoscale.longterm.build_bank(ds, eval_days, seed=s)`), not yet
   run for this experiment's exact day ranges.
4. Leakage/identity test suite (spec section 15) -- partially covered by
   existing `withinday_tests.py` / `dualtime_tests.py` causal tests
   (future-label perturbation, no-history identity, projection bounds
   already have direct analogues), but the spec's *specific* list
   (shared-q identity between OPS and DualTime, shared-expert-bank
   identity across all 5 non-Expanding methods, "no test day in any HPO
   loss calculation" programmatic assertion) has not been written or run
   as its own explicit suite yet. Should produce `leakage_tests.txt`.
5. Best Fixed Window HPO -- not run for this experiment (trivial: reuse
   `long_term_predictions(bank, days, mode=h)` for h in {roll3, roll7,
   expanding}, pick by dev loss, per seed then average per section 6).
6. ARW HPO (delta in {0.05, 0.10, 0.20}) -- not run.
7. AdaMoE HPO (lambda in {0, 0.25, 0.50, 0.75, 0.99}) -- not run.
8. Shared adaptive cross-day mixture HPO (eta x halflife, 15 configs) --
   not run for this experiment's exact day range/seed set (a very similar
   grid has been run before for the twoscale/withinday lines, but not
   with the 3-seed-averaged selection rule section 6/8 specifies here,
   and not saved under `final_experiments/`).
9. OPS HPO (32 configs) -- not run under this experiment's protocol
   (block widths per dataset per section 9.2 -- Avazu at 3600s is new).
10. DualTime B_w HPO (5 configs: {0.25,0.5,1,2,4}) -- not run. This is the
    only new HPO grid this experiment needs beyond what's reusable.
11. `selected_configs.json` -- not written (depends on 5-10).
12. **Primary 3-seed chronological test, all 6 methods** -- not run. This
    produces the 12 TBD values. Nothing in the headline table is filled in.
13. Rolling-origin confirmation (Criteo days 16-30 = 15 outer days, Avazu
    days 5-9 = 5 outer days) for all 6 methods -- not run. Note this is a
    *different* outer-day range for Criteo than the earlier (pre-this-plan)
    `withinday_experiments/rolling/criteo` run (16-21 only, V5 only) --
    that earlier run is preserved as-is and is NOT a substitute.
14. Day-level statistical analysis (section 14) -- not run; reusable
    building blocks already exist in `withinday/daystats.py`
    (`day_summary`, `leave_one_day_out`, `moving_block_bootstrap_ci`) and
    should be reused rather than reimplemented.
15. Tables/figures (`headline_results.csv/.tex`, per_seed/per_day CSVs,
    the 3 required figures) -- not generated.
16. Paper text updates -- not started (no paper file has been located in
    this repo; ask the user where it lives before attempting this step).

## Concurrently running / already-committed OLD work (separate from this plan)

A `withinday_experiments/rolling/criteo` job (job 12491243, started
before this new plan was given) was still running as of the last check
(~1h20m elapsed, no per-day progress printed -- expect several more hours
per the cost model in `withinday_experiments/ROLLING_PROTOCOL_FREEZE.md`).
This is the *old* single-method (V5 only) rolling-origin analysis over
Criteo days 16-21, explicitly superseded/reframed as "preliminary /
exploratory" evidence by the new plan (section 20's instruction to keep
the frozen-head capacity-ladder table labeled as such) -- it is safe to
let finish (it still feeds that exploratory section) and does not block
starting new-plan work, which needs new code paths regardless.

## Criteo primary 3-seed locked test: DONE (2026-09-05, job 12491604)

`final_experiments/run_final.py` written this session (reads only frozen
`selected_configs.json`, no per-seed re-tuning) and run at full scale
(`final_criteo.slurm`, 16.5M rows x 3 seeds, 838s). Output:
`final_experiments/criteo/final/{headline_results.csv,summary.json,
seed{0,1,2}/}`.

Headline (mean impression-weighted log loss, 3 seeds, test days 22-30):

| method | mean log loss | delta vs Expanding | seed-day win frac |
|---|---|---|---|
| **OPS** | **0.606958** | -0.001120 (CI excl. 0) | 27/27 |
| DualTime-CTR | 0.607070 | -0.001010 (CI excl. 0) | 27/27 |
| AdaMoE | 0.607157 | -0.000922 (CI excl. 0) | 27/27 |
| Best Fixed Window | 0.607290 | -0.000790 (CI excl. 0) | 27/27 |
| ARW | 0.607301 | -0.000782 (CI excl. 0) | 27/27 |
| Expanding | 0.608067 | -- | -- |

All five adaptive methods beat plain Expanding significantly and
unanimously (27/27 seed-days). **OPS narrowly beats DualTime-CTR on
Criteo** (delta +0.000112, i.e. DualTime-CTR's online within-day residual
adds nothing over OPS's plain global scalar calibration here) -- consistent
with this whole repo's standing finding of shallow real intraday drift on
Criteo (same story as `twoscale`'s `combined ~= long_only` and
`withinday`'s sub-materiality result). DualTime-CTR still clearly beats
ARW/Best-Fixed-Window/AdaMoE. Whether this OPS-over-DualTime-CTR ordering
also holds on Avazu (thinner, more diurnal structure -- where the older
`withinday` V5 result and the AMG-TP line both found a small but real
effect) is the open question the Avazu final run will answer.

Statistical caveat: RESOLVED (2026-09-05, review comment 1). `run_final.py`
pools (seed, day) as the replicate unit, which over-states precision for a
temporal claim. `final_experiments/day_level_stats.py` re-does it right --
`Lbar_{m,d} = mean_s L_{m,d,s}` then bootstrap/sign-test across the D days
-- see `DAY_LEVEL_STATS.md` (fixed origin) and `DAY_LEVEL_STATS_ROLLING.md`
(rolling origin, `run_rolling.py`). The Criteo conclusion survives (9/9
and 15/15 days, day-level CI excludes 0); Avazu is under-powered (D=3
fixed / D=5 rolling, sign-test floor p >= 0.0625). Use the rolling-origin
tables, not the "27/27 seed-days" line, as the significance statement.

## Avazu HPO: DONE (2026-09-05, job 12491538, 4090s)

Full 40M-row Avazu, 3 seeds. `final_experiments/avazu/hpo/selected_configs.json`:
Best Fixed Window -> roll3, ARW delta=0.05, AdaMoE lambda=0, **shared mixture
eta=1e6** -- CORRECTED (review comment 2): this is NOT "approximately equal
weighting", it is follow-the-leader / winner-take-all (`w_h propto
exp{-eta*(Lbar_h - min)}`, so eta->inf -> one-hot on the argmin horizon;
equal weighting is eta->0). Confirmed empirically in
`avazu/diagnostic/seed*/mixture_weights.csv` (one-hot daily weights, mostly
roll3). So Avazu's mixture does hard daily horizon *selection* (favouring
the short window), not blending -- cross-day horizon *blending* and the
within-day residual module still don't help on Avazu, but cross-day
*selection* does a little. OPS B=0.25/eta0=0.3/const (same
as Criteo), DualTime B_w=4.0. `final_avazu.slurm` (job 12491705) submitted
immediately after -- 3-seed locked test on full data, `--warmup 3`, into
`final_experiments/avazu/final/`.

## Avazu primary 3-seed locked test: DONE (2026-09-05, job 12491705, 2357s)

Full 40M-row Avazu, 3 seeds, test days 7-9. Output:
`final_experiments/avazu/final/{headline_results.csv,summary.json,seed{0,1,2}/}`.

| method | mean log loss | delta vs Expanding (day-level) | CI excl. 0? |
|---|---|---|---|
| **AdaMoE** | **0.387402** | -0.000211 | **yes** |
| OPS | 0.387443 | -0.000201 | no |
| ARW | 0.387530 | -0.000076 | no |
| Expanding | 0.387596 | -- | -- |
| DualTime-CTR | 0.387610 | -0.000041 | no |
| Best Fixed Window | 0.388147 | +0.000443 | no |

**Much weaker signal than Criteo.** Only AdaMoE clears significance (and
even that's a small ~0.05% effect); OPS/ARW/DualTime-CTR are directionally
better than Expanding but every CI crosses zero -- not distinguishable from
noise at 3 seeds. **DualTime-CTR shows no real edge on Avazu in this final
protocol** -- its online within-day residual does not reproduce the
"small but real diurnal effect" the earlier AMG-TP and capacity-ladder V5
lines found (those were different comparisons: AMG-TP vs Han ARW, and V5
vs Online Platt with `sample_frac=0.2`, not this 6-method full-data
protocol against Expanding). Note the sign quirk for DualTime-CTR: its
`mean_imp_wt_ll` (0.387610) is fractionally *worse* than Expanding's
(0.387596, impression-weighted across all test days) while its
`mean_delta_vs_expanding` (day-equal-weighted, seed x day pooled) is
slightly negative/better -- the two metrics use different weighting and
can disagree at this small a margin; report both, don't collapse to one
sentence.

**Both halves of the primary 3-seed locked test (spec section 12, item 12
of section 19) are now DONE.** Full 12-cell headline table (6 methods x 2
datasets) exists across `final_experiments/{criteo,avazu}/final/
headline_results.csv`. Headline picture: on Criteo, OPS/DualTime-CTR/
AdaMoE/Best-Fixed-Window/ARW all significantly beat Expanding (DualTime-CTR
2nd behind OPS); on Avazu, only AdaMoE clears significance and every
method is close to Expanding -- the adaptive-training story is much
stronger on Criteo than Avazu in this final protocol, opposite of what
the project's earlier (different-protocol) Avazu results might suggest.
**This distinction matters for the paper's headline claim and should be
stated as-is, not smoothed over.**

## Review follow-ups: DONE (2026-09-05) -- see `final_experiments/REVIEW_RESPONSE.md`

Three pre-decision review comments, all addressed:

1. **Day-level statistics** -- `day_level_stats.py` (seeds averaged first,
   then bootstrap/sign-test across days). `DAY_LEVEL_STATS.md` (fixed
   origin) + `DAY_LEVEL_STATS_ROLLING.md`. **6-method rolling-origin
   runner** `run_rolling.py` (job 12496670/12496671): Criteo 15 origins,
   Avazu 5 origins, 3 seeds, per-origin re-selection of the cheap knobs.
   Criteo: all adaptive methods beat Expanding 15/15, day-level p 6e-5.
   Avazu: D=5, AdaMoE 5/5 with CI excluding 0 (p 0.0625 floor); others
   directional. Across every origin, OPS >= online DualTime-CTR.
2. **eta=1e6 = follow-the-leader**, not equal weighting (see Avazu HPO
   section above). FINDINGS.md corrected.
3. **Frozen V5 vs online DualTime diagnostic** -- `run_diagnostic.py`
   (job 12496668/12496669), `{criteo,avazu}/diagnostic/`. DEVELOPMENT
   EVIDENCE ONLY. Criteo: L(frozen V5) ~= L(online DualTime), both < OPS.
   Avazu: L(frozen V5) < L(online DualTime) but only ties OPS, D=3, ns.
   **Recommendation: do not present the current w_{d,0}=0 online
   DualTime-CTR as beating the baselines -- it never beats OPS. Either
   report it as a negative result or pursue the warm-start refinement and
   validate on a fresh stream.**

Still not done: rolling-origin figures (CSV tables + day-level stats only),
paper text (no paper file located in this repo).

## Suggested resumption order

Follow spec section 19 literally, steps 1-4 first (manifest, full-data
build for both datasets on slurm, 3-seed expert bank, leakage tests) since
nothing after that can be trusted without them. Given compute cost, run
HPO stages (5-10) as separate slurm jobs per dataset, writing each grid's
full table (not just the winner) to `final_experiments/{dataset}/hpo/`.
Do not run the primary 3-seed test (step 12-13) until `selected_configs.json`
is frozen and committed.
