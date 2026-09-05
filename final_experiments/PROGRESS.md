# DualTime-CTR final experiment: progress log

Resumption document for the "Experimental Plan to Complete All TBD Results
in DualTime-CTR" spec (6-method headline table: Expanding, Best Fixed
Window, ARW, AdaMoE, OPS, DualTime-CTR; Criteo + Avazu; seeds 0,1,2). If
this session ends before the plan is finished, **read this file first**,
then the spec (in the conversation that requested it) for exact formulas.

## Current status (read this first)

- **Full-scale HPO is running right now**: `final_experiments/hpo_criteo.slurm`
  (job 12491537) and `hpo_avazu.slurm` (job 12491538), both submitted,
  `--mem` sized to this cluster's confirmed ~250G node cap. Check with
  `squeue -u $USER` / `sacct -j 12491537,12491538` and read
  `final_experiments/logs/hpo_{criteo,avazu}_<jobid>.out`. When done, each
  writes `final_experiments/{criteo,avazu}/hpo/{hpo_best_fixed,hpo_arw,
  hpo_adamoe,hpo_longterm,hpo_ops,hpo_dualtime}.csv` and
  `selected_configs.json`.
- Everything through the HPO pipeline (methods.py, run_hpo.py,
  leakage_tests.py) is written, unit- and smoke-tested, and committed
  (commits `3eb8f37`, `fb14f81`, `bd294da`; leakage_tests.txt commit
  pending as of this edit). **The headline table is still all TBD** --
  nothing below "primary 3-seed test" in the section-19 checklist has
  been started.
- **Not yet built at all**: the primary 3-seed final-test runner (spec
  section 12 -- reads `selected_configs.json`, runs all 6 methods on the
  locked test days, produces the 12 headline numbers), the rolling-origin
  runner for all 6 methods (section 13, different outer-day ranges than
  the old exploratory `withinday_experiments/rolling/` line -- Criteo
  16-30 = 15 days, Avazu 5-9 = 5 days), day-level stats wiring (section
  14 -- reuse `withinday/daystats.py`, don't reimplement), all output
  tables/figures (sections 17-18), paper text updates (section 20 -- no
  paper file located in this repo yet, ask the user where it lives).
- **Next concrete step once HPO finishes**: write
  `final_experiments/run_final.py` (3-seed locked test, all 6 methods,
  using frozen `selected_configs.json`) -- this is what actually fills in
  the 12 TBD cells.

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

## Suggested resumption order

Follow spec section 19 literally, steps 1-4 first (manifest, full-data
build for both datasets on slurm, 3-seed expert bank, leakage tests) since
nothing after that can be trusted without them. Given compute cost, run
HPO stages (5-10) as separate slurm jobs per dataset, writing each grid's
full table (not just the winner) to `final_experiments/{dataset}/hpo/`.
Do not run the primary 3-seed test (step 12-13) until `selected_configs.json`
is frozen and committed.
