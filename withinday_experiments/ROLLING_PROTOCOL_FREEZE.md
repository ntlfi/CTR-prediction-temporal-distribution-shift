# Frozen protocol: day-level rolling-origin evaluation

Written and committed **before** running any of the new analysis, per the
protocol's own requirement (its section 1) to freeze choices in advance.
Everything below is locked; results are not permitted to change it.

## Day accounting (computed once, before any new modeling)

`twoscale.splits.make_split` on the datasets already used gives:

| dataset | n_days | train (base-model history) | dev | test (original locked) | untouched |
|---|---|---|---|---|---|
| Criteo | 31 | 0-15 (16 days) | 16-21 (6 days) | 22-30 (9 days) | **0** |
| Avazu | 10 | 0-4 (5 days) | 5-6 (2 days) | 7-9 (3 days) | **0** |

Both datasets are fully accounted for -- 16+6+9=31 and 5+2+3=10. This
resolves section 4's branch **before** looking at any new-day results:
**no untouched Criteo days exist.** Consequences, fixed now:

- Criteo: the original 9-day locked test (days 22-30) stays the primary
  confirmatory result, unchanged, in its existing files. The new analysis
  is a **"Post-selection rolling-origin stability analysis"** over the 6
  original *dev* days (16-21) -- eligible, non-locked, never used to fit
  or score the *final* V5 architecture choice from the earlier session
  (they were used to fit the long-term base predictors and to pick V5's
  hyperparameters *once*, globally -- see caveat below).
- Avazu: per section 2, rolling-origin evaluation covers every day from
  the earliest one the existing minimum-history rule (`warmup=3` ->
  actual `n_train=5`) permits through the end of the stream: **days
  5,6,7,8,9** (5 outer test days -- the original dev+test split boundary
  is dissolved into one continuous rolling sequence, exactly what section
  2 asks for). Days 0-4 remain pure history, never scored, matching "do
  not shorten required historical windows."

**Caveat on "untouched," stated plainly:** the *code* (block/summary
construction, the 12-config V5 grid, the ablation set) was designed and
its viability confirmed by looking at aggregate results on these same
dev/test days in the prior session. This new analysis cannot claim a
never-before-seen test set for either dataset -- it is a stability check
of an already-fixed method, not a fresh confirmatory holdout. The
manifest below records this honestly rather than overstating it.

## Frozen choices (locked before any new-day results are inspected)

- **V5 feature construction:** unchanged -- `withinday/blocks.py` block
  token (eq 5) + deterministic summary (eq 11), `withinday/contextsketch.py`
  signed-hash sketch.
- **Block length / feedback delay:** `block_sec=900` (15 min),
  `delay_sec=1800` (30 min) -- plan defaults, as used throughout.
- **History reset/carryover:** reset every day (no cross-day state for V5's
  block history; `online_platt`'s intercept/slope also resets daily,
  `carryover_rho=0.0`, matching every prior run).
- **Clipping / regularization:** `eps=1e-5` logit clip; `lam_delta=0.0`;
  weight decay and dropout come from whichever grid config is selected
  (see below) -- no new regularization introduced.
- **Candidate hyperparameter grid:** *exactly* the existing frozen V5 grid
  from `withinday_hpo.py` -- `cross_dim in {16,32,64} x lr in {3e-4,1e-3}
  x weight_decay in {1e-5,1e-4}` = 12 configs. Not touched.
- **Model-selection rule:** lowest mean inner-validation impression-weighted
  log loss; ties within 1 SE of the best broken toward the smallest
  `cross_dim` (the ladder's only complexity axis inside V5).
- **Materiality threshold:** `2e-4`, unchanged.
- **Code commit:** `b8c6649` (the commit this file is added on top of).
  Every rolling-origin manifest row also records this hash directly.
- **Data processing:** identical loaders/args as the original locked-test
  runs -- Criteo `sample_frac=1.0`, Avazu `sample_frac=0.2` with a single
  fixed data-sampling seed (`0`, i.e. exactly seed 0's subsample from the
  original 8-seed sweep) used for **every** outer day and every method in
  this analysis. Model-initialization seed (`torch.manual_seed`, fixed at
  `0` for the main sweep) is tracked as a **separate** column from the
  data-sampling seed in every output file, per the instruction not to
  conflate the two. A small model-seed sensitivity check (seeds 0/1/2 on
  the final selected configs only, not the full nested search) is run
  separately and reported as a range, not as extra "days."

## Necessary new choices not specified by the protocol (made now, documented)

1. **`long_only` and `online_platt` need no new "rolling" machinery.**
   `twoscale.longterm.build_bank` already fits every candidate on
   `day < d` only for each `d` in whatever `eval_days` it is given, and
   `twoscale.methods.build_suite` replays days in order with the
   calibrator resetting daily. Calling these once over the *full* eligible
   day range (Avazu 0-9, Criteo dev range 0-21) reproduces exactly the
   same causal, day-by-day predictions a rolling-origin loop would compute
   one day at a time -- so only V5 needs genuinely new per-outer-day
   fitting logic.
2. **Inner validation window is capped, not exhaustive.** The protocol
   asks for inner rolling-origin validation "on eligible days before d,"
   averaged equally. An exhaustive expanding-window inner CV (every day
   before d, each its own fold) grows the training-set size and the fold
   count together as `d` increases, and V5 training cost scales with
   training-day count -- full exhaustion would cost several times more
   than the entire original multi-seed HPO sweep. Inner validation instead
   uses the **3 most recent eligible days strictly before `d`** (fewer if
   `d` is early in the sequence) as validation folds, each fold's model
   trained on all *its own* strictly-earlier days. This keeps the total
   grid search tractable (bounded per outer day regardless of how large
   `d` gets) while still averaging over multiple folds rather than one.
   Recorded explicitly in `rolling_origin_manifest.csv`.
3. **No-history / shuffled-history controls are evaluated, not retrained,
   per outer day.** Retraining under every ablation for every outer day
   would roughly quadruple the already-large nested-search cost. Instead,
   for the one model actually deployed on day `d` (fit on all days `< d`
   at the selected config), the no-history and shuffled-history controls
   zero/permute its block-token input **at prediction time only** and
   reuse the same trained weights. This tests the deployed model's
   reliance on real, chronological history; it is a weaker (cheaper)
   check than retraining a dedicated ablated model per day, and is
   reported as such.
4. **Inner-fold adapter-train/adapter-dev split.** Each inner fold and
   each final per-outer-day refit needs its own early-stopping split
   inside "all days `< d`" (or `< v` for an inner fold): the single most
   recent of those days is held out for early stopping, the rest are
   adapter-training days (mirrors the original run's 70/30 convention,
   simplified to "last day held out" since these windows are short).
5. **Post-selection Criteo analysis reuses train-period days (0-15) as
   inner-validation folds when needed.** For early outer days (16, 17),
   there are few or no *dev*-period days strictly before them to validate
   on; the 3-most-recent-eligible-day rule pulls from the tail of the
   train period (days 13-15) in that case. Those days were already used
   to fit the long-term base predictors (a different modeling decision
   from V5's own hyperparameters), so this does not leak information
   about the choice being validated.

## What this analysis cannot do

With only 5 (Avazu) and 6 (Criteo, post-hoc) outer days, day-level
inference is necessarily descriptive more than asymptotic -- per the
protocol's own section 5, day-level patterns are emphasized over
significance claims, and the moving-block bootstrap is reported only
where there are enough days for it to mean anything (documented per-run
if skipped).
