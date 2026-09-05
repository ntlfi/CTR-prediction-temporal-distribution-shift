# Within-day capacity-ladder adapters

Self-contained implementation of
`CTR_Within_Day_Capacity_Ladder_Experiment_Plan.pdf`: a ladder of adapters
(context-query Transformer down to a linear interaction model) that read a
causal per-block within-day history and learn a *heterogeneous* residual
correction on top of the frozen [[twoscale]] long-term predictor -- the
follow-on to that project's finding that a single *global* scalar calibrator
finds ~no within-day signal on Criteo/Avazu.

Builds on `twoscale/` (data loading, day splits, the long-term adaptive
mixture, metrics) rather than duplicating it; everything specific to this
plan lives in `withinday/`.

## Layout

```
withinday/
  contextsketch.py  fixed signed-hash context sketch c(x) (eq 6), by
                     re-hashing twoscale's existing hashed features to width m
  blocks.py         causal block partitioning + maturation clock (eq 3),
                     per-block token u_k (eq 5), deterministic summary s_k
                     (eq 11), chronology-shuffle helper
  cache.py          Stage A: DayCache -- ties q/y/sec_in_day (straight from
                     the twoscale bank) to the block tokens/summaries/current-
                     impression input for a day
  adapters.py       V1 Transformer / V2 GRU / V3 MLP / V4 bilinear / V5 linear
                     (eq 7-15), all zero-initialized (delta==0 at init)
  ablations.py       the 5 required ablations (plan section 7, items 1-5)
  train.py          Stage B training + early stopping; Stage C frozen replay

withinday_run.py              one (dataset, seed) cell: Stage A -> B -> decision
                               rules -> Stage C (locked test opened only if a
                               candidate clears the gates) -> CSV + summary.json
withinday_tests.py            causality / identity / shape checks (35 checks)
withinday_{criteo,avazu}.slurm
```

## Run

```bash
.venv/bin/python withinday_tests.py

# one cell locally on a subsample
.venv/bin/python withinday_run.py --source criteo --sample-frac 0.05 --seed 0 \
    --out withinday_experiments/dev

# full battery
sbatch withinday_criteo.slurm      # 3 seeds, full data
sbatch withinday_avazu.slurm       # 8 seeds, disjoint 20% subsamples
```

## What each run does

For each of V1-V5, trains the normal model and its 5 required ablations
(no-history / shuffled-chronology / no-context-interaction / no-residual-
sketch / label-free-history) on the early development days, early-stopping
on the later development days. Applies the plan's decision rules (section 8)
per candidate -- beats long-only and Online Platt, beats its own no-history
control, shows a chronology-or-context-interaction effect above a margin --
then the parsimony rule (simplest candidate within 1 SE of the best) among
whichever candidates clear all four gates. The locked test days are replayed
**once**, only for the selected candidate, and only if at least one candidate
cleared the gates; otherwise the run stops and reports that within-day
history is not exploitable by this capacity ladder, per the plan's explicit
stop condition.

## Status (2026-09-04): DONE

Implemented, tested (`withinday_tests.py`, 35 checks) and run at full scale
with tuned per-variant hyperparameters (`withinday_hpo.py`): 3 Criteo seeds
(full 16.5M rows), 8 Avazu seeds (20% subsamples). Full write-up in
**`FINDINGS.md`**.

Headline: V5 (the simplest, linear interaction adapter) is selected in 10
of 11 seeds. On Criteo it beats both long-only and Online Platt in all 3
seeds, reproducibly, but never clears the plan's own materiality floor --
a "statistically detectable, operationally negligible" result. On Avazu
the average effect is real but not stable seed-to-seed (1 of 8 seeds
reversed direction on the locked test after clearing every dev gate).
Neither dataset licenses proceeding to downstream extensions as specified.

**Not implemented** (plan gates all of these on clearing the locked-test
bar, which was not met on either dataset):

* Conditional ablations 6-8 (campaign-only history, temporal-granularity
  sensitivity, reset-vs-carryover).
* The theory-friendly frozen-encoder online-regret variant (eq 16-17).
* Downstream autobidding (plan section 11).

`withinday_hpo.py` is also a staged coordinate search, not the plan's full
validation-grid cross product (see its docstring for why) -- a more
exhaustive search is a possible follow-up if this line is revisited.

Related: [[twoscale]] (the long-term backbone and prior scalar-calibration
result this project extends).
