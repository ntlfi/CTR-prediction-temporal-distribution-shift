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

## Status (2026-09-04)

Implemented and tested (`withinday_tests.py`, 35 checks; smoke-tested
end-to-end against real Criteo data on a small subsample -- see the
session that added this file for the exact commands): causal cache, all 5
ladder variants (zero-init identity verified), all 5 required ablations,
decision rules, parsimony selection, locked-test gating.

**Not yet implemented** (plan sections not covered by this first pass):

* `withinday_hpo.py` -- the plan's small dev-only validation grid (section
  5.2: context-sketch dim, hidden dim, lr, weight decay, dropout, bilinear
  rank, correction cap). `withinday_run.py` currently trains every
  candidate at one fixed default config (`withinday.train.DEFAULT_CFG`,
  overridable via `--config`), analogous to only ever using twoscale's
  `FROZEN.json` grid center.
* Conditional ablations 6-8 (campaign-only history, temporal-granularity
  sensitivity, reset-vs-carryover) -- plan section 7 gates these on a
  positive development result, which (consistent with the twoscale finding)
  may not occur.
* The theory-friendly frozen-encoder online-regret variant (eq 16-17) and
  downstream autobidding (gated on beating both baselines on the locked
  test, same gate that kept twoscale's autobid from ever running).
* Full-scale locked-test results (3 Criteo seeds / 8 Avazu seeds via the
  `.slurm` scripts) -- not yet run.

Related: [[twoscale]] (the long-term backbone and prior scalar-calibration
result this project extends).
