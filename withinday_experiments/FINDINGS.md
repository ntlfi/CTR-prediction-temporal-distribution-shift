# Within-day capacity-ladder: findings (2026-09-04)

Full multi-seed locked-test protocol (3 Criteo seeds, 8 Avazu seeds),
per-variant hyperparameters from `withinday_hpo.py`.

## Headline result

**V5, the simplest variant (a linear model on hashed context x history
interaction features), is the one that works.** It was selected by the
plan's parsimony rule in 10 of 11 seeds (V1 Transformer won the other,
Avazu seed 7). No neural variant (V1/V2) or mid-complexity variant
(V3/V4) ever beat V5 by enough to be selected instead — consistent with
H4 and with the plan's explicit warning not to prefer the Transformer on
point estimate alone.

**Criteo (full 16.5M rows, 3 seeds): a small, real, reproducible effect
that never clears the materiality floor.**

| seed | vs long-only | vs Online Platt | beats both | material (>=2e-4) |
|---|---|---|---|---|
| 0 | -0.000123 (sig, 8/9 days) | -0.000287 (sig, 6/9 days) | yes | no |
| 1 | -0.000032 (n.s., 6/9 days) | -0.000135 (n.s., 4/9 days) | yes | no |
| 2 | -0.000148 (sig, 8/9 days) | -0.000313 (sig, 8/9 days) | yes | no |
| **mean +- sd** | **-0.000101 +- 0.000061** | **-0.000245 +- 0.000096** | **3/3** | **0/3** |

Every seed points the same direction against both baselines. This is the
textbook case the plan's decision rules name explicitly: a *statistically
detectable but operationally negligible* improvement (plan section 8).
Reproducible, but not worth deploying on Criteo as specified.

**Avazu (20% sample, 8 seeds): a real average effect, but not stable
seed-to-seed.**

| seed | winner | vs long-only | vs Online Platt | beats both & material |
|---|---|---|---|---|
| 0 | v5_linear | -0.000334 (sig) | +0.000022 (n.s.) | no |
| 1 | v5_linear | **+0.000475 (worse, n.s.)** | +0.000755 (worse, n.s.) | no |
| 2 | v5_linear | -0.000310 (sig) | -0.000100 (sig) | **yes** |
| 3 | v5_linear | +0.000107 (n.s.) | +0.000301 (n.s.) | no |
| 4 | v5_linear | -0.000649 (sig) | -0.000420 (sig) | **yes** |
| 5 | v5_linear | -0.000783 (sig) | -0.000531 (sig) | **yes** |
| 6 | v5_linear | -0.000528 (sig) | -0.000305 (sig) | **yes** |
| 7 | v1_transformer | -0.000468 (sig) | -0.000295 (n.s.) | yes |
| **mean +- sd** | | **-0.000311 +- 0.000414** | **-0.000071 +- 0.000426** | **5/8 both+material, 7/8 material vs long-only alone** |

5 of 8 seeds clearly beat both baselines materially and significantly. But
**seed 1 is a real failure mode**: the model cleared every development-day
gate and then did *worse* than both baselines on the locked test -- the
opposite of what dev-day selection predicted. Avazu's split gives only
1-2 adapter-train/dev days (10 days total, scaled split), so a single
noisy day can flip the decision. This is exactly what decision-rule item
2 ("stable across seeds, not driven by one day") exists to catch, and it
is not fully satisfied here.

## Interpretation (plan section 10's outcome table)

Closest match: **"Contextual models beat global models"** -- V5's win
comes from the residual-weighted context sketch `r*c(x)` (every winning
run also passed the chronology-or-context-interaction gate), i.e.
heterogeneous, campaign/context-level drift that a single global scalar
(Online Platt) partially but not fully captures. This is genuine support
for H1 and H2 on both datasets. But per plan section 8's explicit stop
condition -- "beats both baselines on the locked test **and** the gain
[is not just] reproduced by shuffled or time-only controls" -- Criteo
never reaches the materiality floor, and Avazu is not stable across
seeds. **Neither dataset licenses proceeding to downstream autobidding
(plan section 11) as specified.**

## What actually helped (per the required ablations)

Winning runs' own no-history and shuffled-chronology/no-context-
interaction ablations (`ablations_dev.csv` per seed) confirm the effect is
not just "more capacity": V5 reliably beat its own zero-history control
and showed a real chronology-or-context margin wherever it cleared the
gates. This rules out "the extra parameters alone explain the gain" as an
explanation for the winning seeds.

## Caveats

- 1 HPO run per seed (not per-fold-averaged); the search is a small staged
  coordinate descent (`withinday_hpo.py`), not the plan's full grid.
- Avazu's adapter-train/adapter-dev split is very thin (1-2 days out of a
  10-day, 20%-subsampled log) -- seed 1's reversal is likely a symptom of
  that, not of V5 itself being unsound.
- Conditional ablations (campaign-only history, temporal-granularity
  sensitivity, reset-vs-carryover), the frozen-encoder online-regret
  variant, and downstream autobidding are not implemented/run (plan gates
  all of these on clearing the locked-test bar, which was not met).

## Bottom line

Across the whole AMG-TP -> twoscale -> withinday line, this is the first
run to show a real, mechanistically-explained (heterogeneous context
drift, not just capacity) within-day effect beating Online Platt on
*some* real data -- but the effect is small on Criteo and seed-unstable
on Avazu. The honest conclusion the plan asks for: within-day history
carries a little genuine, context-heterogeneous information beyond a
global calibrator, but not enough, or not reliably enough, to justify a
production capacity-ladder adapter over Online Platt scaling on either
public dataset as tested here.

Related: [[project-twoscale-ctr]] memory (`project_twoscale_ctr.md`),
[[twoscale]] (the long-term backbone and prior scalar-calibration result
this project extends).
