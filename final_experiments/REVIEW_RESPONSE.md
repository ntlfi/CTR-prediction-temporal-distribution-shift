# Response to the three pre-decision review comments

Status of the three requested items before deciding whether to keep the
current algorithm (online DualTime-CTR). All code lives in
`final_experiments/`; nothing here re-opens a fresh locked test.

---

## 1. Fix the statistical analysis (day is the unit, not the seed-day)

**Done.** `final_experiments/day_level_stats.py` re-does every significance
statement the way the comment asks:

```
Lbar_{m,d} = (1/3) sum_s L_{m,d,s}          # average the 3 seeds FIRST
```

then bootstrap / sign-test **across calendar days** (Criteo D=9, Avazu
D=3), reusing `withinday.daystats.day_summary` (percentile bootstrap,
two-sided sign test, leave-one-day-out, moving-block bootstrap). It reads
the per-seed `per_day_metrics.csv` files the runs already wrote, so no
recompute. Report: `final_experiments/DAY_LEVEL_STATS.md`.

The headline table's "27/27 seed-days" / its pooled CI is **retired as the
significance statement** and replaced by:

- day-level bootstrap CI + sign test over the D origin days (fixed origin), and
- the rolling-origin run below (more origins), which is now the primary
  temporal-significance evidence.

### Fixed origin (`DAY_LEVEL_STATS.md`)

- **Criteo, D=9:** all 5 adaptive methods beat Expanding on **9/9 days**,
  day-level bootstrap CI excludes 0, sign p = 0.004 (the D=9 floor).
  Cross-seed spread of each method's day-mean delta is 1e-5-9e-5 vs an
  effect of 8e-4-1.1e-3 -- seed is ~20x smaller than day, which is exactly
  why pooling them was wrong but also why the conclusion survives.
- **Avazu, D=3:** no day-level test can reach significance (3/3 sweep =>
  p = 0.25). Descriptive only.

### Rolling origin (`DAY_LEVEL_STATS_ROLLING.md`, `run_rolling.py`)

Walk-forward, per-origin re-selection of mixture / best-fixed / ARW /
AdaMoE (OPS `(B,eta0,sched)` and DualTime `B_w` held at the frozen values
-- both reset daily, so a day's prediction only depends on that day's q;
documented in the runner). The per-origin knob choices *do* move (Criteo
origins pick mix-eta in {30, 150, 1e6}, h* in {roll3, roll7}), so this is
a real robustness check.

**Criteo (D = 15 origins):**

| method | Δ vs Expanding (day-wt, 95% CI) | W-L-T | Δ vs long_only | vs long_only sig? |
|---|---|---|---|---|
| **ops** | −0.001174 [−0.001369, −0.001007] | 15-0-0 | **−0.000257** [−0.000353, −0.000164] | **yes** (15/15, p 6e-5) |
| dualtime | −0.001058 [−0.001251, −0.000893] | 15-0-0 | −0.000140 [−0.000227, −0.000058] | yes but weak (12/15, p 0.035) |
| adamoe | −0.000934 | 15-0-0 | −0.000017 | no (7-8) |
| long_only | −0.000918 | 15-0-0 | — | — |
| arw | −0.000843 | 15-0-0 | +0.000074 | *worse* than long_only |
| best_fixed | −0.000755 | 14-1-0 | +0.000162 | *worse* than long_only |

**Avazu (D = 5 origins; sign-test floor p = 0.0625):**

| method | Δ vs Expanding (day-wt, 95% CI) | W-L-T | Δ vs long_only | 
|---|---|---|---|
| **ops** | −0.000458 [−0.000763, −0.000058] | 4-1-0 | −0.000440 [−0.000850, −0.000140], 5/5 |
| dualtime | −0.000349 [−0.000668, **+0.000106**] | 4-1-0 | −0.000331 [−0.000675, −0.000096], 5/5 |
| **adamoe** | −0.000335 [−0.000550, −0.000151] | **5-0-0** | −0.000317 [−0.000952, +0.000000] |
| long_only | −0.000018 | 4-1-0 | — |

On Avazu, AdaMoE is the only method that beats Expanding on all 5 origins
with a CI excluding 0 (p = 0.0625, the best attainable at D=5). OPS and
DualTime beat the (per-origin-selected) `long_only` on all 5 origins but
their Expanding CIs are wider; DualTime's Expanding CI **crosses 0**.

**Rolling-origin verdict:** the adaptive-training gain over Expanding is
rock-solid on Criteo (15/15, p 6e-5) and real-but-underpowered on Avazu
(D=5). Across every origin on both datasets, **OPS ≥ online DualTime-CTR**
(Criteo −0.000257 vs −0.000140 vs long_only; Avazu −0.000440 vs
−0.000331). DualTime's within-day residual module adds nothing a global
online intercept doesn't already get.

## 2. What eta = 1e6 actually does on Avazu

The mixture weight rule is

```
w_h  ∝  exp{ -eta * (Lbar_h - min_j Lbar_j) }
```

(`twoscale/longterm.py::adaptive_weights`). With `eta -> inf` every
non-argmin horizon gets `exp(-large) -> 0`: this is **follow-the-leader /
winner-take-all over the 3 horizons**, the opposite of equal weighting
(which is `eta -> 0`).

Empirically confirmed on full Criteo (`criteo/diagnostic/seed*/mixture_weights.csv`):

| eta          | realised daily weights (roll3, roll7, expanding) |
|--------------|--------------------------------------------------|
| 0 (equal)    | (0.333, 0.333, 0.333) every day |
| 150 (Criteo selected) | genuine blend, ~(0.34, 0.35, 0.31) every day |
| **1e6**      | **(≈0, 1, ≈0) — a hard one-hot on roll7 every day** |

So Criteo's selected mixture *blends*; Avazu's selected mixture
(`eta=1e6`) does *hard daily selection*.

So the write-up line *"1e6 (degenerate — approximately equal weighting)"*
is **wrong** and is corrected in `FINDINGS.md` / `PROGRESS.md`. The
correct reading: on Avazu, dev loss preferred **hard daily selection of the
single best horizon** over any blend (dev loss 0.42168 at eta=1e6 vs
0.42200 for softer eta; all three half-lives identical at eta=1e6, the
signature of a hard argmax where discounting is irrelevant).

Avazu realised `eta=1e6` weights per test day (`avazu/diagnostic/seed0/mixture_weights.csv`):
day 5 uniform (no history), day 6 → 100% roll3, day 7 → 100% roll3,
day 8 → 50/50 roll7/expanding, day 9 → 97% roll3. Hard daily selection
that flips between the short (roll3) and medium (roll7) window — genuinely
adaptive *selection*, and it favours the short window, consistent with
faster drift on Avazu. It is emphatically **not** `(1/3, 1/3, 1/3)`.

## 3. Frozen V5 vs online DualTime diagnostic

**Development evidence, not a fresh locked test** (the test days have been
inspected). `final_experiments/run_diagnostic.py` runs four arms under the
exact final protocol (full data, 3 seeds, same shared bank, same frozen
`q_{d,i}`): `long_only`, `ops`, `dualtime` (online), `frozen_v5` (offline-
trained on dev days, frozen for the test period; `withinday.adapters.V5Linear`
on a `withinday.cache` built from this protocol's q).

### Criteo (full data, 3 seeds) — `final_experiments/criteo/diagnostic/`

| arm | mean imp-wt log loss | day-level Δ vs `long_only` (D=9) | vs `long_only` sig? |
|---|---|---|---|
| **ops** | **0.606958** | −0.000185  [−0.000297, −0.000072], 8/9 | **yes** |
| dualtime (online) | 0.607070 | −0.000076  [−0.000167, +0.000015], 6/9 | no |
| frozen_v5 (offline) | 0.607087 | −0.000057  [−0.000098, −0.000014], 7/9 | borderline (sign p 0.18) |
| long_only | 0.607145 | — | — |

Head-to-head (day-level, D=9): `ops` beats `dualtime` by +0.000109
[+0.000070, +0.000152], DualTime winning **0/9** days (sign p 0.004);
`ops` beats `frozen_v5` by +0.000128 [+0.000021, +0.000236], 1/9.

**On Criteo, `L(frozen_v5) ≈ L(dualtime)`, both a hair *worse* than the
online arm's own headline number and both clearly beaten by plain OPS.**
So the comment's "if `L(V5) < L(DualTime)` the bottleneck is the online
conversion" branch **does not fire on Criteo**: offline-frozen and online
land in the same place. The reading is the stronger one — there is no
within-day contextual residual signal here for *either* φ(x,h) model to
extract; Criteo's entire adaptive-training gain is cross-day horizon
mixing (`long_only`) plus a global online intercept (`ops`). This matches
every prior Criteo result in the repo (`twoscale` `combined ≈ long_only`,
`withinday` sub-materiality, `withinday_experiments` V5 below the
materiality floor).

### Avazu (full data, 3 seeds) — `final_experiments/avazu/diagnostic/`

| arm | mean imp-wt log loss (± seed sd) | day-level Δ vs `long_only` (D=3) |
|---|---|---|
| **frozen_v5 (offline)** | **0.387406 ± 0.000178** | −0.000684, 3/3 |
| ops | 0.387443 ± 0.000058 | −0.000654, 3/3 |
| dualtime (online) | 0.387610 ± 0.000077 | −0.000494, 3/3 |
| long_only (FTL mixture q) | 0.388154 ± 0.000203 | — |

Head-to-head (day-level, D=3 — **not significant, descriptive only**):
`ops` beats `dualtime` +0.000160, DualTime 0/3 days; `frozen_v5` vs `ops`
−0.000030 (frozen_v5 2/3, a wash).

**On Avazu, `L(frozen_v5) = 0.387406 < L(dualtime) = 0.387610` — the
comment's diagnostic condition holds here.** The offline-frozen contextual
residual model matches OPS and edges `long_only`; the online daily-reset
OGD version (`dualtime`) is ~0.0002 worse and loses every day to OPS. So
on Avazu the bottleneck *is* plausibly the conversion of V5 into the
theory-friendly online learner (w reset to 0 each morning), not the
within-day information.

Two cautions that keep this from being a green light:
1. **D = 3, nothing is significant**, and `frozen_v5`'s cross-seed sd
   (0.000178) is ~3× the other arms' — V5 offline training is itself
   unstable at this data size.
2. `frozen_v5` does **not beat plain OPS** (−0.00003, a wash). The best
   contextual model available only *matches* global scalar calibration.
3. `long_only` here (0.388154) is *worse* than plain Expanding (0.387596)
   — Avazu's `eta=1e6` FTL mixture q is volatile (it flips roll3→roll7→roll3
   across the 3 test days) and it is the downstream calibration, not the
   mixture, doing the work.

### Reading across both datasets

| | Criteo | Avazu |
|---|---|---|
| `L(frozen_v5)` vs `L(dualtime)` | ≈ equal (V5 a hair worse) | **V5 better by ~2e-4** |
| best contextual arm vs OPS | OPS wins by 1.3e-4 (sig) | wash (−3e-5, ns) |
| significance | D=9, clean | D=3, none |

Criteo says "no within-day contextual signal for either model."
Avazu *hints* that offline > online for the contextual model, i.e. the
online conversion loses something — but only to the point of matching OPS,
never beating it, and not significantly. Neither dataset shows the current
online DualTime-CTR adding value over OPS.


The comment's boxed refinement — warm-started OGD (`w_{d,1} =
w_historical`, then projected OGD within the day) instead of `w_{d,0}=0`
every morning; the regret bound still holds with a non-zero start, its
constant just becomes the initial distance to the comparator — is the
right next step *if* the within-day contextual model is worth keeping at
all. **Not implemented / not evaluated here** on purpose: per the
comment's caution, it must be confirmed on a genuinely untouched
chronological stream, not these (now-inspected) days.

---

## Recommendation: do not keep online DualTime-CTR as the headline method

Putting the three items together:

1. **Online DualTime-CTR never beats OPS.** 0/9 Criteo days and 0/3 Avazu
   days in the diagnostic; confirmed by rolling-origin (Criteo Δ vs
   long_only −0.000140 for DualTime vs −0.000257 for OPS; Avazu −0.000331
   vs −0.000440). It is a strictly more complex method (contextual
   φ(x,h), online projected-GD over a vector `w`) that loses to a
   2-parameter online scalar.
2. **On Criteo there is no within-day contextual signal at all** — frozen
   offline V5 ≈ online DualTime, both below OPS. The Criteo adaptive-
   training gain is entirely cross-day (horizon mixing) + a global online
   intercept.
3. **On Avazu, `L(V5) < L(DualTime)`**, so the online conversion *is* the
   thing losing information there — but even the offline model only
   *matches* OPS, on D=3/D=5 with no significance and unstable V5 training.

**Defensible headline: OPS** (simple, best-or-tied everywhere, day-level
significant vs both Expanding and long_only on Criteo). Two honest ways to
write the paper:

- **(a) Negative result on the online contextual model.** Keep DualTime-CTR
  in the paper as the theory-friendly online instantiation of the
  within-day idea and report that it does not extract more than a global
  online scalar on either dataset — with the Avazu offline-vs-online gap
  as the diagnostic for *why* (daily `w<-0` reset discards a useful
  historical prior).
- **(b) Pursue the warm-start refinement** (`w_{d,1}=w_historical` +
  projected OGD) and validate it on a **new** chronological stream before
  any headline claim. Only Avazu currently hints it would help, and that
  hint is not significant.

Either way the current `w_{d,0}=0` online DualTime-CTR should not be
presented as beating the baselines, because on this evidence it does not.
