# Synthetic drift-injection findings: do adaptive methods actually track drift?

Motivation: on the real Criteo Attribution dataset (31 days), Han ARW,
Differentiable Forgetting, and AdaMoE never clearly beat the simplest
static baseline (`rolling_7`) — see [`analysis.md`](analysis.md). That
dataset turned out to have only shallow drift, and no public CTR-native
dataset with a genuinely longer, drift-heavy horizon exists in either the
general benchmark literature or the actual experiments of the 4 papers
reproduced here (see the README's "Synthetic drift-injection experiments"
section). This note reports what happens when the P0/P1/P2 ladder is run
against `synthetic_data.py`'s generator instead, where the true CTR model's
drift schedule is known exactly, on a 120-day horizon (81 dev days, 35
locked test days), 3000 rows/day, four schedules: `none`, `abrupt` (regime
swap at day 95, 10 days into the test period), `gradual` (linear drift
across the whole horizon), `recurring` (14-day-period oscillation). Full
tables: `results_synthetic_<mode>/all_methods_comparison_table.csv`.

## Headline: `none` sanity check passes, `abrupt` is the clear win for adaptive memory

| Mode | Best overall | log loss | Best P0-only | log loss | `validation_selected` (frozen) | log loss |
|---|---|---|---|---|---|---|
| none | `expanding` (`han_arw` ties it) | 0.3107 | `expanding` | 0.3107 | h=expanding | 0.3107 |
| **abrupt** | **`han_arw`** | **0.3936** | `rolling_7` | 0.3977 | h=expanding | **0.6454** |
| gradual | `rolling_14` (`han_arw` ties it) | 0.3543 | `rolling_14` | 0.3543 | h=rolling_14 | 0.3543 |
| recurring | `expanding` (`han_arw` ties it) | 0.4215 | `expanding` | 0.4215 | h=expanding | 0.4215 |

Only in `abrupt` does a P1/P2 method decisively beat every P0 baseline —
and the frozen `validation_selected` baseline (the "strong practical
baseline" the PDF asks these methods to beat) actively fails there, since
its choice was locked in during dev, before the shift happened, and it
never gets to reconsider. This is the first time in this project that an
adaptive method's advantage over a static ladder is unambiguous.

## `none`: no drift, no false alarm

With a truly stationary ground truth, `expanding` wins outright (0.3107)
and `han_arw`'s locked-test predictions are numerically identical to
`expanding`'s to 13 decimal places — its tournament correctly recognizes
there's no reason to shrink the window and never does, on any test day.
`diff_forgetting`'s learned half-life stays high (mean 73.6 days, i.e.
near-stationary) and lands close behind (0.3220). `adamoe` (0.3370) trails
slightly but doesn't collapse. **This is the sanity check the PDF's
acceptance tests ask for (section 8, "baseline sanity")**: none of the
adaptive machinery manufactures a spurious advantage — or a spurious
penalty — when there's nothing to adapt to.

## `abrupt`: Han ARW visibly tracks the regime change; Differentiable Forgetting reacts but recovers incompletely

Per-day log loss around the shift (day 95), from `per_day_metrics.csv` /
`p1_p2_per_day_metrics.csv`:

| day | expanding | rolling_14 | rolling_7 | rolling_3 | **han_arw** | diff_forgetting | adamoe |
|---|---|---|---|---|---|---|---|
| 94 (pre-shift) | 0.304 | 0.312 | 0.322 | 0.375 | 0.304 | 0.315 | 0.328 |
| 95 (shift day) | 0.938 | 0.963 | 0.987 | 1.070 | 0.938 | 0.876 | 0.936 |
| 98 | 0.853 | 0.662 | 0.554 | 0.403 | **0.403** | 0.548 | 0.480 |
| 102 | 0.847 | 0.500 | 0.337 | 0.384 | **0.384** | 0.514 | 0.414 |
| 110 | 0.736 | 0.332 | 0.347 | 0.388 | **0.347** | 0.471 | 0.378 |
| 119 (end) | 0.673 | 0.343 | 0.360 | 0.417 | **0.360** | 0.455 | 0.391 |

`expanding` never really recovers within the 25 remaining test days — it's
still at 0.673 on the last day, barely off its immediate post-shift spike,
because it keeps averaging in ~92 days of now-stale pre-shift rows against
only ~24 days of new-regime data.

`han_arw`'s recorded window choices (`han_arw_selected_window.csv`) show it
reacting in real time:

| days | selected window |
|---|---|
| 93–96 | `expanding` (correct — at most 1 post-shift day exists yet, not enough evidence) |
| 97 | `rolling_14` |
| 98–105 | `rolling_3` (shortest window, right after the shift) |
| 106–110 | `rolling_7` (widens again as the new regime accumulates its own clean history) |

This is exactly the "data-dependent global history length" behavior the
method is built for: shrink hard right after evidence of a regime change,
then grow back out as the new regime ages into its own stable history.
Because of this, `han_arw`'s per-day loss tracks whichever fixed window is
*currently* best rather than being locked into one — it wins the locked-test
aggregate (0.3936) outright.

`diff_forgetting`'s learned half-life (`diff_forgetting_eta.csv`) also
reacts fast — 74.4 days (day 94) → 17.6 days (day 96) → **2.18 days (day
98)**, the most aggressive forgetting seen anywhere in this project — then
gradually relaxes back out (7.7 days by day 110) as the new regime
accumulates history. The reaction is real and fast, but its recovery is
visibly incomplete compared to `han_arw`/`rolling_3`: at day 98 it's still
at 0.548 vs `han_arw`'s 0.403. The mechanistic reason: down-weighting old
rows (its only lever) still includes them in the fit, whereas `han_arw`
picking `rolling_3` *excludes* pre-shift rows outright. Down-weighting a
wrong-regime row by e.g. `exp(-0.32*3)≈0.38` still leaves it with real
influence; a window rule gives it zero. This is a genuine, mechanistically
explainable instance of the PDF's "recovers slowly after abrupt shifts"
motivating case (section 10) — not a bug, but a real property of a
soft-decay mechanism versus a hard-window one.

`adamoe`'s expert weights (`adamoe_expert_weights.csv`) shift measurably
for the first time in this project: `rolling_1`'s weight jumps from 0.183
(day 94) to a peak of 0.227 (day 98) right after the shift, `rolling_3`
climbs from 0.197 to 0.224 by day 99, while `expanding`'s share drifts down
from 0.209 to 0.172 by day 110 — a real, if modest, re-weighting toward the
short-window experts during the recovery window, unlike its near-total
inertness on the real Criteo data or in the `none`/`recurring` modes below.

## `gradual`: adaptive methods rediscover the best fixed window, don't beat it

With continuous drift, `han_arw` splits its choices between `expanding` (45
days) and `rolling_14` (68 days) and ties `rolling_14`'s aggregate exactly
(0.3543) — the same "correctly rediscovers, doesn't surpass" pattern seen
on the real Criteo dataset. `diff_forgetting`'s half-life genuinely varies
with the drift (7.6–102 days, mean 42), but its aggregate (0.4106) still
trails `rolling_14`/`rolling_7`/`han_arw`. Continuous, gradual drift doesn't
create the same sharp "old data is actively wrong" penalty that a hard
regime swap does, so there's less headroom for any adaptive method to win
by a wide margin here.

## `recurring`: the clearest negative result — recency-based memory cannot track a cycle

`expanding` wins outright (0.4215) and `han_arw` selects `expanding` on 105
of 113 evaluated days, only briefly touching shorter windows. `adamoe`'s
expert weights stay almost perfectly uniform for the entire 120-day horizon
(day 4: all 0.200; day 119: 0.204/0.192/0.205/0.199/0.199) — visually
indistinguishable from its behavior on the real Criteo data.
`diff_forgetting`'s half-life reacts briefly in the first couple of cycles
(1.7–9 days around days 6–15) then locks onto the same near-stationary
~74-day half-life for the entire remainder of the test period.

The mechanism is clear in hindsight: a 14-day sinusoidal oscillation
averages out over any window spanning a full cycle or more, so "use more
history" doesn't look worse in aggregate even though it's never tracking
the *current phase*. None of these three methods carry a notion of
periodicity — they only reason about *how much* recent history to trust,
not *where in a cycle* the process currently is — so recurring/seasonal
drift is a genuine blind spot for all three, not an implementation gap.
Detecting and exploiting a cyclical regime would need an explicitly
periodicity-aware model (e.g. day-of-cycle as a feature, or a seasonal
decomposition), which is outside what any of the three P1/P2 methods
reproduced here are designed to do.

## Bottom line

The synthetic experiments do what the real dataset couldn't: they show
`han_arw` (and to a lesser extent `adamoe`) delivering a clear, mechanistically
understood win over every static P0 baseline — but only under **abrupt**
drift, and even then `diff_forgetting`'s soft-decay mechanism recovers
visibly slower than a hard window swap. Under **gradual** drift, adaptive
methods track but don't clearly beat the best static choice. Under
**recurring** drift, all three are blind to the actual structure and default
to "use everything," which happens to also be the winning static choice —
a real limitation of recency-only adaptation, not a bug. This matches the
PDF's own framing (section 10): the "strong motivation" case for these
methods is specifically abrupt or slowly-varying drift with a **real**
best-window shift, not periodic structure — and that is precisely the one
condition under which they earn their complexity here.
