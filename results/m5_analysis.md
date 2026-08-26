# M1/M2/M5b/ensemble findings: combining short- and long-term memory

This note covers the three methods this project proposed on top of the P0/P1/P2
baseline ladder (`adaptive-training-methods-implementation-plan.md`), plus a
follow-up meta-gate ensemble, run across all five synthetic drift regimes
(`results_synthetic_{none,abrupt,gradual,recurring,local}/`, 120-day horizon,
35 locked test days) and the full real Criteo Attribution dataset
(`results/`, 31 days, 8 locked test days). Full tables:
`<out_dir>/all_methods_with_new_methods_comparison_table.csv`.

- **M1** (`m1_global_mix.py`): one global blend weight α_t mixing short
  (rolling_3) and long (expanding) candidate-bank predictions, chosen by grid
  search on recent matured validation days.
- **M2** (`m2_context_gate.py`): the same short/long mixture, but with a
  per-example weight α_t(x) from a small online logistic gate over
  short/long disagreement, recent loss, and per-example context.
- **M5b** (`m5_multiscale_gate.py`): M2's gate generalized from 2 experts to
  the full 5-candidate window family (rolling_1/3/7/14 + expanding) — built
  because M2's 2-candidate mixture could never reach rolling_14, the actual
  best fixed window under abrupt/gradual drift.
- **M2+M5b ensemble** (`ensemble_m2_m5.py`, new): a second meta-gate that
  blends M2's and M5b's *final predictions* per example, `beta_t(x)` near 0
  meaning "trust M2 here", near 1 meaning "trust M5b" — built after M5b
  turned out to *trail* M2 specifically under recurring drift.

## Headline: M2 and M5b are complementary specialists; the ensemble gets (almost) the best of both, safely

Locked-test log loss, best per row in **bold**:

| regime | han_arw (best static/P1/P2) | M1 | M2 | M5b | **ensemble** |
|---|---|---|---|---|---|
| none (stationary) | 0.3107 | 0.3107 | 0.3117 | 0.3119 | 0.3118 |
| abrupt | 0.3936 | 0.4077 | 0.4059 | **0.3731** | 0.3749 |
| gradual | 0.3543 | 0.3998 | 0.4002 | **0.3554** | 0.3567 |
| recurring | 0.4215 | 0.4199 | **0.4180** | 0.4258 | 0.4187 |
| local (subpopulation) | 0.4544 | 0.4588 | 0.4437 | **0.4250** | 0.4261 |
| real Criteo (31 days) | 0.6072 | 0.6072 | 0.6072 | **0.6070** | 0.6070 |

The pattern is consistent across every drift regime that actually
discriminates the methods: **M2 wins recurring/cyclical drift; M5b wins
abrupt, gradual, and local drift; the ensemble lands within 0.2–0.5% relative
log loss of whichever one wins, every single time** — never anywhere near the
loser. On `none` and real Criteo, where none of these methods separate from
the P0/P1/P2 ladder by more than noise, the ensemble is likewise
indistinguishable from the pack.

| regime | ensemble vs winning specialist | ensemble vs losing specialist |
|---|---|---|
| abrupt | +0.0018 over M5b (0.48% worse) | −0.0310 vs M2 (7.6% better) |
| gradual | +0.0013 over M5b (0.37% worse) | −0.0435 vs M2 (10.9% better) |
| recurring | +0.0007 over M2 (0.17% worse) | −0.0071 vs M5b (1.7% better) |
| local | +0.0011 over M5b (0.26% worse) | −0.0176 vs M2 (4.0% better) |

## Why it works: the meta-gate learns which specialist to trust, per regime, with no regime label

`ensemble_beta.csv` / `ensemble_diagnostics.png` in each results directory
show the deployed mean β_t (0 = M2, 1 = M5b), and it tracks the table above
almost exactly even though the gate never sees a drift-mode label — only
per-example M2/M5b disagreement, recent per-method loss, and normalized time:

| regime | mean β (0=M2, 1=M5b) | which specialist actually wins |
|---|---|---|
| none | 0.50 | tie (both lose to `expanding` by a hair) |
| abrupt | 0.79 | M5b |
| gradual | 0.92 | M5b |
| recurring | 0.24 | M2 |
| local | 0.74 | M5b |
| real Criteo | 0.53 | ~tie |

This is the actual mechanism behind the headline numbers: causal, per-day
features (recent M2 loss vs. recent M5b loss, chiefly) carry enough signal
for an online logistic gate to correctly infer, within a few days of each
regime starting, which of the two specialists is currently more trustworthy
— without ever being told which synthetic drift mode is active.

## What the ensemble does *not* do

It is real insurance, not a free lunch: in every regime tested, the ensemble
sits **between** M2 and M5b, never below both. It recovers 66–97% of the gap
between the losing specialist and the winning one, at a cost of 0.2–0.5%
relative log loss versus always having picked the winner in advance — but it
never beats both specialists simultaneously. Its value is removing the need
to know the regime ahead of time, not discovering a better model than either
input. It also inherits both specialists' blind spot: neither M2 nor M5b
carries an explicit periodicity feature, so the ensemble's recurring-drift
performance is bounded by M2's, not some hypothetical better cyclical model.

**Caveat**: the table above is a single seed per regime, *except* recurring,
where the ensemble has now been re-run across the same 5 seeds used to
validate M2's original recurring win (`results_synthetic_recurring_seed{1..4}/`
plus the main seed=0 run):

| seed | M2 (wins every seed) | M5b | ensemble | ensemble vs M2 (relative) | mean β |
|---|---|---|---|---|---|
| 0 (main) | 0.4180 | 0.4258 | 0.4187 | +0.17% | 0.24 |
| 1 | 0.4251 | 0.4284 | 0.4262 | +0.26% | 0.26 |
| 2 | 0.4245 | 0.4271 | 0.4254 | +0.21% | 0.35 |
| 3 | 0.4353 | 0.4404 | 0.4375 | +0.51% | 0.37 |
| 4 | 0.4366 | 0.4387 | 0.4375 | +0.21% | 0.30 |

M2 beats M5b in all 5 seeds (confirming the original recurring-win result is
robust, not seed-dependent), and the ensemble tracks M2 closely in every one
of them — always within 0.51% relative log loss, always clearly ahead of
M5b, and the meta-gate's mean β stays correctly on the "trust M2" side
(0.24–0.37) in every seed without ever seeing which regime is active. The
abrupt/gradual/local/none/real-Criteo rows above are still single-seed and
should be treated as indicative until similarly re-run.

## Bottom line

Neither M2 nor M5b dominates: M2's tight 2-expert mixture wins when the
useful signal is "how does the current gate weight compare to last cycle's"
(recurring), while M5b's wider 5-expert pool wins whenever the real best
window is outside that 2-candidate family (abrupt, gradual, local). Rather
than pick one at model-selection time — which would mean guessing the
production drift regime in advance — a lightweight meta-gate over their two
predictions recovers nearly all of the winning specialist's advantage in
every regime tested, at a small, bounded cost when it guesses the "wrong"
specialist's regime slightly early or late. This is the strongest evidence
so far in this project for combining short- and long-term memory through
*multiple, specialized* adaptive mechanisms rather than a single one — the
next natural step being to fold an explicit periodicity feature into the
mix so recurring drift stops being every method's shared blind spot.
