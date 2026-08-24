# SFTL findings: a faithful reproduction that underperforms here, and why

`sftl.py` reproduces Zhu et al.'s SFTL (AAAI 2024) -- three copies of one
model (working/slow/fast learner) coupled by a bipartite-ranking
"trajectory loss," with the fast (EMA) learner served at inference. Unlike
Han ARW / Differentiable Forgetting / AdaMoE, it is not built on the
existing `SGDClassifier` window-family infrastructure: it needed a genuine
neural model (embedding + MLP) and its own continuous streaming-training
loop across the full day range. Same 120-day, four-drift-mode synthetic
suite as [`synthetic_analysis.md`](synthetic_analysis.md): `--epochs-per-domain
5 --batch-size 256`, 3000 rows/day.

## A real bug was found and fixed before any of this was measurable

The paper's trajectory loss (eq. 9) has no natural floor: since the slow
learner is a hard copy of the working learner at each domain boundary,
"beat the slow learner's margin" becomes "beat your own slightly-more-
confident past self," every single domain -- a positive feedback loop with
no equilibrium. At the paper's own undisclosed loss weights guessed as
`lambda_slow = lambda_fast = 1.0`, this diverged outright: predicted
probabilities escalated toward 0/1 within a handful of domains, and log
loss on a small diagnostic run hit **4.80** before locked-test evaluation
even started (worse than random guessing, `log(2) = 0.69`). A day-by-day
diagnostic (predicted-probability range widening from `[0.17, 0.40]` at
domain 0 to `[0.00, 0.94]` by domain 9) isolated this to the trajectory
loss specifically -- disabling it entirely gave stable, reasonable log loss
(~0.5-0.6) throughout. Lowering `lambda_slow = lambda_fast` to **0.05**
(found empirically, not disclosed in the paper) keeps the escalation slow
enough that the BCE term's calibration pull dominates over a 60-domain
test run. This is documented in `sftl.py`'s module docstring. All numbers
below use this fixed, stable configuration -- the NaN/divergence bug is
not the reason for the results that follow.

## Even stabilized, SFTL underperforms every other method in every mode

| Mode | sftl log loss | Best method (log loss) | sftl vs. best |
|---|---|---|---|
| none | 0.925 | expanding / han_arw (0.311) | 3.0x worse |
| **abrupt** | **1.773** | han_arw (0.394) | **4.5x worse** |
| gradual | 0.788 | rolling_14 / han_arw (0.354) | 2.2x worse |
| recurring | 0.787 | expanding / han_arw (0.421) | 1.9x worse |

In every single mode, `sftl` is the worst method in the entire ladder --
worse than `rolling_1`, worse than `decay_hl1`, worse than the frozen
`validation_selected` baseline. Full tables:
`results_synthetic_<mode>/all_methods_with_sftl_comparison_table.csv`.

## The abrupt-drift result is the opposite of what a dual-timescale method should show

Day-by-day log loss around the shift (day 95), from
`results_synthetic_abrupt/all_methods_with_sftl_per_day_metrics.csv`:

| day | expanding | rolling_7 | han_arw | diff_forgetting | **sftl** |
|---|---|---|---|---|---|
| 92-94 (pre-shift) | 0.30-0.32 | 0.32-0.35 | 0.30-0.32 | 0.32-0.33 | **0.86-1.00** |
| 95 (shift) | 0.938 | 0.987 | 0.938 | 0.876 | **3.619** |
| 98 (peak) | 0.853 | 0.554 | 0.403 | 0.548 | **4.662** |
| 110 | 0.736 | 0.347 | 0.347 | 0.471 | **1.195** |
| 119 (end, 24d post-shift) | 0.673 | 0.360 | 0.360 | 0.455 | **0.993** |

Two distinct problems compound here, not one:

1. **SFTL is already badly miscalibrated before the shift even happens**
   (0.86-1.00 vs. ~0.30-0.35 for every other method on days 92-94, a
   *stable* pre-shift regime). This matches the long-run diagnostic: even
   at `lambda=0.05`, predicted-probability range keeps slowly widening
   over a long training run (`[0.00, 0.96]` by domain 50 in the stability
   test) -- the escalation is only slowed, not eliminated.
2. **The shift makes this catastrophically worse**, because log loss
   punishes confident-wrong predictions super-linearly. A model that was
   already overconfident on the *old* regime hits a regime change in the
   worst possible way: its most confident predictions are now the most
   wrong ones. Peak log loss (4.66) is nearly 12x the pre-shift baseline.
   It also never fully recovers within the 25 remaining test days (0.99 at
   day 119, still ~3x worse than `rolling_7`/`han_arw`'s ~0.35-0.36
   steady state).

This is the **opposite** pattern from Han ARW, whose best *relative*
performance (a decisive win over every static baseline) was specifically
under abrupt drift. A method built around a fast-adapting learner should,
in principle, have an edge exactly here -- instead it has its worst mode.
The mechanistic reason is specific to SFTL's trajectory loss, not to
"neural nets are bad at CTR": the ranking objective has no absolute
calibration target, only a relative one (beat your own recent past), so it
has no way to *notice* that the regime it's confident about no longer
applies -- unlike Han ARW's tournament, which is explicitly comparing
against a real held-out loss on live data every day.

## Honest caveats: this may be an underpowered reproduction, not proof the method is bad

Two things distinguish this reproduction from the paper's own setup, both
plausible (partial) explanations for the gap that are distinct from "SFTL
doesn't work":

- **`lambda_slow`/`lambda_fast` were tuned for stability, not performance.**
  The paper doesn't disclose these (nor `alpha`, nor the warmup length) --
  0.05 is the largest value that didn't diverge in a quick empirical check,
  not the result of a real hyperparameter search on held-out loss. The
  paper's own value could plausibly balance calibration and adaptivity
  better than a stability-only search found.
- **Far less data and capacity per domain than the paper's likely setup.**
  This benchmark uses 3000 rows/day and a from-scratch embedding + MLP
  (128-64 hidden, sized down from the paper's 1024-512-256); their
  datasets (Avazu, Taobao, CIKM2019) are industrial-scale with presumably
  much larger daily volumes, giving the embeddings far more gradient
  signal to leave random initialization before being asked to make
  confident, well-calibrated predictions.

## Verdict

Implementing SFTL was worth doing to answer the original question
precisely -- it's a genuinely different, purpose-built answer to "combine
short- and long-term signals," and finding + documenting a real
instability bug in a faithful reproduction of a published method is a
useful result on its own. But as built and tuned here, it does not
demonstrate the paper's claimed advantage, and actively does worse under
the one condition (abrupt drift) where Han ARW shines. Fairly validating
it would need: (1) a proper hyperparameter search for `lambda_slow`,
`lambda_fast`, and `alpha` against held-out dev loss rather than a
stability-only check, (2) much larger per-domain row counts (or GPU-
accelerated training to afford more epochs/larger hidden dims at the
current scale) closer to the paper's own data regime, and (3) probably a
calibration safeguard the paper doesn't mention explicitly -- e.g.
capping the trajectory loss's effective margin, or annealing
`lambda_slow`/`lambda_fast` down over time -- since the core issue (a
ranking objective with no absolute anchor) doesn't fully go away just by
weighting it small. Given AdaMoE already fills the P2 slot and performed
respectably (competitive with the P0 ladder in every mode, and the best
overall method under `abrupt` drift's runner-up spot), the practical
recommendation stands as before: **Han ARW for production, AdaMoE as the
CTR-specific reference; SFTL is scoped, implemented, and shown not to help
at this benchmark's scale without further investment.**
