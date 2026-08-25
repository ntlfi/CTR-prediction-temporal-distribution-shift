# SFTL debugging findings (per `sftl-debugging-plan.pdf`)

Full staged failure analysis, per the debugging plan's protocol. All eight
stages were run through P1 (the plan's own stop-gate); P2 (Stage 7
calibration stabilizers, Stage 8 native-setting reproduction) is addressed
in the verdict below rather than executed, per the reasoning there.

## Diagnosis: **H2 (calibration/confidence runaway), mechanistically caused by H4 (trajectory term dominates BCE)**

Not H1 (bug — cleanly ruled out). Not primarily H3 (EMA timescale — real
but a symptom-modulator, not the cause). Not H5 (genuine limitation — a
concrete, quantified, actionable lever exists). This is the plan's own H2
decision rule verified almost exactly: *"AUC improves or remains
competitive while log loss/Brier deteriorate sharply."*

## Stage 1 — implementation invariant: PASS

With `lambda_slow = lambda_fast = 0`, the working learner's predictions
matched a plain BCE-only model (same seed, same data order) to **exactly
0.00e+00** max absolute difference over the first 10 domains. The loss
composition and gradient flow are correct; nothing here is a coding bug.

## Stage 2/3 — production config (120 days, shift at day 95): the runaway is real and severe

`stage2_abrupt_shift_auc_logloss.png`, `stage3_margin_logit_trajectories.png`.

| Day (pre-shift) | AUC (fast) | log loss (fast) | max \|logit\| |
|---|---|---|---|
| 90 | 0.876 | 1.150 | 29,474 |
| 92 | 0.880 | 0.987 | 24,207 |
| 94 | 0.887 | 0.859 | 24,883 |

Logits in the **tens of thousands**, days before the shift, in a period
where nothing is actually changing. Sigmoid outputs at this magnitude are
numerically indistinguishable from exactly 0 or 1. AUC stays good (0.876–
0.887) throughout — ranking survives — while log loss (0.86–1.15) is
already far worse than every other method's ~0.30–0.35 on the same days.
**This directly corrects the earlier claim** (in `sftl.py`'s docstring and
`results/sftl_analysis.md`) that `lambda=0.05` "keeps training stable...
comparable to a no-trajectory-loss ablation over a 60-domain run" — that
check only ran 60 domains; over the full 120-domain production horizon the
same escalation continues and reaches these extreme magnitudes. **λ=0.05
delays the runaway, it does not prevent it.**

Post-shift, AUC also collapses (0.49–0.59) alongside log loss spiking to
4.66 (day 98) — the pre-existing extreme overconfidence appears to have
corrupted the model's actual representations, not just its calibration,
which plausibly explains the incomplete recovery documented in
`results/sftl_analysis.md`.

## Stage 4 — EMA half-life sweep: masks the symptom, doesn't fix the cause

| Half-life H (steps) | alpha | Pre-shift LL | Peak post-shift LL | Recovery (days) | Overall LL |
|---|---|---|---|---|---|
| 10 | 0.9330 | 0.653 | 3.589 | 2 | 0.714 |
| 30 | 0.9772 | 0.582 | 2.879 | 4 | 0.658 |
| 100 | 0.9931 | 0.528 | 2.108 | 7 | 0.622 |
| **1000** | 0.9993 | 0.602 | **0.693** | **0** | **0.615** |

A much slower EMA (H=1000, i.e. the served fast learner barely moves)
gives the best numbers by far. This is real but is a **lag effect, not a
fix**: the slow learner is a hard copy of the working learner at every
domain boundary regardless of EMA speed, so it inherits the full
escalation no matter how slowly the fast learner updates. A slow fast-
learner just means the *served* predictions reflect an older, less-
escalated snapshot of a working learner that is still running away
underneath. Extending this run past 60 domains would be expected to
eventually expose the same problem once the fast learner catches up.

## Stage 5 — ablation: the clean H2 signature

`stage5_ablation_table.csv` (fast-learner metrics on locked test, proxy config):

| Variant | log loss | AUC | mean \|logit\| |
|---|---|---|---|
| A. BCE only | 0.583 | 0.656 | 1.49 |
| B. Slow-trajectory only | 0.648 (+11%) | 0.656 | 2.09 |
| C. Fast-trajectory only | 0.657 (+13%) | 0.658 | 2.19 |
| D. Full SFTL (slow + fast) | **0.963 (+65%)** | 0.658 | **6.30** |

AUC is flat within noise (0.656–0.658) across all four variants — ranking
quality is untouched. `mean_abs_logit` and log loss both escalate cleanly,
and the *combination* of both trajectory terms is far worse than either
alone (6.30 vs. ~2.1–2.2) — a synergistic interaction, not additive. This
is the plan's H2 symptom, produced with no ambiguity: the failure is in
probability calibration, not in ranking. (Note: a simple `calib_error =
|mean_pred - true_rate|` metric stayed flat, ~0.011–0.014, across all four
variants — too coarse to see this failure; log loss and logit magnitude
are what expose it. Worth remembering for any future SFTL work.)

## Stage 6 — gradient-contribution measurement: the mechanism, quantified

`stage6_gradient_snapshot.json`, `stage6_gradient_ratio_lambda_sweep.csv`.

At `lambda=1`, the trajectory term's raw gradient norm is **already 2.6–4.5x
larger than BCE's**, from the very first domain it activates in (domain 3,
right after warmup):

| Snapshot | domain | \|\|grad BCE\|\| | \|\|grad traj_slow\|\| | ratio (slow) | \|\|grad traj_fast\|\| | ratio (fast) |
|---|---|---|---|---|---|---|
| early | 3 | 0.059 | 0.178 | 3.00x | 0.165 | 2.78x |
| pre-shift | 47 | 4.561 | 20.60 | **4.52x** | 12.05 | 2.64x |

The slow-trajectory term's dominance over BCE *grows* over training (3.00x
→ 4.52x over 44 domains) even with nothing in the environment changing —
this is the runaway's actual driver: the ranking loss has no absolute
scale, so as margins grow its gradient keeps growing too, while BCE's
gradient (also growing, since a miscalibrated model gets larger BCE
gradients on new data) grows more slowly. Note also that the trajectory
*loss value* itself stays near a flat plateau (0.69–0.89, close to
`log(2)`) throughout — **monitoring the loss value alone would have
completely hidden this problem**; only the gradient norm reveals it. This
is exactly the debugging plan's own warning: instrument the optimization,
don't just watch the headline loss.

Choosing `lambda` by target initial gradient contribution (not arbitrary
scale) gives a clean, monotonic dose-response, run at the same proxy
config:

| Target contribution | λ_slow | λ_fast | log loss | AUC |
|---|---|---|---|---|
| 1% | 0.0033 | 0.0036 | **0.587** | 0.653 |
| 5% | 0.0167 | 0.0180 | 0.615 | 0.654 |
| 10% | 0.0333 | 0.0359 | 0.740 | 0.658 |
| 25% | 0.0833 | 0.0898 | 1.575 | 0.656 |
| 50% | 0.1666 | 0.1797 | 3.277 | 0.665 |

At 1% target contribution, log loss (0.587) is nearly identical to the
BCE-only baseline (0.583) — **a small enough λ genuinely prevents the
runaway, at least over this 60-domain test window.** For calibration: the
originally-used `lambda=0.05` corresponds to roughly a **14–15% initial
contribution** (0.05 × 3.00 ≈ 15%, 0.05 × 2.78 ≈ 14%) — already past the
point (10%) where clear degradation shows up, which is consistent with
both the earlier 60-domain check looking "fine" (degradation was still
mild there) and the full 120-domain production run reaching catastrophic
values (the ratio itself grows with more domains).

**Caveat on the 1% result:** since the trajectory-to-BCE gradient ratio
itself grows over training (confirmed above), a fixed λ calibrated against
an *early* snapshot is not guaranteed to stay at 1% contribution
indefinitely — it could still drift upward given enough domains. This was
only verified over 60 domains, not the full 120-domain production horizon.
Call this "very likely much better, not proven safe indefinitely."

## Verdict and recommendation

**Primary diagnosis: H2/H4 — calibration/confidence runaway caused by the
trajectory loss's disproportionate and growing gradient contribution
relative to BCE.** Not a bug. Not primarily an EMA timescale problem
(that's a secondary lag effect). A concrete, quantified fix direction
exists: λ needs to target roughly 1–5% of BCE's gradient norm, not the
10–50%+ that naive guesses (including the original 0.05 default) land in.

**But the debugging plan's stop criterion still applies.** Even the
cleanest variant tested — BCE-only, no trajectory loss at all — scores
0.583 log loss on the proxy config, still clearly worse than that same
config's best static P0 baselines (`rolling_7`/`decay_hl3` ≈ 0.50, from
the original SFTL scoping work). The calibration-runaway problem and the
from-scratch-neural-net-on-small-per-domain-data problem are separate and
compounding; fixing the first (a real, now well-understood, fixable issue)
would not by itself close the gap to Han ARW or the simple window
baselines. Per the plan: *"If the faithful version remains worse than
simple rolling-window, Han ARW, and AdaMoE baselines, stop investing in
it. Preserve the failure analysis as evidence."*

**Recommendation:** stop here on faithful SFTL. This debugging pass
succeeded at its actual goal — it rules out implementation error, gives a
precise mechanistic account of the failure (not "neural nets are bad at
CTR," a specific and fixable gradient-scale issue in the trajectory loss),
and quantifies exactly how much a proper fix would need to change. That is
a complete, useful result on its own. Actually deploying a corrected
version (λ retuned to ~1-5% target contribution, validated over the full
production horizon, and ideally with more per-domain data to address the
separate undertraining issue) would be a new, explicitly-labeled variant
per the plan's Stage 7 rule ("test modified versions under separate
names") — not a silent edit to what's reported as SFTL here.
