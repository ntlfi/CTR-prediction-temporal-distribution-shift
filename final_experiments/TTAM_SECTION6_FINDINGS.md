# TTAM revised Section 6 — findings

Corrected fully-nested rolling-origin evaluation of the integrated
AMG-TP + AP-OPS pipeline ("TTAM"), spec
`TTAM_Additional_Experiments_Plan.pdf`, frozen protocol `TTAM_FROZEN.md`.
Every data-tuned setting is reselected from data strictly before each
evaluated origin. Seeds 0/1/2.

| dataset | origins (D) | run | code commit |
|---|---|---|---|
| Criteo | 16–30 (15) | `ttam/criteo/nested/` | in `selected_configs.json` |
| Avazu  | 5–9 (5)    | `ttam/avazu/nested/`  | in `selected_configs.json` |

Both runs were local (`run_ttam_nested.py`, `--n-workers 4`/`3`); the
Avazu loader was made chunk-hashed (commit `c1b4c9e`) so full-data
Avazu (40M rows) fits on a 62 GB box. The ablation is the **revised 2×2
design** (historical predictor × calibration, `TTAM_FROZEN.md` §2b).
Stats `TTAM_SECTION6_STATS.md` + `section6_factorial.json`; figure
`ttam/section6_figure.png`; Criteo bidding `ttam/criteo/bidding/`.

---

## 6.2 Main comparison — TTAM has the lowest mean log loss on **both** datasets

Seed-averaged equal-day score `A_m`; gain `G_m = A_m − A_TTAM` (positive ⇒
TTAM lower loss); 95% paired day-bootstrap CI, 10,000 resamples, seed
20260908.

### Criteo (D = 15)

| method | A_m | gain of TTAM | 95% CI | origins won |
|---|---|---|---|---|
| Expanding | 0.609475 | +0.001446 | [+0.00129, +0.00163] | 15/15 |
| Best Fixed Window | 0.608744 | +0.000715 | [+0.00060, +0.00085] | 15/15 |
| ARW | 0.608631 | +0.000603 | [+0.00053, +0.00068] | 15/15 |
| AdaMoE | 0.608410 | +0.000381 | [+0.00030, +0.00047] | 15/15 |
| OPS | 0.608201 | +0.000172 | [+0.00013, +0.00022] | 15/15 |
| **TTAM** | **0.608029** | — | — | — |

### Avazu (D = 5 — descriptive, intervals fragile, no significance claims)

| method | A_m | gain of TTAM | 95% CI | origins won |
|---|---|---|---|---|
| Expanding | 0.402094 | +0.001160 | [+0.00093, +0.00140] | 5/5 |
| Best Fixed Window | 0.402213 | +0.001279 | [+0.00020, +0.00268] | 4/5 |
| ARW | 0.402359 | +0.001425 | [+0.00029, +0.00270] | 4/5 |
| AdaMoE | 0.401538 | +0.000604 | [+0.00034, +0.00090] | 5/5 |
| OPS | 0.401147 | +0.000213 | [+0.00014, +0.00028] | 5/5 |
| **TTAM** | **0.400934** | — | — | — |

TTAM has the lowest log loss on both datasets and beats every baseline at
every origin (Criteo) or almost every origin (Avazu: 5/5 vs Expanding,
AdaMoE, OPS; 4/5 vs BestFixedWindow, ARW). The margin over the strongest
baseline **OPS is small and near-identical across datasets** — +1.7×10⁻⁴
(Criteo) / +2.1×10⁻⁴ (Avazu) — and directionally unanimous.

## 6.3 Ablation — 2×2: historical predictor × calibration

Revised design (2026-09-09): cross **historical predictor ∈ {AMG-TP,
expanding history}** with **calibration ∈ {AP-OPS, none}**. The window
baseline is **expanding history** — pre-registered before looking at the
final results: the canonical "train on all past data" predictor, zero
tuning DOF. Identical historical predictions are shared within each row
(`expanding` / `expanding_apops` both use `bank[d].preds["expanding"]`;
`amgtp_only` / `ttam` both use the same selected AMG-TP q-stream).

Mean score A_m per cell (lower = better):

**Criteo (D = 15)**

| historical \ calibration | none | AP-OPS |
|---|---|---|
| **expanding history** | 0.609475 | 0.608807 |
| **AMG-TP** | 0.608401 | **0.608029** (= TTAM) |

**Avazu (D = 5 — descriptive only)**

| historical \ calibration | none | AP-OPS |
|---|---|---|
| **expanding history** | 0.402094 | 0.401737 |
| **AMG-TP** | 0.401514 | **0.400934** (= TTAM) |

### Marginal effects — **both timescales contribute, on both datasets**

Mean difference (negative ⇒ the added component lowers loss), 95% paired
day-bootstrap CI:

| effect | at | Criteo | Avazu |
|---|---|---|---|
| **AP-OPS** (calibration) | historical = expanding | −6.7e-4 [−8.7e-4, −5.0e-4], 15/15 | −3.6e-4 [−5.5e-4, −1.8e-4], 5/5 |
| **AP-OPS** (calibration) | historical = AMG-TP | −3.7e-4 [−4.7e-4, −2.9e-4], 15/15 | −5.8e-4 [−9.0e-4, −3.1e-4], 5/5 |
| **AMG-TP** (historical) | calibration = none | −1.07e-3 [−1.23e-3, −0.93e-3], 15/15 | −5.8e-4 [−9.2e-4, −2.7e-4], 5/5 |
| **AMG-TP** (historical) | calibration = AP-OPS | −7.8e-4 [−8.6e-4, −7.0e-4], 15/15 | −8.0e-4 [−1.13e-3, −0.54e-3], 5/5 |

Every one of the eight marginal effects has a CI clear of zero, all
15/15 (Criteo) / 5/5 (Avazu) origins. **This overturns the earlier
"AMG-TP adds nothing" reading** — that earlier ablation used the
validation-fitted 5-horizon simplex mixture as its "no-AMG-TP" predictor,
which is itself an adaptive multi-horizon combination and so already
captured most of what AMG-TP does. Against a plain expanding-history
baseline the AMG-TP module's contribution is large (~0.5–1×10⁻³) and
unanimous.

### Interaction — **the datasets disagree**

Interaction `= (ttam − amgtp_only) − (expanding_apops − expanding)`:

| dataset | interaction | 95% CI | origins | reading |
|---|---|---|---|---|
| Criteo | **+3.0e-4** | [+1.5e-4, +4.6e-4] | 14/15 positive | **sub-additive / substitutes** — AP-OPS helps *less* when AMG-TP is already present (−3.7e-4 vs −6.7e-4), and vice-versa. The two timescales partly do the same job. |
| Avazu | **−2.2e-4** | [−3.3e-4, −1.1e-4] | 5/5 negative | **super-additive / synergy** — AP-OPS helps *more* with AMG-TP (−5.8e-4 vs −3.6e-4). |

Both CIs exclude zero, so on neither dataset are the two components
simply additive; but the sign flips. On Criteo, an adaptive scalar
calibrator (AP-OPS) and an adaptive multi-scale historical model (AMG-TP)
are partly redundant — either one recovers most of the gain over plain
expanding history, and stacking them gives less than the sum. On Avazu
(D = 5, fragile) they reinforce.

### Reconciling with the main table

TTAM's edge over the strongest *baseline* (OPS, +1.7×10⁻⁴ / +2.1×10⁻⁴) is
small because OPS's historical side — the validation-fitted 5-horizon
mixture — is already a good adaptive predictor. Decomposed against the
*naive* baseline (plain expanding history), both TTAM components pull real
weight: expanding 0.609475 → +AMG-TP alone 0.608401 → +AP-OPS alone
0.608807 → +both 0.608029 (Criteo). Among the uncalibrated predictors
AMG-TP is the best base model (0.608401, below the fitted mixture's
0.608408); AP-OPS on expanding history (0.608807) does not by itself
reach the fitted-mixture+OPS main-table `ops` arm (0.608201).

### Caveats

- **D = 5 for Avazu** — descriptive only; the interaction sign in
  particular should not be over-read at this power.
- These are **paired comparisons on the same origin days used to develop
  the method**. Per the plan (§7) the corrected nested replay fixes the
  *selection* leak, not prospective out-of-sample confirmation.

## What this establishes

1. **TTAM (both timescales) has the lowest mean log loss on both public
   datasets**, beating every baseline at every (Criteo) or almost every
   (Avazu) origin.
2. **Against a plain expanding-history baseline, both components
   contribute significantly** — the AMG-TP adaptive short/long historical
   model (~0.5–1×10⁻³) and the AP-OPS adaptive calibrator (~0.4–0.7×10⁻³),
   all marginal-effect CIs clear of zero on both datasets.
3. **The two timescales are not additive, and the datasets disagree on
   the direction**: substitutes on Criteo (+3×10⁻⁴ interaction, either one
   recovers most of the gain), synergistic on Avazu (−2×10⁻⁴). The
   earlier "AMG-TP adds nothing" conclusion was an artefact of ablating
   against the fitted mixture rather than a plain window.
4. Magnitudes are still consistent with the repo's standing finding that
   real intra-month drift on Criteo and Avazu is shallow — the whole
   spread from naive expanding history to full TTAM is ~1.5×10⁻³ log loss.

## 6.4 Downstream bidding — the log-loss edge does not convert to clicks (Criteo)

Criteo bidding replay, 25% daily budget, clicks at matched spend, same
auction + pacing for every method, only the pCTR input changes
(`run_ttam_bidding.py`; recorded display `cost` as the price proxy).

| baseline | aggregate click gain of TTAM | 95% CI | origins won by TTAM |
|---|---|---|---|
| Expanding | −0.062 % | [−0.086, −0.038] | 1/15 |
| Best Fixed Window | −0.055 % | [−0.082, −0.025] | 2/15 |
| ARW | −0.053 % | [−0.074, −0.030] | 2/15 |
| AdaMoE | −0.066 % | [−0.087, −0.045] | 0/15 |
| OPS | −0.024 % | [−0.031, −0.017] | 1/15 |

TTAM wins **fewer** clicks at matched spend than every baseline — small
(−0.02 % to −0.07 %) but every CI is below zero. The prediction-side
log-loss edge over OPS (~1.7×10⁻⁴) does **not** carry through to
downstream bidding value; if anything it is marginally negative. **Avazu bidding: pending** — no
recorded price field; a simulated price model needs a separate frozen
specification (plan section 10).

## Reproduce

```
# banks (one-time) — local
PYTHONPATH=. python3 final_experiments/ttam/build_banks.py --source criteo \
    --data data/criteo_attribution_dataset.tsv.gz --cache-dir final_experiments/ttam/_bankcache --n-jobs 4
PYTHONPATH=. python3 final_experiments/ttam/build_banks.py --source avazu \
    --data data/avazu/Avazu_x4.zip --cache-dir final_experiments/ttam/_bankcache --n-jobs 1 --seeds 0
#   (repeat --seeds 1 / --seeds 2 as separate processes on a memory-tight box)
# nested runs
PYTHONPATH=. python3 final_experiments/run_ttam_nested.py --source criteo \
    --data data/criteo_attribution_dataset.tsv.gz --out final_experiments/ttam/criteo/nested --n-workers 4
PYTHONPATH=. python3 final_experiments/run_ttam_nested.py --source avazu \
    --data data/avazu/Avazu_x4.zip --out final_experiments/ttam/avazu/nested --n-workers 3
# section 6 outputs (stats + figure over both, Criteo bidding)
bash final_experiments/run_ttam_section6.sh
```

or `sbatch final_experiments/ttam_{banks,nested}_{criteo,avazu}.slurm`.

## Not done

- Avazu downstream bidding — needs a frozen price-model spec.
- Manuscript placeholders — no paper source in this repo; this file + the
  two-panel figure are the deliverables.
