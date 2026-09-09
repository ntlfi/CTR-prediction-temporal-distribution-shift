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
Avazu (40M rows) fits on a 62 GB box. Stats `TTAM_SECTION6_STATS.md`;
figure `ttam/section6_figure.png`; Criteo bidding `ttam/criteo/bidding/`.

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

## 6.3 Ablation — the gain is the AP-OPS calibration layer, **not** AMG-TP, on both datasets

`G_m` = gain of TTAM over the variant (positive ⇒ TTAM better).

### Criteo

| variant | historical | calibration | A_m | G_m | 95% CI | origins won |
|---|---|---|---|---|---|---|
| without both | fixed mixture | none | 0.608408 | +0.000379 | [+0.00030, +0.00047] | 15/15 |
| AMG-TP only | AMG-TP | none | 0.608401 | +0.000372 | [+0.00029, +0.00047] | 15/15 |
| AP-OPS only | fixed mixture | AP-OPS | 0.608031 | **+2.2e-6** | **[−1.0e-6, +5.3e-6]** | **11/15** |
| TTAM | AMG-TP | AP-OPS | 0.608029 | — | — | — |

### Avazu

| variant | historical | calibration | A_m | G_m | 95% CI | origins won |
|---|---|---|---|---|---|---|
| without both | fixed mixture | none | 0.401535 | +0.000600 | [+0.00033, +0.00090] | 5/5 |
| AMG-TP only | AMG-TP | none | 0.401514 | +0.000580 | [+0.00031, +0.00090] | 5/5 |
| AP-OPS only | fixed mixture | AP-OPS | 0.400940 | **+5.6e-6** | **[−1.8e-5, +3.1e-5]** | **4/5** |
| TTAM | AMG-TP | AP-OPS | 0.400934 | — | — | — |

The plan's two primary contrasts land the same way on both datasets:

1. **`A_{AP-OPS only} − A_{TTAM}` ≈ 0** — CI straddles zero, TTAM wins
   only 11/15 (Criteo) / 4/5 (Avazu) origins. Adding the AMG-TP historical
   module on top of AP-OPS changes nothing measurable.
2. **`A_{AMG-TP only} − A_{TTAM}` = +0.00037 (Criteo) / +0.00058 (Avazu)**,
   essentially identical to `without both`. The AMG-TP module on its own
   (sample-gated short/long mixing + learned persistence + deployed-weight
   memory) contributes ~7×10⁻⁶ (Criteo) / ~2×10⁻⁵ (Avazu) — indistinguishable
   from just fitting a constant mixture.

So on **both** public datasets, **the entire TTAM improvement over OPS is
the AP-OPS calibration layer**; the historical short/long gating machinery
adds nothing, with or without AP-OPS. **No synergy** — the full method has
the lowest mean only because AP-OPS does. The nested selection agrees: on
Criteo it pins `amgtp_rho` at the grid minimum (0.2) at every origin; on
Avazu `amgtp_rho` wanders (0.2–0.5) with no effect on the score; and
`apops_fixed` and `apops_amgtp` pick the **same** calibration config at
every origin on both datasets.

## What this establishes

**AP-OPS re-confirmed under the corrected fully-nested protocol, now on
two datasets.** The earlier AP-OPS result inherited its ONS / meta /
mixture / OPS settings from a frozen config whose development window
overlapped early outer days. Reselecting *every* knob from strictly
pre-origin data, the result stands on both Criteo and Avazu: **AP-OPS
beats plain OPS, 15/15 and 5/5 origins, CI excluding zero**, same small
(~2×10⁻⁴) magnitude, same directional unanimity, now with no protocol
objection.

The corollary is a clean **negative result on the historical module**:
the two-timescale short/long adaptive-memory idea (M1–M6 / AMG-TP line) —
sample-specific gating, learned persistence, deployed-weight memory — does
**not** add predictive value over a well-tuned adaptive scalar calibrator
on either public dataset, consistent with this repo's standing finding
that real intra-month drift on Criteo and Avazu is shallow.

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
(−0.02 % to −0.07 %) but every CI is below zero. The tiny prediction-side
calibration gain does **not** carry through to downstream bidding value;
if anything it is marginally negative. **Avazu bidding: pending** — no
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
