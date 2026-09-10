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

Both runs local (`run_ttam_nested.py`); both dataset loaders chunk-hashed
so full data fits a 62 GB box. Ablation = the **2×2 design** (historical
predictor × calibration, `TTAM_FROZEN.md` §2b). Stats
`TTAM_SECTION6_STATS.md` + `section6_factorial.json`; figure
`ttam/section6_figure.png`; Criteo bidding `ttam/criteo/bidding/`.

**Protocol-conformance pass (2026-09-10, this is the final run).** Relative
to the first 2×2 run, this run additionally:
1. **enforces the 30-min label delay everywhere a cutoff reads labels** —
   expert-bank fits (day `d`'s experts exclude day `d-1`'s post-84600 s
   tail; empty-fit fallback uses matured-history CTR, never the scored
   day), the inner-validation objective, AMG-TP's state summaries, the
   ARW/AdaMoE causal loss histories — not only the online calibrator
   queues;
2. **selects the AP-OPS reset-anchor OPS settings and the meta learning
   rate `eta_m` per origin** (pass 2a / 2b), instead of inheriting
   `eta_m = 100` and `B 0.25 / eta0 0.3 / const` from tuning that
   overlapped the evaluation period.

Effect on the numbers: everything moved by ≤ 5×10⁻⁵ log loss and **no
conclusion changed**. Two things worth noting from the per-origin picks:
the reset-anchor OPS converges to `B 0.25 / eta0 0.3 / const` at **every
origin, every stream, both datasets** — the previously-hardcoded value,
now independently re-derived; but `eta_m` does **not** — Criteo origins
split over {30, 100, 300} (7 pick 300), so it genuinely needed to be
selected rather than fixed.

---

## 6.2 Main comparison — TTAM has the lowest mean log loss on **both** datasets

Seed-averaged equal-day score `A_m`; gain `G_m = A_m − A_TTAM` (positive ⇒
TTAM lower loss); 95% paired day-bootstrap CI, 10,000 resamples, seed
20260908.

### Criteo (D = 15)

| method | A_m | gain of TTAM | 95% CI | origins won |
|---|---|---|---|---|
| Expanding | 0.609525 | +0.001496 | [+0.00131, +0.00172] | 15/15 |
| Best Fixed Window | 0.608713 | +0.000684 | [+0.00055, +0.00086] | 15/15 |
| ARW | 0.608643 | +0.000614 | [+0.00052, +0.00071] | 15/15 |
| AdaMoE | 0.608464 | +0.000435 | [+0.00032, +0.00059] | 15/15 |
| OPS | 0.608203 | +0.000174 | [+0.00014, +0.00022] | 15/15 |
| **TTAM** | **0.608029** | — | — | — |

### Avazu (D = 5 — descriptive, intervals fragile, no significance claims)

| method | A_m | gain of TTAM | 95% CI | origins won |
|---|---|---|---|---|
| Expanding | 0.402094 | +0.001157 | [+0.00092, +0.00140] | 5/5 |
| Best Fixed Window | 0.402213 | +0.001277 | [+0.00020, +0.00268] | 4/5 |
| ARW | 0.402359 | +0.001423 | [+0.00028, +0.00269] | 4/5 |
| AdaMoE | 0.401538 | +0.000601 | [+0.00034, +0.00089] | 5/5 |
| OPS | 0.401147 | +0.000210 | [+0.00014, +0.00028] | 5/5 |
| **TTAM** | **0.400937** | — | — | — |

TTAM has the lowest log loss on both datasets and beats every baseline at
every origin (Criteo) or almost every origin (Avazu: 5/5 vs Expanding,
AdaMoE, OPS; 4/5 vs BestFixedWindow, ARW). The margin over the strongest
baseline **OPS is small and near-identical across datasets** — +1.7×10⁻⁴
(Criteo) / +2.1×10⁻⁴ (Avazu) — and directionally unanimous.

## 6.3 Ablation — 2×2: historical predictor × calibration

Cross **historical predictor ∈ {AMG-TP, expanding history}** with
**calibration ∈ {AP-OPS, none}**. The window baseline is **expanding
history** — pre-registered before looking at the final results: the
canonical "train on all past data" predictor, zero tuning DOF. Identical
historical predictions are shared within each row (`expanding` /
`expanding_apops` both use `bank[d].preds["expanding"]`; `amgtp_only` /
`ttam` both use the same selected AMG-TP q-stream). All effects are read
**relative to the Expanding control**.

Mean score A_m per cell (lower = better):

**Criteo (D = 15)**

| historical \ calibration | none | AP-OPS |
|---|---|---|
| **expanding history** | 0.609525 | 0.608833 |
| **AMG-TP** | 0.608457 | **0.608029** (= TTAM) |

**Avazu (D = 5 — descriptive only)**

| historical \ calibration | none | AP-OPS |
|---|---|---|
| **expanding history** | 0.402094 | 0.401742 |
| **AMG-TP** | 0.401514 | **0.400937** (= TTAM) |

### Marginal effects — **both timescales contribute, on both datasets**

Mean difference (negative ⇒ the added component lowers loss), 95% paired
day-bootstrap CI:

| effect | at | Criteo | Avazu |
|---|---|---|---|
| **AP-OPS** (calibration) | historical = expanding | −6.9e-4 [−9.3e-4, −5.1e-4], 15/15 | −3.5e-4 [−5.4e-4, −1.8e-4], 5/5 |
| **AP-OPS** (calibration) | historical = AMG-TP | −4.3e-4 [−5.8e-4, −3.1e-4], 15/15 | −5.8e-4 [−8.9e-4, −3.0e-4], 5/5 |
| **AMG-TP** (historical) | calibration = none | −1.07e-3 [−1.20e-3, −0.94e-3], 15/15 | −5.8e-4 [−9.2e-4, −2.7e-4], 5/5 |
| **AMG-TP** (historical) | calibration = AP-OPS | −8.0e-4 [−8.8e-4, −7.4e-4], 15/15 | −8.1e-4 [−1.14e-3, −0.53e-3], 5/5 |

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
| Criteo | **+2.6e-4** | [+1.4e-4, +4.0e-4] | 14/15 positive | **sub-additive / substitutes** — AP-OPS helps *less* when AMG-TP is already present (−4.3e-4 vs −6.9e-4), and vice-versa. The two timescales partly do the same job. |
| Avazu | **−2.3e-4** | [−3.4e-4, −1.1e-4] | 5/5 negative | **super-additive / synergy** — AP-OPS helps *more* with AMG-TP (−5.8e-4 vs −3.5e-4). |

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
weight: expanding 0.609525 → +AMG-TP alone 0.608457 → +AP-OPS alone
0.608833 → +both 0.608029 (Criteo). Among the uncalibrated predictors
AMG-TP is the best base model; AP-OPS on plain expanding history
(0.608833) does not by itself reach the fitted-mixture + OPS main-table
`ops` arm (0.608203).

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
   the direction**: substitutes on Criteo (+2.6×10⁻⁴ interaction, either
   one recovers most of the gain), synergistic on Avazu (−2.3×10⁻⁴,
   D = 5 — descriptive). The earlier "AMG-TP adds nothing" conclusion was
   an artefact of ablating against the fitted mixture rather than a plain
   window.
4. Magnitudes are still consistent with the repo's standing finding that
   real intra-month drift on Criteo and Avazu is shallow — the whole
   spread from naive expanding history to full TTAM is ~1.5×10⁻³ log loss.
5. **The protocol-conformance pass did not move the story** — enforcing
   the label delay across the historical side and selecting `eta_m` /
   the reset anchor per origin shifted every number by ≤ 5×10⁻⁵ and
   changed no sign or significance verdict.

## 6.4 Downstream bidding — the log-loss edge does not convert to clicks (Criteo)

Criteo **interpolated offline replay**, 25 % daily budget, clicks at each
origin's target spend, same auction + pacing for every method, only the
pCTR input changes (`run_ttam_bidding.py`; recorded display `cost` as the
price proxy). Every one of the 1080 (origin × seed × method × budget)
target spends lies **inside a non-degenerate interpolation bracket** —
none clamped (`summary.json::bracket_check`); per-cell realised (paced)
spend is in `bidding_cells.csv`.

| baseline | aggregate click gain of TTAM | 95 % CI | origins won by TTAM |
|---|---|---|---|
| Expanding | −0.062 % | [−0.085, −0.039] | 1/15 |
| Best Fixed Window | −0.051 % | [−0.078, −0.022] | 4/15 |
| ARW | −0.054 % | [−0.075, −0.032] | 2/15 |
| AdaMoE | −0.066 % | [−0.089, −0.043] | 1/15 |
| OPS | −0.024 % | [−0.032, −0.016] | 1/15 |

TTAM wins **fewer** clicks at matched spend than every baseline — small
(−0.02 % to −0.07 %) but every CI is below zero. The prediction-side
log-loss edge over OPS (~1.7×10⁻⁴) does **not** carry through to
downstream bidding value; if anything it is marginally negative. This is
reported as observed. **Avazu bidding is outside the main experiment** —
no recorded price field; a simulated price model would need a separate
frozen specification (plan section 10).

## Reproduce

```
# banks (one-time) -- one seed per process on a memory-tight shared box
for src_data in "criteo data/criteo_attribution_dataset.tsv.gz 4" "avazu data/avazu/Avazu_x4.zip 1"; do
  set -- $src_data
  for s in 0 1 2; do
    PYTHONPATH=. python3 final_experiments/ttam/build_banks.py --source $1 --data $2 \
        --cache-dir final_experiments/ttam/_bankcache --n-jobs $3 --seeds $s
  done
done
# nested runs (pass 1 -> 2a anchor -> 2b AP-OPS+eta_m -> 3)
PYTHONPATH=. python3 final_experiments/run_ttam_nested.py --source criteo \
    --data data/criteo_attribution_dataset.tsv.gz --out final_experiments/ttam/criteo/nested --n-workers 4
PYTHONPATH=. python3 final_experiments/run_ttam_nested.py --source avazu \
    --data data/avazu/Avazu_x4.zip --out final_experiments/ttam/avazu/nested --n-workers 3
# section 6 outputs (stats + figure over both, Criteo bidding)
bash final_experiments/run_ttam_section6.sh
```

or `sbatch final_experiments/ttam_{banks,nested}_{criteo,avazu}.slurm`. Bank caches are
keyed `_d1800`; `run_ttam_nested.py` resumes from `pass1_*` / `pass2a_*` / `pass2b_*`
per-seed checkpoints.

## Not done

- Avazu downstream bidding — needs a frozen price-model spec.
- Manuscript placeholders — no paper source in this repo; this file + the
  two-panel figure are the deliverables.
