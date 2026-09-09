# TTAM revised Section 6 — findings (Criteo)

Corrected fully-nested rolling-origin evaluation of the integrated
AMG-TP + AP-OPS pipeline ("TTAM"), spec
`TTAM_Additional_Experiments_Plan.pdf`, frozen protocol `TTAM_FROZEN.md`.
Every data-tuned setting is reselected from data strictly before each
evaluated origin. Criteo origins 16–30 (D = 15), seeds 0/1/2.

Run: `final_experiments/ttam/criteo/nested/` (code commit in
`selected_configs.json`; local run, `run_ttam_nested.py --n-workers 4`,
~1 h). Stats `TTAM_SECTION6_STATS.md`; figure
`final_experiments/ttam/section6_figure.png`; bidding
`final_experiments/ttam/criteo/bidding/`.

Avazu is **not** run here (40M rows, no Slurm on the run host; deferred).

---

## 6.2 Main comparison — TTAM has the lowest mean log loss on Criteo

Seed-averaged, equal-day score `A_m`; gain `G_m = A_m − A_TTAM` (positive
⇒ TTAM lower loss); 95% paired day-bootstrap CI, 10,000 resamples, seed
20260908.

| method | A_m | gain of TTAM | 95% CI | origins won by TTAM |
|---|---|---|---|---|
| Expanding | 0.609475 | +0.001446 | [+0.00129, +0.00163] | 15/15 |
| Best Fixed Window | 0.608744 | +0.000715 | [+0.00060, +0.00085] | 15/15 |
| ARW | 0.608631 | +0.000603 | [+0.00053, +0.00068] | 15/15 |
| AdaMoE | 0.608410 | +0.000381 | [+0.00030, +0.00047] | 15/15 |
| OPS | 0.608201 | +0.000172 | [+0.00013, +0.00022] | 15/15 |
| **TTAM** | **0.608029** | — | — | — |

TTAM beats every baseline at every one of the 15 origins; all CIs exclude
zero; the block-2 moving-block bootstrap agrees. The margin over the
strongest baseline (OPS) is small — **1.7 × 10⁻⁴ log loss** — but
directionally unanimous.

## 6.3 Ablation — the gain is the AP-OPS calibration layer, not AMG-TP

| variant | historical | calibration | A_m | gain of TTAM | 95% CI | origins won |
|---|---|---|---|---|---|---|
| without both | fixed mixture | none | 0.608408 | +0.000379 | [+0.00030, +0.00047] | 15/15 |
| AMG-TP only | AMG-TP | none | 0.608401 | +0.000372 | [+0.00029, +0.00047] | 15/15 |
| AP-OPS only | fixed mixture | AP-OPS | 0.608031 | **+0.0000022** | **[−1.0e-6, +5.3e-6]** | **11/15** |
| TTAM | AMG-TP | AP-OPS | 0.608029 | — | — | — |

Two clean reads, both matching the plan's primary contrasts:

1. **`A_AP-OPS-only − A_TTAM` ≈ 0.** The CI straddles zero and TTAM wins
   only 11/15 origins. Adding the AMG-TP historical module on top of
   AP-OPS changes nothing measurable.
2. **`A_AMG-TP-only − A_TTAM` = +0.00037**, essentially identical to
   `without both` (+0.00038). The AMG-TP module on its own contributes
   ~7 × 10⁻⁶ — indistinguishable from just fitting a constant mixture.

So **the entire TTAM improvement over OPS is the AP-OPS calibration
layer**; the sample-gated / learned-persistence / deployed-weight-memory
historical module adds nothing on Criteo, with or without AP-OPS. **No
synergy.** The nested selection corroborates this independently: it picks
the *smallest* AMG-TP memory rate (`amgtp_rho = 0.2`) at all 15 origins —
the protocol turns the historical module down as far as the grid allows —
while keeping AP-OPS's adaptive mass high (`lambda_AP = 0.75`
everywhere).

## What this establishes

This is **AP-OPS re-confirmed under the corrected fully-nested protocol.**
The earlier AP-OPS result inherited its ONS / meta / mixture / OPS
settings from a frozen config whose development window overlapped early
outer days. Reselecting *every* knob from strictly pre-origin data, the
result stands: **AP-OPS beats plain OPS on Criteo, 15/15 origins, CI
excluding zero** — same small (~1.7e-4) magnitude, same directional
unanimity, now with no protocol objection.

Consistent with this repo's standing finding: real intra-month drift on
Criteo is shallow. The calibration-memory adaptation (AP-OPS) is the only
mechanism in the AMG-TP / twoscale / withinday / DualTime / AP-OPS line
that produces a reproducible edge; the historical short/long gating and
mixture-of-experts machinery does not add to it here.

## 6.4 Downstream bidding — the log-loss edge does not convert to clicks

Criteo bidding replay, 25% daily budget, clicks at matched spend, same
auction + pacing for every method, only the pCTR input changes
(`run_ttam_bidding.py`; recorded display `cost` as the price proxy).

| baseline | TTAM clicks | baseline clicks | aggregate click gain | 95% CI | origins won |
|---|---|---|---|---|---|
| Expanding | 133 882 | 133 964 | **−0.062 %** | [−0.086, −0.038] | 1/15 |
| Best Fixed Window | 133 882 | 133 956 | −0.055 % | [−0.082, −0.025] | 2/15 |
| ARW | 133 882 | 133 952 | −0.053 % | [−0.074, −0.030] | 2/15 |
| AdaMoE | 133 882 | 133 970 | −0.066 % | [−0.087, −0.045] | 0/15 |
| OPS | 133 882 | 133 913 | **−0.024 %** | [−0.031, −0.017] | 1/15 |

TTAM wins **fewer** clicks at matched spend than every baseline — small
(−0.02 % to −0.07 %, i.e. 30–90 clicks out of ~134 000) but every CI is
below zero. The tiny prediction-side calibration gain does **not** carry
through to downstream bidding value here; if anything it is marginally
negative. The plan's caution applies directly — do not read a small
log-loss improvement as a downstream win.

## Reproduce

```
# banks (one-time)
PYTHONPATH=. python3 final_experiments/ttam/build_banks.py \
    --source criteo --data data/criteo_attribution_dataset.tsv.gz \
    --cache-dir final_experiments/ttam/_bankcache --n-jobs 4
# nested run
PYTHONPATH=. python3 final_experiments/run_ttam_nested.py \
    --source criteo --data data/criteo_attribution_dataset.tsv.gz \
    --out final_experiments/ttam/criteo/nested --n-workers 4 --n-jobs 4
# section 6 outputs
bash final_experiments/run_ttam_section6.sh
```

or `sbatch final_experiments/ttam_nested_criteo.slurm`.

## Not done

- **Avazu** nested run (D = 5 origins; descriptive only). Needs the
  cluster or a low-memory bank build — `ttam_banks_avazu.slurm` /
  `ttam_nested_avazu.slurm`. Re-running `run_ttam_section6.sh` after it
  exists adds the Avazu panel to the stats + figure automatically.
- Manuscript placeholders — no paper source in this repo; this file + the
  figure are the deliverables.
