# TTAM revised Section 6 -- paired day-level statistics

Seeds are averaged within each origin day *before* any interval is
formed; the origin day is the unit. Gain `G_{m,d} = A_{m,d} -
A_{TTAM,d}` is positive when TTAM has the lower loss. Bootstrap:
10,000 paired day resamples, resampling seed 20260908;
`mbb` is the block-2 moving-block bootstrap (serial-dependence
sensitivity). Impression-weighted loss / Brier / ECE are appendix
metrics only (see the CSVs) and are not mixed with the equal-day gain.

## Criteo  (D = 15 origins: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])

### 6.2 Main comparison (gain of TTAM over each)

| method | mean score A_m | mean gain G_m | 95% day-bootstrap CI | block-2 MBB CI | origins won by TTAM |
|---|---|---|---|---|---|
| Expanding | 0.609475 | +0.001446 | [+0.001290, +0.001628] | [+0.001249, +0.001654] | 15/15 |
| Best Fixed Window | 0.608744 | +0.000715 | [+0.000595, +0.000850] | [+0.000588, +0.000837] | 15/15 |
| ARW | 0.608631 | +0.000603 | [+0.000525, +0.000678] | [+0.000504, +0.000686] | 15/15 |
| AdaMoE | 0.608410 | +0.000381 | [+0.000300, +0.000475] | [+0.000292, +0.000473] | 15/15 |
| OPS | 0.608201 | +0.000172 | [+0.000133, +0.000219] | [+0.000134, +0.000212] | 15/15 |
| TTAM | 0.608029 | (reference) | -- | -- | -- |

### 6.3 Ablation -- 2x2: historical predictor x calibration

Mean score A_m per cell (lower = better):

| historical \ calibration | none | AP-OPS |
|---|---|---|
| **expanding history** | 0.609475 | 0.608807 |
| **AMG-TP** | 0.608401 | 0.608029 (= TTAM) |

Marginal effects (mean difference, negative => the added component lowers loss;
95% paired day-bootstrap CI; "excl. 0" = interval does not contain zero):

| effect | at | mean | 95% day-bootstrap CI | lower on | excl. 0 |
|---|---|---|---|---|---|
| AP-OPS (calibration) | historical = expanding | -6.68e-04 | [-8.74e-04, -5.04e-04] | 15/15 | yes |
| AP-OPS (calibration) | historical = AMG-TP | -3.72e-04 | [-4.65e-04, -2.91e-04] | 15/15 | yes |
| AMG-TP (historical) | calibration = none | -1.07e-03 | [-1.23e-03, -9.26e-04] | 15/15 | yes |
| AMG-TP (historical) | calibration = AP-OPS | -7.78e-04 | [-8.58e-04, -7.04e-04] | 15/15 | yes |
| interaction | (t-a) - (ea-e) | +2.96e-04 | [+1.46e-04, +4.58e-04] | 1/15 | yes |

Interaction < 0 would mean AMG-TP makes AP-OPS help more (synergy); a CI containing 0 means the two timescales are additive.

## Avazu  (D = 5 origins: [5, 6, 7, 8, 9])

> Only 5 origins -- both interval estimates are descriptive and fragile. No significance claims; unfavourable origins are retained.

### 6.2 Main comparison (gain of TTAM over each)

| method | mean score A_m | mean gain G_m | 95% day-bootstrap CI | block-2 MBB CI | origins won by TTAM |
|---|---|---|---|---|---|
| Expanding | 0.402094 | +0.001160 | [+0.000925, +0.001402] | [+0.001035, +0.001340] | 5/5 |
| Best Fixed Window | 0.402213 | +0.001279 | [+0.000204, +0.002682] | [+0.000080, +0.002559] | 4/5 |
| ARW | 0.402359 | +0.001425 | [+0.000286, +0.002695] | [+0.000080, +0.002713] | 4/5 |
| AdaMoE | 0.401538 | +0.000604 | [+0.000344, +0.000897] | [+0.000344, +0.000883] | 5/5 |
| OPS | 0.401147 | +0.000213 | [+0.000137, +0.000281] | [+0.000165, +0.000248] | 5/5 |
| TTAM | 0.400934 | (reference) | -- | -- | -- |

### 6.3 Ablation -- 2x2: historical predictor x calibration

Mean score A_m per cell (lower = better):

| historical \ calibration | none | AP-OPS |
|---|---|---|
| **expanding history** | 0.402094 | 0.401737 |
| **AMG-TP** | 0.401514 | 0.400934 (= TTAM) |

Marginal effects (mean difference, negative => the added component lowers loss;
95% paired day-bootstrap CI; "excl. 0" = interval does not contain zero):

| effect | at | mean | 95% day-bootstrap CI | lower on | excl. 0 |
|---|---|---|---|---|---|
| AP-OPS (calibration) | historical = expanding | -3.57e-04 | [-5.48e-04, -1.81e-04] | 5/5 | yes |
| AP-OPS (calibration) | historical = AMG-TP | -5.80e-04 | [-8.97e-04, -3.05e-04] | 5/5 | yes |
| AMG-TP (historical) | calibration = none | -5.80e-04 | [-9.18e-04, -2.71e-04] | 5/5 | yes |
| AMG-TP (historical) | calibration = AP-OPS | -8.02e-04 | [-1.13e-03, -5.35e-04] | 5/5 | yes |
| interaction | (t-a) - (ea-e) | -2.22e-04 | [-3.33e-04, -1.12e-04] | 5/5 | yes |

Interaction < 0 would mean AMG-TP makes AP-OPS help more (synergy); a CI containing 0 means the two timescales are additive.
