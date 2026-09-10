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
| Expanding | 0.609525 | +0.001496 | [+0.001314, +0.001715] | [+0.001267, +0.001744] | 15/15 |
| Best Fixed Window | 0.608713 | +0.000684 | [+0.000551, +0.000863] | [+0.000574, +0.000818] | 15/15 |
| ARW | 0.608643 | +0.000614 | [+0.000522, +0.000714] | [+0.000504, +0.000723] | 15/15 |
| AdaMoE | 0.608464 | +0.000435 | [+0.000320, +0.000586] | [+0.000317, +0.000588] | 15/15 |
| OPS | 0.608203 | +0.000174 | [+0.000136, +0.000220] | [+0.000137, +0.000213] | 15/15 |
| TTAM | 0.608029 | (reference) | -- | -- | -- |

### 6.3 Ablation -- 2x2: historical predictor x calibration

Mean score A_m per cell (lower = better):

| historical \ calibration | none | AP-OPS |
|---|---|---|
| **expanding history** | 0.609525 | 0.608833 |
| **AMG-TP** | 0.608457 | 0.608029 (= TTAM) |

Marginal effects (mean difference, negative => the added component lowers loss;
95% paired day-bootstrap CI; "excl. 0" = interval does not contain zero):

| effect | at | mean | 95% day-bootstrap CI | lower on | excl. 0 |
|---|---|---|---|---|---|
| AP-OPS (calibration) | historical = expanding | -6.92e-04 | [-9.33e-04, -5.06e-04] | 15/15 | yes |
| AP-OPS (calibration) | historical = AMG-TP | -4.28e-04 | [-5.77e-04, -3.13e-04] | 15/15 | yes |
| AMG-TP (historical) | calibration = none | -1.07e-03 | [-1.20e-03, -9.37e-04] | 15/15 | yes |
| AMG-TP (historical) | calibration = AP-OPS | -8.04e-04 | [-8.79e-04, -7.38e-04] | 15/15 | yes |
| interaction | (t-a) - (ea-e) | +2.64e-04 | [+1.37e-04, +4.00e-04] | 1/15 | yes |

Interaction < 0 would mean AMG-TP makes AP-OPS help more (synergy); a CI containing 0 means the two timescales are additive.

## Avazu  (D = 5 origins: [5, 6, 7, 8, 9])

> Only 5 origins -- both interval estimates are descriptive and fragile. No significance claims; unfavourable origins are retained.

### 6.2 Main comparison (gain of TTAM over each)

| method | mean score A_m | mean gain G_m | 95% day-bootstrap CI | block-2 MBB CI | origins won by TTAM |
|---|---|---|---|---|---|
| Expanding | 0.402094 | +0.001157 | [+0.000923, +0.001400] | [+0.001033, +0.001337] | 5/5 |
| Best Fixed Window | 0.402213 | +0.001277 | [+0.000202, +0.002679] | [+0.000078, +0.002555] | 4/5 |
| ARW | 0.402359 | +0.001423 | [+0.000284, +0.002690] | [+0.000078, +0.002709] | 4/5 |
| AdaMoE | 0.401538 | +0.000601 | [+0.000342, +0.000894] | [+0.000342, +0.000880] | 5/5 |
| OPS | 0.401147 | +0.000210 | [+0.000135, +0.000279] | [+0.000162, +0.000246] | 5/5 |
| TTAM | 0.400937 | (reference) | -- | -- | -- |

### 6.3 Ablation -- 2x2: historical predictor x calibration

Mean score A_m per cell (lower = better):

| historical \ calibration | none | AP-OPS |
|---|---|---|
| **expanding history** | 0.402094 | 0.401742 |
| **AMG-TP** | 0.401514 | 0.400937 (= TTAM) |

Marginal effects (mean difference, negative => the added component lowers loss;
95% paired day-bootstrap CI; "excl. 0" = interval does not contain zero):

| effect | at | mean | 95% day-bootstrap CI | lower on | excl. 0 |
|---|---|---|---|---|---|
| AP-OPS (calibration) | historical = expanding | -3.52e-04 | [-5.42e-04, -1.76e-04] | 5/5 | yes |
| AP-OPS (calibration) | historical = AMG-TP | -5.77e-04 | [-8.94e-04, -3.03e-04] | 5/5 | yes |
| AMG-TP (historical) | calibration = none | -5.80e-04 | [-9.18e-04, -2.71e-04] | 5/5 | yes |
| AMG-TP (historical) | calibration = AP-OPS | -8.05e-04 | [-1.14e-03, -5.31e-04] | 5/5 | yes |
| interaction | (t-a) - (ea-e) | -2.25e-04 | [-3.35e-04, -1.14e-04] | 5/5 | yes |

Interaction < 0 would mean AMG-TP makes AP-OPS help more (synergy); a CI containing 0 means the two timescales are additive.
