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

### 6.3 Ablation (gain of TTAM over each ablation)

| variant | mean score A_m | mean gain G_m | 95% day-bootstrap CI | block-2 MBB CI | origins won by TTAM |
|---|---|---|---|---|---|
| Without both modules | 0.608408 | +0.000379 | [+0.000298, +0.000473] | [+0.000290, +0.000471] | 15/15 |
| AMG-TP only | 0.608401 | +0.000372 | [+0.000291, +0.000465] | [+0.000284, +0.000461] | 15/15 |
| AP-OPS only | 0.608031 | +0.000002 | [-0.000001, +0.000005] | [-0.000001, +0.000005] | 11/15 |

### 6.3b Internal contrasts (mean difference `a - b`, paired day-bootstrap)

Not anchored on TTAM. `a - b` negative => `a` has the lower loss; the
CI is the 95% paired day-bootstrap interval for that mean difference.
"excl. 0" means the interval does not contain zero -- a *detectable*
ordering on these origins, which at magnitudes <1e-4 need not be a
*material* one.

| a | b | mean `a - b` | 95% day-bootstrap CI | block-2 MBB CI | `a` better on | excl. 0 |
|---|---|---|---|---|---|---|
| AP-OPS only | OPS | -1.70e-04 | [-2.17e-04, -1.30e-04] | [-2.11e-04, -1.31e-04] | 15/15 | yes |
| AMG-TP only | Without both modules | -7.43e-06 | [-1.44e-05, -8.23e-07] | [-1.47e-05, +6.37e-08] | 10/15 | yes |
| TTAM | AP-OPS only | -2.23e-06 | [-5.25e-06, +1.00e-06] | [-4.89e-06, +1.00e-06] | 11/15 | no |
| TTAM | AMG-TP only | -3.72e-04 | [-4.65e-04, -2.91e-04] | [-4.61e-04, -2.84e-04] | 15/15 | yes |
| Without both modules | AdaMoE | -1.74e-06 | [-3.19e-06, -4.74e-07] | [-3.32e-06, -5.35e-07] | 13/15 | yes |

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

### 6.3 Ablation (gain of TTAM over each ablation)

| variant | mean score A_m | mean gain G_m | 95% day-bootstrap CI | block-2 MBB CI | origins won by TTAM |
|---|---|---|---|---|---|
| Without both modules | 0.401535 | +0.000600 | [+0.000332, +0.000903] | [+0.000332, +0.000886] | 5/5 |
| AMG-TP only | 0.401514 | +0.000580 | [+0.000305, +0.000897] | [+0.000304, +0.000871] | 5/5 |
| AP-OPS only | 0.400940 | +0.000006 | [-0.000018, +0.000031] | [-0.000013, +0.000024] | 4/5 |

### 6.3b Internal contrasts (mean difference `a - b`, paired day-bootstrap)

Not anchored on TTAM. `a - b` negative => `a` has the lower loss; the
CI is the 95% paired day-bootstrap interval for that mean difference.
"excl. 0" means the interval does not contain zero -- a *detectable*
ordering on these origins, which at magnitudes <1e-4 need not be a
*material* one.

| a | b | mean `a - b` | 95% day-bootstrap CI | block-2 MBB CI | `a` better on | excl. 0 |
|---|---|---|---|---|---|---|
| AP-OPS only | OPS | -2.07e-04 | [-2.77e-04, -1.31e-04] | [-2.43e-04, -1.50e-04] | 5/5 | yes |
| AMG-TP only | Without both modules | -2.08e-05 | [-4.00e-05, -2.48e-06] | [-3.13e-05, -5.83e-06] | 4/5 | yes |
| TTAM | AP-OPS only | -5.56e-06 | [-3.13e-05, +1.80e-05] | [-2.44e-05, +1.28e-05] | 4/5 | no |
| TTAM | AMG-TP only | -5.80e-04 | [-8.97e-04, -3.05e-04] | [-8.71e-04, -3.04e-04] | 5/5 | yes |
| Without both modules | AdaMoE | -3.23e-06 | [-1.01e-05, +3.55e-06] | [-1.06e-05, +2.94e-06] | 4/5 | no |
