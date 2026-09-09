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
