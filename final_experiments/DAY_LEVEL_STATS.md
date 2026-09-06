# Day-level significance (seeds averaged first, then day-level inference)

Review comment 1. `Lbar_{m,d}` averages the 3 seeds on each calendar
day; inference is then over the D calendar days (the exchangeable unit
for a temporal claim), **not** the 27 / 9 pooled (seed, day) cells the
headline table used. Deltas are equal-day-weighted mean log-loss
differences; negative favours the method. Bootstrap CI is the
percentile bootstrap over the D days; `sign p` is the two-sided
sign test; `p floor` is the smallest two-sided sign-test p attainable
at this D (a clean sweep).

## Criteo (fixed origin, test days 22-30)  (D = 9 days: [22, 23, 24, 25, 26, 27, 28, 29, 30])

### vs `expanding`

| method | mean d (day-wt) | imp-wt d | 95% CI (day bootstrap) | CI excl 0 | W-L-T | sign p | p floor | seed spread |
|---|---|---|---|---|---|---|---|---|
| ops | -0.001120 | -0.001109 | [-0.001316, -0.000935] | yes | 9-0-0 | 0.004 | 0.004 | 0.000013 |
| dualtime | -0.001010 | -0.000997 | [-0.001216, -0.000831] | yes | 9-0-0 | 0.004 | 0.004 | 0.000060 |
| adamoe | -0.000922 | -0.000911 | [-0.001080, -0.000780] | yes | 9-0-0 | 0.004 | 0.004 | 0.000064 |
| best_fixed | -0.000790 | -0.000777 | [-0.001007, -0.000594] | yes | 9-0-0 | 0.004 | 0.004 | 0.000078 |
| arw | -0.000782 | -0.000766 | [-0.001009, -0.000579] | yes | 9-0-0 | 0.004 | 0.004 | 0.000089 |

## Avazu (fixed origin, test days 7-9)  (D = 3 days: [7, 8, 9])

### vs `expanding`

| method | mean d (day-wt) | imp-wt d | 95% CI (day bootstrap) | CI excl 0 | W-L-T | sign p | p floor | seed spread |
|---|---|---|---|---|---|---|---|---|
| adamoe | -0.000211 | -0.000194 | [-0.000363, -0.000053] | yes | 3-0-0 | 0.250 | 0.250 | 0.000110 |
| ops | -0.000201 | -0.000153 | [-0.000465, +0.000311] | no | 2-1-0 | 1.000 | 0.250 | 0.000176 |
| arw | -0.000076 | -0.000066 | [-0.000224, +0.000000] | no | 2-0-1 | 0.500 | 0.250 | 0.000162 |
| dualtime | -0.000041 | +0.000014 | [-0.000338, +0.000538] | no | 2-1-0 | 1.000 | 0.250 | 0.000228 |
| best_fixed | +0.000443 | +0.000551 | [-0.000261, +0.001535] | no | 1-2-0 | 1.000 | 0.250 | 0.000457 |

> **3 days is too few for a day-level bootstrap CI or a significant sign test.** Even a clean sweep gives two-sided p = 0.250. Treat these rows as descriptive; the temporal significance statement for this dataset has to come from the rolling-origin run (more origins) or a fresh chronological stream.
