# Day-level significance (seeds averaged first, then day-level inference)

Review comment 1. `Lbar_{m,d}` averages the 3 seeds on each calendar
day; inference is then over the D calendar days (the exchangeable unit
for a temporal claim), **not** the 27 / 9 pooled (seed, day) cells the
headline table used. Deltas are equal-day-weighted mean log-loss
differences; negative favours the method. Bootstrap CI is the
percentile bootstrap over the D days; `sign p` is the two-sided
sign test; `p floor` is the smallest two-sided sign-test p attainable
at this D (a clean sweep).

## Criteo diagnostic (dev evidence)  (D = 9 days: [22, 23, 24, 25, 26, 27, 28, 29, 30])

### vs `long_only`

| method | mean d (day-wt) | imp-wt d | 95% CI (day bootstrap) | CI excl 0 | W-L-T | sign p | p floor | seed spread |
|---|---|---|---|---|---|---|---|---|
| ops | -0.000185 | -0.000187 | [-0.000297, -0.000072] | yes | 8-1-0 | 0.039 | 0.004 | 0.000070 |
| dualtime | -0.000076 | -0.000075 | [-0.000167, +0.000015] | no | 6-3-0 | 0.508 | 0.004 | 0.000108 |
| frozen_v5 | -0.000057 | -0.000058 | [-0.000098, -0.000014] | yes | 7-2-0 | 0.180 | 0.004 | 0.000145 |

## Avazu diagnostic (dev evidence)  (D = 3 days: [7, 8, 9])

### vs `long_only`

| method | mean d (day-wt) | imp-wt d | 95% CI (day bootstrap) | CI excl 0 | W-L-T | sign p | p floor | seed spread |
|---|---|---|---|---|---|---|---|---|
| frozen_v5 | -0.000684 | -0.000748 | [-0.001322, -0.000227] | yes | 3-0-0 | 0.250 | 0.250 | 0.000206 |
| ops | -0.000654 | -0.000711 | [-0.001224, -0.000277] | yes | 3-0-0 | 0.250 | 0.250 | 0.000296 |
| dualtime | -0.000494 | -0.000544 | [-0.000996, -0.000152] | yes | 3-0-0 | 0.250 | 0.250 | 0.000286 |

> **3 days is too few for a day-level bootstrap CI or a significant sign test.** Even a clean sweep gives two-sided p = 0.250. Treat these rows as descriptive; the temporal significance statement for this dataset has to come from the rolling-origin run (more origins) or a fresh chronological stream.
