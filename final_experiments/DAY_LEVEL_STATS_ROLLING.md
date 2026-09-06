# Day-level significance (seeds averaged first, then day-level inference)

Review comment 1. `Lbar_{m,d}` averages the 3 seeds on each calendar
day; inference is then over the D calendar days (the exchangeable unit
for a temporal claim), **not** the 27 / 9 pooled (seed, day) cells the
headline table used. Deltas are equal-day-weighted mean log-loss
differences; negative favours the method. Bootstrap CI is the
percentile bootstrap over the D days; `sign p` is the two-sided
sign test; `p floor` is the smallest two-sided sign-test p attainable
at this D (a clean sweep).

## Criteo rolling-origin (15 origins, days 16-30)  (D = 15 days: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])

### vs `expanding`

| method | mean d (day-wt) | imp-wt d | 95% CI (day bootstrap) | CI excl 0 | W-L-T | sign p | p floor | seed spread |
|---|---|---|---|---|---|---|---|---|
| ops | -0.001174 | -0.001157 | [-0.001369, -0.001007] | yes | 15-0-0 | 6.1e-05 | 6.1e-05 | 0.000112 |
| dualtime | -0.001058 | -0.001040 | [-0.001251, -0.000893] | yes | 15-0-0 | 6.1e-05 | 6.1e-05 | 0.000174 |
| adamoe | -0.000934 | -0.000921 | [-0.001060, -0.000823] | yes | 15-0-0 | 6.1e-05 | 6.1e-05 | 0.000059 |
| long_only | -0.000918 | -0.000903 | [-0.001054, -0.000798] | yes | 15-0-0 | 6.1e-05 | 6.1e-05 | 0.000082 |
| arw | -0.000843 | -0.000830 | [-0.001009, -0.000692] | yes | 15-0-0 | 6.1e-05 | 6.1e-05 | 0.000190 |
| best_fixed | -0.000755 | -0.000731 | [-0.000968, -0.000547] | yes | 14-1-0 | 9.8e-04 | 6.1e-05 | 0.000280 |

### vs `long_only`

| method | mean d (day-wt) | imp-wt d | 95% CI (day bootstrap) | CI excl 0 | W-L-T | sign p | p floor | seed spread |
|---|---|---|---|---|---|---|---|---|
| ops | -0.000257 | -0.000254 | [-0.000353, -0.000164] | yes | 15-0-0 | 6.1e-05 | 6.1e-05 | 0.000105 |
| dualtime | -0.000140 | -0.000137 | [-0.000227, -0.000058] | yes | 12-3-0 | 0.035 | 6.1e-05 | 0.000136 |
| adamoe | -0.000017 | -0.000017 | [-0.000043, +0.000005] | no | 7-8-0 | 1.000 | 6.1e-05 | 0.000046 |
| arw | +0.000074 | +0.000074 | [+0.000023, +0.000131] | yes | 3-12-0 | 0.035 | 6.1e-05 | 0.000108 |
| best_fixed | +0.000162 | +0.000173 | [+0.000057, +0.000285] | yes | 4-11-0 | 0.118 | 6.1e-05 | 0.000205 |
| expanding | +0.000918 | +0.000903 | [+0.000798, +0.001054] | yes | 0-15-0 | 6.1e-05 | 6.1e-05 | 0.000082 |

## Avazu rolling-origin (5 origins, days 5-9)  (D = 5 days: [5, 6, 7, 8, 9])

### vs `expanding`

| method | mean d (day-wt) | imp-wt d | 95% CI (day bootstrap) | CI excl 0 | W-L-T | sign p | p floor | seed spread |
|---|---|---|---|---|---|---|---|---|
| ops | -0.000458 | -0.000395 | [-0.000763, -0.000058] | yes | 4-1-0 | 0.375 | 0.062 | 0.000041 |
| dualtime | -0.000349 | -0.000276 | [-0.000668, +0.000106] | no | 4-1-0 | 0.375 | 0.062 | 0.000057 |
| adamoe | -0.000335 | -0.000301 | [-0.000550, -0.000151] | yes | 5-0-0 | 0.062 | 0.062 | 0.000115 |
| arw | -0.000033 | +0.000122 | [-0.000883, +0.000876] | no | 3-1-1 | 0.625 | 0.062 | 0.000303 |
| long_only | -0.000018 | +0.000111 | [-0.000550, +0.000802] | no | 4-1-0 | 0.375 | 0.062 | 0.000268 |
| best_fixed | +0.000017 | +0.000173 | [-0.000837, +0.000921] | no | 2-2-1 | 1.000 | 0.062 | 0.000303 |

### vs `long_only`

| method | mean d (day-wt) | imp-wt d | 95% CI (day bootstrap) | CI excl 0 | W-L-T | sign p | p floor | seed spread |
|---|---|---|---|---|---|---|---|---|
| ops | -0.000440 | -0.000506 | [-0.000850, -0.000140] | yes | 5-0-0 | 0.062 | 0.062 | 0.000227 |
| dualtime | -0.000331 | -0.000387 | [-0.000675, -0.000096] | yes | 5-0-0 | 0.062 | 0.062 | 0.000227 |
| adamoe | -0.000317 | -0.000411 | [-0.000952, +0.000000] | no | 2-3-0 | 1.000 | 0.062 | 0.000153 |
| arw | -0.000015 | +0.000012 | [-0.000376, +0.000232] | no | 1-3-1 | 0.625 | 0.062 | 0.000087 |
| expanding | +0.000018 | -0.000111 | [-0.000802, +0.000550] | no | 1-4-0 | 0.375 | 0.062 | 0.000268 |
| best_fixed | +0.000035 | +0.000063 | [-0.000371, +0.000328] | no | 1-3-1 | 0.625 | 0.062 | 0.000098 |
