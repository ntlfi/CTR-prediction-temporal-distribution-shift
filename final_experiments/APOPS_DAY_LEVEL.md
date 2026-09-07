## Results (day-level inference: seeds averaged within day, then bootstrap over days)

### Criteo -- fixed test (days 22-30)  (D = 9 days: [22, 23, 24, 25, 26, 27, 28, 29, 30])

| row | seed-avg mean log loss | mean d vs OPS (day-wt) | imp-wt d | 95% CI (day bootstrap) | CI excl 0 | days won | sign p | worst-day d |
|---|---|---|---|---|---|---|---|---|
| Current OPS | 0.607217 | — | — | — | — | — | — | — |
| AP-OPS | 0.607092 | -0.000125 | -0.000125 | [-0.000165, -0.000083] | yes | 9/9 | 0.004 | -0.000029 |
| Single-memory | 0.607078 | -0.000139 | -0.000139 | [-0.000198, -0.000082] | yes | 9/9 | 0.004 | -0.000021 |
| No-slope | 0.607246 | +0.000029 | +0.000031 | [-0.000017, +0.000081] | no | 4/9 | 1.000 | +0.000189 |

Frozen `delta_NI = 3.251e-05`.
**Decision: Improves** (CI upper -0.0001008799946580645).

Temporal (pooled log loss, mean over seeds):

| row | pre-feedback | first quarter | worst day |
|---|---|---|---|
| Current OPS | 0.619245 | 0.614087 | 0.613835 |
| AP-OPS | 0.619172 | 0.613880 | 0.613787 |
| Single-memory | 0.619250 | 0.613921 | 0.613814 |
| No-slope | 0.619165 | 0.613791 | 0.613826 |

AP-OPS mechanism (seed 0): final weights R/S/L = 0.280 / 0.356 / 0.364; blocks where each dominates = 218 / 565 / 649; weight range [0.062, 0.868] over 1432 updates.

### Criteo -- rolling origin (16-30)  (D = 15 days: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])

| row | seed-avg mean log loss | mean d vs OPS (day-wt) | imp-wt d | 95% CI (day bootstrap) | CI excl 0 | days won | sign p | worst-day d |
|---|---|---|---|---|---|---|---|---|
| Current OPS | 0.608283 | — | — | — | — | — | — | — |
| AP-OPS | 0.608164 | -0.000119 | -0.000118 | [-0.000145, -0.000092] | yes | 15/15 | 6.1e-05 | -0.000030 |
| Single-memory | 0.608156 | -0.000127 | -0.000127 | [-0.000163, -0.000092] | yes | 15/15 | 6.1e-05 | -0.000021 |
| No-slope | 0.608320 | +0.000037 | +0.000040 | [-0.000006, +0.000081] | no | 6/15 | 0.607 | +0.000203 |

Frozen `delta_NI = 3.251e-05`.
**Decision: Improves** (CI upper -0.0002378099226211227).

### Avazu -- fixed test (days 7-9)  (D = 3 days: [7, 8, 9])

| row | seed-avg mean log loss | mean d vs OPS (day-wt) | imp-wt d | 95% CI (day bootstrap) | CI excl 0 | days won | sign p | worst-day d |
|---|---|---|---|---|---|---|---|---|
| Current OPS | 0.388489 | — | — | — | — | — | — | — |
| AP-OPS | 0.388219 | -0.000270 | -0.000272 | [-0.000296, -0.000235] | yes | 3/3 | 0.250 | -0.000235 |
| Single-memory | 0.388313 | -0.000176 | -0.000174 | [-0.000224, -0.000136] | yes | 3/3 | 0.250 | -0.000136 |
| No-slope | 0.388766 | +0.000277 | +0.000293 | [+0.000174, +0.000437] | yes | 0/3 | 0.250 | +0.000437 |

Frozen `delta_NI = 1.293e-05`.

Temporal (pooled log loss, mean over seeds):

| row | pre-feedback | first quarter | worst day |
|---|---|---|---|
| Current OPS | 0.408304 | 0.385674 | 0.406003 |
| AP-OPS | 0.407788 | 0.385359 | 0.405769 |
| Single-memory | 0.407837 | 0.385546 | 0.405867 |
| No-slope | 0.408205 | 0.385798 | 0.406224 |

AP-OPS mechanism (seed 0): final weights R/S/L = 0.340 / 0.330 / 0.330; blocks where each dominates = 28 / 16 / 76; weight range [0.252, 0.474] over 118 updates.

### Avazu -- rolling origin (5-9)  (D = 5 days: [5, 6, 7, 8, 9])

| row | seed-avg mean log loss | mean d vs OPS (day-wt) | imp-wt d | 95% CI (day bootstrap) | CI excl 0 | days won | sign p | worst-day d |
|---|---|---|---|---|---|---|---|---|
| Current OPS | 0.401655 | — | — | — | — | — | — | — |
| AP-OPS | 0.401373 | -0.000281 | -0.000277 | [-0.000356, -0.000217] | yes | 5/5 | 0.062 | -0.000194 |
| Single-memory | 0.401428 | -0.000227 | -0.000215 | [-0.000337, -0.000146] | yes | 5/5 | 0.062 | -0.000110 |
| No-slope | 0.401786 | +0.000131 | +0.000163 | [-0.000039, +0.000302] | no | 2/5 | 1.000 | +0.000437 |

Frozen `delta_NI = 1.293e-05`.

