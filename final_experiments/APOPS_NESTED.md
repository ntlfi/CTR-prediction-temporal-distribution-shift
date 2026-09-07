## Nested rolling-origin results (day-level inference: seeds averaged within day, then bootstrap over days)

### Criteo -- nested rolling origin (16-30)  (D = 15 days: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])

| row | seed-avg mean log loss | mean d vs OPS (day-wt) | imp-wt d | 95% CI (day bootstrap) | CI excl 0 | days won | sign p | worst-day d |
|---|---|---|---|---|---|---|---|---|
| OPS | 0.608288 | — | — | — | — | — | — | — |
| Reset-ONS | 0.608120 | -0.000167 | -0.000168 | [-0.000218, -0.000125] | yes | 15/15 | 6.1e-05 | -0.000024 |
| Persistent-ONS | 0.608118 | -0.000170 | -0.000170 | [-0.000220, -0.000128] | yes | 15/15 | 6.1e-05 | -0.000031 |
| AP-OPS | 0.608116 | -0.000172 | -0.000172 | [-0.000220, -0.000131] | yes | 15/15 | 6.1e-05 | -0.000046 |
| No-slope | 0.608285 | -0.000002 | +0.000001 | [-0.000053, +0.000047] | no | 6/15 | 0.607 | +0.000155 |


**Decision rules:**

- Cross-day persistence: Persistent-ONS -0.000170 vs Reset-ONS -0.000167 (vs OPS; Reset-ONS CI excl 0: True) -> indistinguishable from Reset-ONS -- the ONS optimizer, not persistence, carries the gain.
- Adaptive aggregation: AP-OPS -0.000172 vs Persistent-ONS -0.000170 -> matches the single persistent expert (no material difference here).
- AP-OPS vs OPS: -0.000172, CI [-0.000220, -0.000131], 15/15 origins -> beats OPS (CI excl 0).
- Extension used: lambda_AP > 0 on 15/15 origins (fraction 1.00); per-origin lambda_AP = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.25, 0.25, 0.25].

### Avazu -- nested rolling origin (5-9)  (D = 5 days: [5, 6, 7, 8, 9])

| row | seed-avg mean log loss | mean d vs OPS (day-wt) | imp-wt d | 95% CI (day bootstrap) | CI excl 0 | days won | sign p | worst-day d |
|---|---|---|---|---|---|---|---|---|
| OPS | 0.401636 | — | — | — | — | — | — | — |
| Reset-ONS | 0.401655 | +0.000019 | +0.000034 | [-0.000090, +0.000133] | no | 3/5 | 1.000 | +0.000192 |
| Persistent-ONS | 0.401587 | -0.000049 | -0.000034 | [-0.000143, +0.000057] | no | 3/5 | 1.000 | +0.000132 |
| AP-OPS | 0.401500 | -0.000136 | -0.000137 | [-0.000181, -0.000098] | yes | 5/5 | 0.062 | -0.000073 |
| No-slope | 0.401865 | +0.000229 | +0.000258 | [+0.000081, +0.000397] | yes | 0/5 | 0.062 | +0.000553 |


**Decision rules:**

- Cross-day persistence: Persistent-ONS -0.000049 vs Reset-ONS +0.000019 (vs OPS; Reset-ONS CI excl 0: False) -> SUPPORTED -- Persistent-ONS materially beats Reset-ONS.
- Adaptive aggregation: AP-OPS -0.000136 vs Persistent-ONS -0.000049 -> SUPPORTED -- AP-OPS materially beats the single persistent expert.
- AP-OPS vs OPS: -0.000136, CI [-0.000181, -0.000098], 5/5 origins -> beats OPS (CI excl 0).
- Extension used: lambda_AP > 0 on 5/5 origins (fraction 1.00); per-origin lambda_AP = [0.25, 0.25, 0.5, 0.5, 0.5].

