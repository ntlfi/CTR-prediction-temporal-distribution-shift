# Package 1 -- comprehensive fixed-beta sweep

beta grid 0.0..1.0 step 0.05, regimes ['s0_none', 's1_abrupt', 's2_gradual', 's3_recurring', 's4_local', 's5_opposing_local', 's6_mixed', 's7_opposing_recurring'], seeds [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(20), np.int64(21), np.int64(22), np.int64(23), np.int64(24), np.int64(25), np.int64(26), np.int64(27), np.int64(28), np.int64(29), np.int64(30), np.int64(31)] (17). Context gate + 5-expert bank held fixed; `fixed_beta_b` = pi = (1-b) q + b m. Excess = log loss minus the per-regime best fixed beta chosen in hindsight (so the oracle row is 0 by construction).

## Per-regime best fixed beta (hindsight) and its log loss

| regime | oracle beta | oracle log loss | AMG-TP | AMG-TP excess | beta=0 excess | global-beta(0) excess | ensemble3 excess |
|---|--:|--:|--:|--:|--:|--:|--:|
| s0_none | 0.00 | 0.3255 | 0.3266 | +0.0011 | +0.0000 | +0.0000 | -0.0001 |
| s1_abrupt | 0.00 | 0.3384 | 0.3391 | +0.0007 | +0.0000 | +0.0000 | +0.0010 |
| s2_gradual | 0.00 | 0.3591 | 0.3601 | +0.0011 | +0.0000 | +0.0000 | +0.0010 |
| s3_recurring | 0.80 | 0.4299 | 0.4284 | -0.0015 | +0.0087 | +0.0087 | -0.0015 |
| s4_local | 0.00 | 0.4581 | 0.4604 | +0.0022 | +0.0000 | +0.0000 | +0.0010 |
| s5_opposing_local | 0.00 | 0.4938 | 0.4952 | +0.0014 | +0.0000 | +0.0000 | +0.0010 |
| s6_mixed | 0.70 | 0.3975 | 0.3974 | -0.0001 | +0.0019 | +0.0019 | +0.0007 |
| s7_opposing_recurring | 0.50 | 0.4339 | 0.4345 | +0.0006 | +0.0004 | +0.0004 | -0.0022 |

## Headline: excess log loss vs the per-regime fixed-beta oracle
Paired bootstrap 95% CI over seed-level excess (pooled across regimes for the mean row; the max is the worst single regime's seed-mean excess).

| method | mean excess [95% CI] | worst-regime excess |
|---|---|--:|
| beta=0 | +0.0014 [-0.0013, +0.0040] | +0.0087 |
| global fixed beta=0 | +0.0014 [-0.0013, +0.0040] | +0.0087 |
| ensemble3 | +0.0001 [-0.0025, +0.0026] | +0.0010 |
| AMG-TP (adaptive) | +0.0007 [-0.0019, +0.0031] | +0.0022 |

## Interpretation
- If AMG-TP's **worst-regime excess** is below the global-fixed-beta row's, adaptive persistence buys real robustness (its stated revised claim).
- If AMG-TP's **mean excess** is ~0 it matches the per-regime oracle on average -- i.e. it is not merely interpolating between the two previously tested configs.
- ensemble3 is the strongest simple robustness alternative; AMG-TP should be close to it at a fraction of the inference cost (one gate vs three).