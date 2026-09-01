# Package 1 -- comprehensive fixed-beta sweep

beta grid 0.0..1.0 step 0.05, regimes ['s0_none', 's1_abrupt', 's2_gradual', 's3_recurring', 's4_local', 's5_opposing_local', 's6_mixed', 's7_opposing_recurring', 's3b_recurring_p9', 's3c_recurring_p11', 's3d_recurring_p17', 's3e_recurring_p21', 's8_irregular_recurring', 's9_heterogeneous'], seeds [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(20), np.int64(21), np.int64(22), np.int64(23), np.int64(24), np.int64(25), np.int64(26), np.int64(27), np.int64(28), np.int64(29), np.int64(30), np.int64(31)] (17). Context gate + 5-expert bank held fixed; `fixed_beta_b` = pi = (1-b) q + b m. Excess = log loss minus the per-regime best fixed beta chosen in hindsight (so the oracle row is 0 by construction).

## Per-regime best fixed beta (hindsight) and its log loss

| regime | oracle beta | oracle log loss | AMG-TP | AMG-TP excess | beta=0 excess | global-beta(0.8) excess | ensemble3 excess |
|---|--:|--:|--:|--:|--:|--:|--:|
| s0_none | 0.00 | 0.3255 | 0.3266 | +0.0011 | +0.0000 | +0.0046 | -0.0001 |
| s1_abrupt | 0.00 | 0.3384 | 0.3391 | +0.0007 | +0.0000 | +0.0105 | +0.0010 |
| s2_gradual | 0.00 | 0.3591 | 0.3601 | +0.0011 | +0.0000 | +0.0054 | +0.0010 |
| s3_recurring | 0.80 | 0.4299 | 0.4284 | -0.0015 | +0.0087 | +0.0000 | -0.0015 |
| s4_local | 0.00 | 0.4581 | 0.4604 | +0.0022 | +0.0000 | +0.0254 | +0.0010 |
| s5_opposing_local | 0.00 | 0.4938 | 0.4952 | +0.0014 | +0.0000 | +0.0066 | +0.0010 |
| s6_mixed | 0.70 | 0.3975 | 0.3974 | -0.0001 | +0.0019 | +0.0017 | +0.0007 |
| s7_opposing_recurring | 0.50 | 0.4339 | 0.4345 | +0.0006 | +0.0004 | +0.0034 | -0.0022 |
| s3b_recurring_p9 | 0.65 | 0.4347 | 0.4349 | +0.0002 | +0.0009 | +0.0012 | -0.0017 |
| s3c_recurring_p11 | 0.80 | 0.4339 | 0.4340 | +0.0001 | +0.0050 | +0.0000 | -0.0006 |
| s3d_recurring_p17 | 0.80 | 0.4244 | 0.4249 | +0.0004 | +0.0147 | +0.0000 | -0.0004 |
| s3e_recurring_p21 | 0.80 | 0.4225 | 0.4235 | +0.0010 | +0.0192 | +0.0000 | -0.0015 |
| s8_irregular_recurring | 0.90 | 0.4491 | 0.4601 | +0.0110 | +0.0408 | +0.0038 | +0.0232 |
| s9_heterogeneous | 0.85 | 0.4327 | 0.4405 | +0.0078 | +0.0168 | +0.0026 | +0.0109 |

## Headline: excess log loss vs the per-regime fixed-beta oracle
Paired bootstrap 95% CI over seed-level excess (pooled across regimes for the mean row; the max is the worst single regime's seed-mean excess).

| method | mean excess [95% CI] | worst-regime excess |
|---|---|--:|
| beta=0 | +0.0077 [+0.0053, +0.0102] | +0.0408 |
| global fixed beta=0.8 | +0.0047 [+0.0028, +0.0065] | +0.0254 |
| ensemble3 | +0.0022 [+0.0002, +0.0043] | +0.0232 |
| AMG-TP (adaptive) | +0.0019 [-0.0001, +0.0038] | +0.0110 |

## Interpretation
- If AMG-TP's **worst-regime excess** is below the global-fixed-beta row's, adaptive persistence buys real robustness (its stated revised claim).
- If AMG-TP's **mean excess** is ~0 it matches the per-regime oracle on average -- i.e. it is not merely interpolating between the two previously tested configs.
- ensemble3 is the strongest simple robustness alternative; AMG-TP should be close to it at a fraction of the inference cost (one gate vs three).