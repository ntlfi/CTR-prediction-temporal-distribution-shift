# Stage 4 -- revised-claim evaluation

Regimes (14): ['s0_none', 's1_abrupt', 's2_gradual', 's3_recurring', 's4_local', 's5_opposing_local', 's6_mixed', 's7_opposing_recurring', 's3b_recurring_p9', 's3c_recurring_p11', 's3d_recurring_p17', 's3e_recurring_p21', 's8_irregular_recurring', 's9_heterogeneous']. Seeds: [0, 1, 2, 3, 4, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] (17). 
Excess = locked-test log loss minus the per-regime best fixed beta (chosen in hindsight from the dense 0..1 sweep). Paired over seeds.

Globally best single fixed beta (over all regimes x seeds): **0.8**.

## Excess vs the per-regime fixed-beta oracle (seed-level, paired bootstrap 95% CI)

| method | mean excess [95% CI] | worst-regime excess [95% CI] |
|---|---|---|
| amgtp | +0.0019 [+0.0014, +0.0023] | +0.0166 [+0.0128, +0.0208] |
| m5b_smooth0.1 | +0.0022 [+0.0018, +0.0025] | +0.0195 [+0.0167, +0.0225] |
| ensemble3 | +0.0022 [+0.0016, +0.0029] | +0.0246 [+0.0203, +0.0290] |
| global_fixed_beta | +0.0047 [+0.0044, +0.0049] | +0.0254 [+0.0248, +0.0260] |
| learn_alpha | +0.0029 [+0.0024, +0.0032] | +0.0273 [+0.0266, +0.0281] |
| fixed_share | +0.0040 [+0.0036, +0.0044] | +0.0275 [+0.0268, +0.0283] |
| han_arw | +0.0059 [+0.0053, +0.0066] | +0.0305 [+0.0291, +0.0324] |
| adamoe | +0.0175 [+0.0169, +0.0181] | +0.0428 [+0.0420, +0.0436] |
| m5b_smooth0.001 | +0.0080 [+0.0070, +0.0090] | +0.0468 [+0.0388, +0.0553] |
| m2_context_gate | +0.0191 [+0.0183, +0.0199] | +0.0608 [+0.0597, +0.0624] |
| expanding | +0.0519 [+0.0499, +0.0541] | +0.2037 [+0.1974, +0.2116] |

## Per-regime AMG-TP minus reference (Wilcoxon, Holm-corrected across regimes)
Negative = AMG-TP better.

| reference | regime | mean Δ | Wilcoxon p | Holm p |
|---|---|--:|--:|--:|
| global_fixed_beta | s0_none | -0.0035 | 1.53e-05 | 0.000214 |
| global_fixed_beta | s1_abrupt | -0.0098 | 1.53e-05 | 0.000214 |
| global_fixed_beta | s2_gradual | -0.0044 | 1.53e-05 | 0.000214 |
| global_fixed_beta | s3_recurring | -0.0015 | 7.63e-05 | 0.000534 |
| global_fixed_beta | s4_local | -0.0231 | 1.53e-05 | 0.000214 |
| global_fixed_beta | s5_opposing_local | -0.0052 | 1.53e-05 | 0.000214 |
| global_fixed_beta | s6_mixed | -0.0018 | 0.284 | 0.852 |
| global_fixed_beta | s7_opposing_recurring | -0.0028 | 1.53e-05 | 0.000214 |
| global_fixed_beta | s3b_recurring_p9 | -0.0011 | 1.53e-05 | 0.000214 |
| global_fixed_beta | s3c_recurring_p11 | +0.0001 | 0.548 | 0.852 |
| global_fixed_beta | s3d_recurring_p17 | +0.0004 | 0.329 | 0.852 |
| global_fixed_beta | s3e_recurring_p21 | +0.0010 | 0.174 | 0.697 |
| global_fixed_beta | s8_irregular_recurring | +0.0072 | 0.0348 | 0.174 |
| global_fixed_beta | s9_heterogeneous | +0.0052 | 0.00209 | 0.0125 |
| ensemble3 | s0_none | +0.0012 | 1.53e-05 | 0.000214 |
| ensemble3 | s1_abrupt | -0.0003 | 1.53e-05 | 0.000214 |
| ensemble3 | s2_gradual | +0.0000 | 0.0267 | 0.213 |
| ensemble3 | s3_recurring | -0.0001 | 0.854 | 1 |
| ensemble3 | s4_local | +0.0012 | 1.53e-05 | 0.000214 |
| ensemble3 | s5_opposing_local | +0.0005 | 0.404 | 1 |
| ensemble3 | s6_mixed | -0.0008 | 0.963 | 1 |
| ensemble3 | s7_opposing_recurring | +0.0029 | 1.53e-05 | 0.000214 |
| ensemble3 | s3b_recurring_p9 | +0.0019 | 1.53e-05 | 0.000214 |
| ensemble3 | s3c_recurring_p11 | +0.0007 | 0.12 | 0.721 |
| ensemble3 | s3d_recurring_p17 | +0.0009 | 0.284 | 1 |
| ensemble3 | s3e_recurring_p21 | +0.0025 | 0.0448 | 0.313 |
| ensemble3 | s8_irregular_recurring | -0.0122 | 0.0079 | 0.0711 |
| ensemble3 | s9_heterogeneous | -0.0031 | 0.243 | 1 |
| learn_alpha | s0_none | +0.0001 | 0.00464 | 0.0232 |
| learn_alpha | s1_abrupt | -0.0075 | 1.53e-05 | 0.000214 |
| learn_alpha | s2_gradual | +0.0001 | 7.63e-05 | 0.000534 |
| learn_alpha | s3_recurring | -0.0020 | 1.53e-05 | 0.000214 |
| learn_alpha | s4_local | -0.0251 | 1.53e-05 | 0.000214 |
| learn_alpha | s5_opposing_local | -0.0003 | 0.225 | 0.674 |
| learn_alpha | s6_mixed | +0.0059 | 0.109 | 0.436 |
| learn_alpha | s7_opposing_recurring | +0.0001 | 0.225 | 0.674 |
| learn_alpha | s3b_recurring_p9 | -0.0011 | 1.53e-05 | 0.000214 |
| learn_alpha | s3c_recurring_p11 | -0.0015 | 1.53e-05 | 0.000214 |
| learn_alpha | s3d_recurring_p17 | -0.0016 | 1.53e-05 | 0.000214 |
| learn_alpha | s3e_recurring_p21 | -0.0003 | 0.284 | 0.674 |
| learn_alpha | s8_irregular_recurring | +0.0128 | 0.00134 | 0.00806 |
| learn_alpha | s9_heterogeneous | +0.0063 | 3.05e-05 | 0.000244 |
| fixed_share | s0_none | -0.0030 | 1.53e-05 | 0.000214 |
| fixed_share | s1_abrupt | -0.0067 | 1.53e-05 | 0.000214 |
| fixed_share | s2_gradual | -0.0017 | 1.53e-05 | 0.000214 |
| fixed_share | s3_recurring | -0.0030 | 1.53e-05 | 0.000214 |
| fixed_share | s4_local | -0.0253 | 1.53e-05 | 0.000214 |
| fixed_share | s5_opposing_local | -0.0006 | 0.0714 | 0.214 |
| fixed_share | s6_mixed | +0.0063 | 0.109 | 0.218 |
| fixed_share | s7_opposing_recurring | -0.0007 | 4.58e-05 | 0.000275 |
| fixed_share | s3b_recurring_p9 | -0.0034 | 1.53e-05 | 0.000214 |
| fixed_share | s3c_recurring_p11 | -0.0018 | 1.53e-05 | 0.000214 |
| fixed_share | s3d_recurring_p17 | -0.0023 | 1.53e-05 | 0.000214 |
| fixed_share | s3e_recurring_p21 | -0.0001 | 0.487 | 0.487 |
| fixed_share | s8_irregular_recurring | +0.0092 | 0.015 | 0.075 |
| fixed_share | s9_heterogeneous | +0.0030 | 0.0267 | 0.107 |
| m5b_smooth0.001 | s0_none | +0.0011 | 1.53e-05 | 0.000214 |
| m5b_smooth0.001 | s1_abrupt | +0.0007 | 1.53e-05 | 0.000214 |
| m5b_smooth0.001 | s2_gradual | +0.0011 | 1.53e-05 | 0.000214 |
| m5b_smooth0.001 | s3_recurring | -0.0078 | 1.53e-05 | 0.000214 |
| m5b_smooth0.001 | s4_local | +0.0023 | 1.53e-05 | 0.000214 |
| m5b_smooth0.001 | s5_opposing_local | +0.0017 | 0.000504 | 0.00302 |
| m5b_smooth0.001 | s6_mixed | -0.0026 | 0.0887 | 0.0887 |
| m5b_smooth0.001 | s7_opposing_recurring | +0.0003 | 0.011 | 0.022 |
| m5b_smooth0.001 | s3b_recurring_p9 | -0.0006 | 0.00385 | 0.0115 |
| m5b_smooth0.001 | s3c_recurring_p11 | -0.0050 | 1.53e-05 | 0.000214 |
| m5b_smooth0.001 | s3d_recurring_p17 | -0.0159 | 1.53e-05 | 0.000214 |
| m5b_smooth0.001 | s3e_recurring_p21 | -0.0224 | 1.53e-05 | 0.000214 |
| m5b_smooth0.001 | s8_irregular_recurring | -0.0312 | 0.000504 | 0.00302 |
| m5b_smooth0.001 | s9_heterogeneous | -0.0084 | 0.000839 | 0.00336 |

## Measured cost (one 120-block, 3000-row/block bank)

| component | seconds | note |
|---|--:|---|
| candidate_bank (shared by all) | 74.39 | 5 SGD experts |
| han_arw (selection) | 1.578 | over the shared bank |
| fixed_share | 0.026 | EW + share, no training |
| learn_alpha | 0.059 | 7 Fixed-Share sub-algs |
| m5b gate (1x) | 0.952 | online context gate |
| ensemble3 | 0.813 | meta-gate ONLY (needs m2 + 2x m5b first: ~3x gate cost) |
| amgtp (adaptive) | 1.15 | one gate + one persistence net |

## Figures
- `figures/fixed_beta_curve.png` -- excess vs beta per regime
- `figures/beta_trace_heterogeneous.png` -- beta_t through S9
- `figures/per_group_weights_local.png` -- per-subgroup beta under S4