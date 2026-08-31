# Downstream autobidding eval -- AMG-TP plan step 8

Frozen CTR models fed into the same auction + per-block pacing (`autobid.py`). Primary metric: **value at matched spend** (clicks; and conversions, which on the synthetic source equal clicks). `_oracle` / `_noskill` / `_shuffled_amgtp` are non-deployable frontier anchors.

_Criteo uses the log's recorded display `cost`. Synthetic has no cost column, so `autobid.synthetic_cost` builds a second-price landscape from the **true** click probability (not from any model under test) -- clicked-likely impressions are genuinely more expensive, the regime where prediction skill pays off. The synthetic numbers therefore test whether a prediction gain *translates* to bidding value, not the absolute value level._

## Clicks won at 25% of historical spend (mean over seeds)

| regime | expanding | rolling_7 | han_arw | m2_context_gate | m5b_smooth0.1 | ensemble3 | amgtp | _oracle | _noskill | _shuffled_amgtp |
|---|---|---|---|---|---|---|---|---|---|---|
| criteo | 1028514.0 | 1028514.8 | 1028527.8 | 1028606.1 | 1028633.1 | 1028659.0 | 1028671.8 | 1288934.9 | 1023685.7 | 1002632.5 |
| s0_none | 17777.7 | 17228.3 | 17777.7 | 17758.9 | 17722.1 | 17752.8 | 17722.5 | 25978.7 | 15129.3 | 13194.0 |
| s1_abrupt | 16317.8 | 18479.2 | 18694.7 | 17857.3 | 18346.4 | 18716.3 | 18726.1 | 27196.6 | 16288.6 | 14157.6 |
| s2_gradual | 15303.9 | 16913.8 | 17174.2 | 16396.1 | 17121.6 | 17130.9 | 17131.3 | 26520.9 | 14679.9 | 13000.8 |
| s3_recurring | 15321.5 | 14605.8 | 15190.8 | 15361.1 | 15432.7 | 15414.5 | 15423.7 | 25993.5 | 14114.2 | 12895.3 |
| s4_local | 16288.8 | 16152.4 | 16334.1 | 16501.8 | 16615.2 | 16694.7 | 16676.0 | 26842.9 | 15893.1 | 14519.0 |
| s5_opposing_local | 16005.0 | 16862.2 | 16999.7 | 16690.8 | 16971.4 | 17017.3 | 17020.1 | 27683.5 | 16565.2 | 15094.5 |
| s6_mixed | 16515.7 | 17804.4 | 17908.9 | 17445.1 | 17658.6 | 17767.0 | 17726.4 | 27138.7 | 15995.8 | 13972.6 |

## AMG-TP minus baseline, value at matched spend (paired over seeds)

Negative `rel_pct` = AMG-TP wins fewer; positive = AMG-TP wins more. `wilcoxon_p` over the per-seed paired differences.

| regime | ref | n_seed | amgtp_mean | ref_mean | rel_pct | n_amgtp_wins | wilcoxon_p |
|---|---|---|---|---|---|---|---|
| s0_none | han_arw | 8 | 17722.500 | 17777.700 | -0.310 | 0 | 0.008 |
| s0_none | expanding | 8 | 17722.500 | 17777.700 | -0.310 | 0 | 0.008 |
| s0_none | ensemble3 | 8 | 17722.500 | 17752.800 | -0.170 | 0 | 0.008 |
| s0_none | m5b_smooth0.1 | 8 | 17722.500 | 17722.100 | 0.000 | 5 | 1.000 |
| s1_abrupt | han_arw | 8 | 18726.100 | 18694.700 | 0.170 | 7 | 0.016 |
| s1_abrupt | expanding | 8 | 18726.100 | 16317.800 | 14.760 | 8 | 0.008 |
| s1_abrupt | ensemble3 | 8 | 18726.100 | 18716.300 | 0.050 | 6 | 0.148 |
| s1_abrupt | m5b_smooth0.1 | 8 | 18726.100 | 18346.400 | 2.070 | 8 | 0.008 |
| s2_gradual | han_arw | 8 | 17131.300 | 17174.200 | -0.250 | 0 | 0.008 |
| s2_gradual | expanding | 8 | 17131.300 | 15303.900 | 11.940 | 8 | 0.008 |
| s2_gradual | ensemble3 | 8 | 17131.300 | 17130.900 | 0.000 | 4 | 0.945 |
| s2_gradual | m5b_smooth0.1 | 8 | 17131.300 | 17121.600 | 0.060 | 6 | 0.039 |
| s3_recurring | han_arw | 8 | 15423.700 | 15190.800 | 1.530 | 8 | 0.008 |
| s3_recurring | expanding | 8 | 15423.700 | 15321.500 | 0.670 | 8 | 0.008 |
| s3_recurring | ensemble3 | 8 | 15423.700 | 15414.500 | 0.060 | 4 | 0.844 |
| s3_recurring | m5b_smooth0.1 | 8 | 15423.700 | 15432.700 | -0.060 | 3 | 0.641 |
| s4_local | han_arw | 8 | 16676.000 | 16334.100 | 2.090 | 8 | 0.008 |
| s4_local | expanding | 8 | 16676.000 | 16288.800 | 2.380 | 8 | 0.008 |
| s4_local | ensemble3 | 8 | 16676.000 | 16694.700 | -0.110 | 1 | 0.023 |
| s4_local | m5b_smooth0.1 | 8 | 16676.000 | 16615.200 | 0.370 | 8 | 0.008 |
| s5_opposing_local | han_arw | 8 | 17020.100 | 16999.700 | 0.120 | 5 | 0.383 |
| s5_opposing_local | expanding | 8 | 17020.100 | 16005.000 | 6.340 | 8 | 0.008 |
| s5_opposing_local | ensemble3 | 8 | 17020.100 | 17017.300 | 0.020 | 6 | 0.547 |
| s5_opposing_local | m5b_smooth0.1 | 8 | 17020.100 | 16971.400 | 0.290 | 8 | 0.008 |
| s6_mixed | han_arw | 8 | 17726.400 | 17908.900 | -1.020 | 0 | 0.008 |
| s6_mixed | expanding | 8 | 17726.400 | 16515.700 | 7.330 | 8 | 0.008 |
| s6_mixed | ensemble3 | 8 | 17726.400 | 17767.000 | -0.230 | 3 | 0.461 |
| s6_mixed | m5b_smooth0.1 | 8 | 17726.400 | 17658.600 | 0.380 | 6 | 0.250 |
| criteo | han_arw | 5 | 1028671.800 | 1028527.800 | 0.010 | 4 | 0.125 |
| criteo | expanding | 5 | 1028671.800 | 1028514.000 | 0.020 | 5 | 0.062 |
| criteo | ensemble3 | 5 | 1028671.800 | 1028659.000 | 0.000 | 4 | 0.312 |
| criteo | m5b_smooth0.1 | 5 | 1028671.800 | 1028633.100 | 0.000 | 3 | 0.625 |

## Read

- **s0_none**: tie (-0.31%, 0/8 seeds, p=0.008)
- **s1_abrupt**: tie (+0.17%, 7/8 seeds, p=0.016)
- **s2_gradual**: tie (-0.25%, 0/8 seeds, p=0.008)
- **s3_recurring**: AMG-TP > Han ARW (+1.53%, 8/8 seeds, p=0.008)
- **s4_local**: AMG-TP > Han ARW (+2.09%, 8/8 seeds, p=0.008)
- **s5_opposing_local**: tie (+0.12%, 5/8 seeds, p=0.383)
- **s6_mixed**: AMG-TP < Han ARW (-1.02%, 0/8 seeds, p=0.008)
- **criteo**: tie (+0.01%, 4/5 seeds, p=0.125)
