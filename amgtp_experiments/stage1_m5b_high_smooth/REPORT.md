# AMG-TP Stage 1 -- M5b-high-smooth vs the baseline suite

Method under test: **`m5b_smooth0.1`** (M5b multiscale gate, `smooth_reg=0.1`).

Synthetic horizon 120 days, 5 dev seeds [0, 1, 2, 3, 4] (hyperparameters frozen on these in earlier project work), 12 disjoint confirmation seeds [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]. All numbers below are confirmation-seed mean +/- SE unless noted.

## Headline: locked-test log loss by regime

| regime | expanding | han_arw | adamoe | diff_forgetting | m2_context_gate | m5b_smooth0.001 | m5b_smooth0.1 | ensemble3 | rolling_14 |
|---|---|---|---|---|---|---|---|---|---|
| s0_none | **0.3270** ±0.0038 | **0.3270** ±0.0038 | 0.3543 ±0.0040 | 0.3385 ±0.0038 | 0.3281 ±0.0038 | 0.3282 ±0.0038 | 0.3285 ±0.0037 | 0.3281 ±0.0038 | 0.3384 ±0.0039 |
| s1_abrupt | 0.5376 ±0.0037 | 0.3389 ±0.0015 | 0.3785 ±0.0018 | 0.4141 ±0.0020 | 0.3983 ±0.0018 | 0.3382 ±0.0016 | 0.3467 ±0.0018 | 0.3392 ±0.0016 | **0.3369** ±0.0016 |
| s2_gradual | 0.4726 ±0.0036 | **0.3581** ±0.0020 | 0.3861 ±0.0023 | 0.4144 ±0.0026 | 0.4045 ±0.0024 | 0.3592 ±0.0021 | 0.3618 ±0.0020 | 0.3603 ±0.0021 | **0.3581** ±0.0020 |
| s3_recurring | 0.4357 ±0.0039 | 0.4368 ±0.0041 | 0.4384 ±0.0037 | 0.4398 ±0.0039 | 0.4333 ±0.0038 | 0.4378 ±0.0037 | **0.4275** ±0.0037 | 0.4306 ±0.0038 | 0.4509 ±0.0040 |
| s4_local | 0.5190 ±0.0045 | 0.4885 ±0.0043 | 0.5017 ±0.0043 | 0.4913 ±0.0041 | 0.4879 ±0.0040 | **0.4589** ±0.0038 | 0.4671 ±0.0040 | 0.4600 ±0.0038 | 0.4885 ±0.0043 |
| s5_opposing_local | 0.6110 ±0.0037 | 0.4947 ±0.0030 | 0.5138 ±0.0029 | 0.5165 ±0.0029 | 0.5354 ±0.0029 | 0.4921 ±0.0029 | 0.4966 ±0.0028 | 0.4933 ±0.0029 | 0.4947 ±0.0030 |
| s6_mixed | 0.5460 ±0.0137 | **0.3919** ±0.0082 | 0.4091 ±0.0055 | 0.4751 ±0.0085 | 0.4300 ±0.0090 | 0.3998 ±0.0104 | 0.4019 ±0.0088 | 0.4000 ±0.0104 | 0.4298 ±0.0149 |

## Paired comparison: `m5b_smooth0.1` minus baseline (mean test log loss, confirmation seeds)
Negative = method-under-test better. CI is a 5000-sample paired bootstrap; p is a Wilcoxon signed-rank test across seeds.

### S0 stationary
| baseline | mean Δ log loss | 95% CI | rel % | better in | Wilcoxon p |
|---|---:|---|---:|---:|---:|
| adamoe | -0.0258 | [-0.0265, -0.0250] | -7.29% | 12/12 | 0.000488 |
| diff_forgetting | -0.0100 | [-0.0105, -0.0093] | -2.95% | 12/12 | 0.000488 |
| rolling_14 | -0.0099 | [-0.0105, -0.0092] | -2.92% | 12/12 | 0.000488 |
| m5b_smooth0.001 | +0.0003 | [-0.0002, +0.0010] | +0.09% | 6/12 | 0.733 |
| ensemble3 | +0.0004 | [-0.0000, +0.0010] | +0.11% | 4/12 | 0.233 |
| m2_context_gate | +0.0004 | [-0.0001, +0.0011] | +0.13% | 5/12 | 0.339 |
| han_arw | +0.0014 | [+0.0010, +0.0021] | +0.44% | 0/12 | 0.000488 |
| expanding | +0.0014 | [+0.0010, +0.0021] | +0.44% | 0/12 | 0.000488 |

### S1 abrupt global
| baseline | mean Δ log loss | 95% CI | rel % | better in | Wilcoxon p |
|---|---:|---|---:|---:|---:|
| expanding | -0.1909 | [-0.1960, -0.1854] | -35.51% | 12/12 | 0.000488 |
| diff_forgetting | -0.0674 | [-0.0697, -0.0649] | -16.27% | 12/12 | 0.000488 |
| m2_context_gate | -0.0516 | [-0.0540, -0.0492] | -12.96% | 12/12 | 0.000488 |
| adamoe | -0.0318 | [-0.0340, -0.0295] | -8.40% | 12/12 | 0.000488 |
| ensemble3 | +0.0075 | [+0.0054, +0.0098] | +2.21% | 0/12 | 0.000488 |
| han_arw | +0.0078 | [+0.0057, +0.0100] | +2.31% | 0/12 | 0.000488 |
| m5b_smooth0.001 | +0.0085 | [+0.0063, +0.0108] | +2.51% | 0/12 | 0.000488 |
| rolling_14 | +0.0098 | [+0.0077, +0.0121] | +2.92% | 0/12 | 0.000488 |

### S2 gradual
| baseline | mean Δ log loss | 95% CI | rel % | better in | Wilcoxon p |
|---|---:|---|---:|---:|---:|
| expanding | -0.1108 | [-0.1149, -0.1070] | -23.45% | 12/12 | 0.000488 |
| diff_forgetting | -0.0526 | [-0.0546, -0.0507] | -12.70% | 12/12 | 0.000488 |
| m2_context_gate | -0.0427 | [-0.0440, -0.0414] | -10.56% | 12/12 | 0.000488 |
| adamoe | -0.0244 | [-0.0255, -0.0231] | -6.31% | 12/12 | 0.000488 |
| ensemble3 | +0.0015 | [+0.0007, +0.0024] | +0.42% | 2/12 | 0.00342 |
| m5b_smooth0.001 | +0.0026 | [+0.0017, +0.0035] | +0.72% | 0/12 | 0.000488 |
| han_arw | +0.0037 | [+0.0029, +0.0046] | +1.03% | 0/12 | 0.000488 |
| rolling_14 | +0.0037 | [+0.0029, +0.0046] | +1.03% | 0/12 | 0.000488 |

### S3 recurring
| baseline | mean Δ log loss | 95% CI | rel % | better in | Wilcoxon p |
|---|---:|---|---:|---:|---:|
| rolling_14 | -0.0234 | [-0.0243, -0.0225] | -5.18% | 12/12 | 0.000488 |
| diff_forgetting | -0.0123 | [-0.0131, -0.0115] | -2.80% | 12/12 | 0.000488 |
| adamoe | -0.0109 | [-0.0115, -0.0103] | -2.49% | 12/12 | 0.000488 |
| m5b_smooth0.001 | -0.0103 | [-0.0119, -0.0089] | -2.35% | 12/12 | 0.000488 |
| han_arw | -0.0092 | [-0.0108, -0.0078] | -2.12% | 12/12 | 0.000488 |
| expanding | -0.0082 | [-0.0091, -0.0073] | -1.88% | 12/12 | 0.000488 |
| m2_context_gate | -0.0058 | [-0.0064, -0.0052] | -1.34% | 12/12 | 0.000488 |
| ensemble3 | -0.0031 | [-0.0037, -0.0024] | -0.71% | 12/12 | 0.000488 |

### S4 local/subpopulation
| baseline | mean Δ log loss | 95% CI | rel % | better in | Wilcoxon p |
|---|---:|---|---:|---:|---:|
| expanding | -0.0519 | [-0.0538, -0.0503] | -10.01% | 12/12 | 0.000488 |
| adamoe | -0.0346 | [-0.0355, -0.0338] | -6.90% | 12/12 | 0.000488 |
| diff_forgetting | -0.0242 | [-0.0249, -0.0236] | -4.93% | 12/12 | 0.000488 |
| rolling_14 | -0.0214 | [-0.0223, -0.0205] | -4.38% | 12/12 | 0.000488 |
| han_arw | -0.0214 | [-0.0223, -0.0205] | -4.38% | 12/12 | 0.000488 |
| m2_context_gate | -0.0208 | [-0.0216, -0.0200] | -4.26% | 12/12 | 0.000488 |
| ensemble3 | +0.0071 | [+0.0066, +0.0076] | +1.55% | 0/12 | 0.000488 |
| m5b_smooth0.001 | +0.0082 | [+0.0076, +0.0087] | +1.78% | 0/12 | 0.000488 |

### S5 opposing local
| baseline | mean Δ log loss | 95% CI | rel % | better in | Wilcoxon p |
|---|---:|---|---:|---:|---:|
| expanding | -0.1144 | [-0.1177, -0.1115] | -18.73% | 12/12 | 0.000488 |
| m2_context_gate | -0.0388 | [-0.0396, -0.0379] | -7.24% | 12/12 | 0.000488 |
| diff_forgetting | -0.0199 | [-0.0208, -0.0191] | -3.85% | 12/12 | 0.000488 |
| adamoe | -0.0172 | [-0.0179, -0.0164] | -3.34% | 12/12 | 0.000488 |
| rolling_14 | +0.0019 | [+0.0011, +0.0027] | +0.39% | 1/12 | 0.00146 |
| han_arw | +0.0019 | [+0.0011, +0.0027] | +0.39% | 1/12 | 0.00146 |
| ensemble3 | +0.0033 | [+0.0023, +0.0044] | +0.68% | 1/12 | 0.000977 |
| m5b_smooth0.001 | +0.0045 | [+0.0034, +0.0055] | +0.92% | 0/12 | 0.000488 |

### S6 mixed unknown
| baseline | mean Δ log loss | 95% CI | rel % | better in | Wilcoxon p |
|---|---:|---|---:|---:|---:|
| expanding | -0.1441 | [-0.1776, -0.1136] | -26.39% | 12/12 | 0.000488 |
| diff_forgetting | -0.0732 | [-0.0826, -0.0640] | -15.41% | 12/12 | 0.000488 |
| m2_context_gate | -0.0281 | [-0.0383, -0.0193] | -6.53% | 12/12 | 0.000488 |
| rolling_14 | -0.0279 | [-0.0442, -0.0126] | -6.49% | 10/12 | 0.00244 |
| adamoe | -0.0072 | [-0.0161, +0.0027] | -1.76% | 7/12 | 0.204 |
| ensemble3 | +0.0019 | [-0.0025, +0.0059] | +0.47% | 4/12 | 0.301 |
| m5b_smooth0.001 | +0.0021 | [-0.0057, +0.0097] | +0.52% | 3/12 | 0.424 |
| han_arw | +0.0100 | [+0.0033, +0.0181] | +2.56% | 3/12 | 0.0269 |

## Adaptation & oracle diagnostics (method under test, confirmation seeds)
`recovery` / `peak post-shift excess` are only defined for regimes with an explicit change point (S1, S4, S5). `oracle persistence = 'high' frac` is the fraction of test days on which fixed `smooth_reg=0.1` beat `1e-3` in hindsight -- how often the optimal persistence regime flips, i.e. the headroom an adaptive beta_t targets.

| regime | mean excess vs per-day oracle | recovery (days) | peak post-shift excess | stationary downside | oracle persistence='high' frac |
|---|---:|---:|---:|---:|---:|
| s0_none | +0.0014 | -- | -- | 0.0014 | 0.53 |
| s1_abrupt | +0.0098 | 25.4 | +0.0213 | -- | 0.02 |
| s2_gradual | +0.0037 | -- | -- | -- | 0.03 |
| s3_recurring | +0.0050 | -- | -- | -- | 0.64 |
| s4_local | -0.0214 | 25.0 | -0.0132 | -- | 0.00 |
| s5_opposing_local | +0.0039 | 5.0 | +0.0120 | -- | 0.05 |
| s6_mixed | +0.0303 | -- | -- | -- | 0.33 |

## Central question
> Is there reproducible evidence that adaptive combination of short- and long-term information outperforms strong recency-based temporal adaptation (Han ARW), and under what shift?

**Beats Han ARW (reproducibly):** s3_recurring (-0.0092, 12/12 seeds, p=0.000488); s4_local (-0.0214, 12/12 seeds, p=0.000488).
**Loses to Han ARW:** s0_none (+0.0014, 0/12 seeds, p=0.000488); s1_abrupt (+0.0078, 0/12 seeds, p=0.000488); s2_gradual (+0.0037, 0/12 seeds, p=0.000488); s5_opposing_local (+0.0019, 1/12 seeds, p=0.00146); s6_mixed (+0.0100, 3/12 seeds, p=0.0269).
**Statistical tie with Han ARW:** none.
**Stationary downside vs expanding ERM (S0):** +0.0014 log loss (+0.4% approx) -- no meaningful downside.

Against the PDF's decision table (section 9): this is a **partial success** for `m5b_smooth0.1` as a fixed configuration -- it replaces the hand-tuned high-persistence specialist on the regimes where persistence helps, but does not dominate the abrupt/gradual/mixed regimes where Han ARW's fast global window still wins. The `oracle persistence='high' frac` column shows why: the optimal persistence regime is not fixed, which is the motivation for the adaptive-beta_t method in Stage 2.