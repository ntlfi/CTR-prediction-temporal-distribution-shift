# Two-timescale CTR forecasting -- avazu
8 cell(s), seeds [0, 1, 2, 3, 4, 5, 6, 7]. Split: train [0, 1, 2, 3, 4], dev [5, 6], test [7, 8, 9].
Frozen calibrator config: `{"B": 0.25, "eta0": 0.3, "eta_schedule": "const", "update": "block", "block_sec": 3600, "delay_sec": 1800, "eps": 1e-05, "platt": false, "a_bounds": [0.2, 5.0], "init_b": 0.0, "carryover_rho": 0.0}`

## Feasibility (dev days, plan section 5)
- mean daily oracle-intercept improvement: 0.00010 log loss (0.023% relative)
- mean longest same-sign intraday residual run: 11.6 blocks; lag-1 residual autocorr 0.797

## Locked-test log loss by method

```
                  imp_wt_ll_mean  imp_wt_ll_std  daily_ll_mean  worst_day_mean  brier_mean  ece_mean  n
method                                                                                                 
expanding                0.38798        0.00045        0.38904         0.40671     0.11992   0.01229  8
rolling_3                0.38882        0.00078        0.38979         0.40711     0.12011   0.01596  8
rolling_7                0.38802        0.00049        0.38908         0.40691     0.11994   0.01233  8
equal_ensemble           0.38779        0.00043        0.38883         0.40648     0.11985   0.01342  8
long_only                0.38861        0.00047        0.38955         0.40676     0.12005   0.01382  8
short_only               0.38777        0.00045        0.38885         0.40656     0.11987   0.01123  8
combined                 0.38834        0.00058        0.38931         0.40661     0.11999   0.01222  8
time_of_day              0.38869        0.00060        0.38961         0.40667     0.12011   0.01380  8
online_platt             0.38792        0.00074        0.38892         0.40633     0.11993   0.01016  8
oracle_intercept         0.38795        0.00042        0.38897         0.40646     0.11995   0.01129  8
```

## Decisive paired comparisons (Delta = method - baseline, <0 favours method)

```
              comparison  n_seeds  mean_delta  delta_sd  seeds_delta_neg  mean_days_won_frac  seeds_ci_below_zero  sign_test_p
   combined_vs_long_only        8    -0.00023   0.00015                8             0.91667                    6      0.00391
  combined_vs_short_only        8     0.00046   0.00029                0             0.50000                    0      1.00000
   combined_vs_expanding        8     0.00027   0.00037                2             0.54167                    0      0.96484
 combined_vs_time_of_day        8    -0.00029   0.00011                8             0.79167                    5      0.00391
 short_only_vs_expanding        8    -0.00019   0.00008                8             1.00000                    8      0.00391
  long_only_vs_expanding        8     0.00051   0.00036                0             0.50000                    0      1.00000
online_platt_vs_combined        8    -0.00040   0.00015                8             1.00000                    8      0.00391
```

## Ablations (plan section 9)

```
          ablation    setting    mean     std  size
  calib_complexity  intercept 0.38834 0.00058     8
  calib_complexity      platt 0.38792 0.00074     8
     carryover_rho        0.0 0.38834 0.00058     8
     carryover_rho        0.5 0.38828 0.00061     8
     carryover_rho        1.0 0.38825 0.00062     8
        chronology       real 0.38834 0.00058     8
        chronology   shuffled 0.38832 0.00059     8
         delay_sec          0 0.38832 0.00059     8
         delay_sec       1800 0.38834 0.00058     8
         delay_sec       3600 0.38834 0.00058     8
         delay_sec        900 0.38834 0.00058     8
long_term_backbone   adaptive 0.38834 0.00058     8
long_term_backbone      equal 0.38756 0.00049     8
long_term_backbone  expanding 0.38777 0.00045     8
update_granularity block_1800 0.38834 0.00058     8
update_granularity block_3600 0.38834 0.00058     8
update_granularity  block_900 0.38837 0.00057     8
update_granularity impression 0.38800 0.00042     8
```

## Verdict (plan section 10)
- criterion 1 (combined beats long-only, paired CI<0): 75% of seeds
- criterion 2 (combined beats short-only): 0% of seeds
- interpretation: **mixed / only-oracle: improve the online update policy**
