# Two-timescale CTR forecasting -- criteo
3 cell(s), seeds [0, 1, 2]. Split: train [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dev [16, 17, 18, 19, 20, 21], test [22, 23, 24, 25, 26, 27, 28, 29, 30].
Frozen calibrator config: `{"B": 0.25, "eta0": 0.3, "eta_schedule": "const", "update": "block", "block_sec": 900, "delay_sec": 1800, "eps": 1e-05, "platt": false, "a_bounds": [0.2, 5.0], "init_b": 0.0, "carryover_rho": 0.0}`

## Feasibility (dev days, plan section 5)
- mean daily oracle-intercept improvement: 0.00023 log loss (0.037% relative)
- mean longest same-sign intraday residual run: 36.3 blocks; lag-1 residual autocorr 0.516

## Locked-test log loss by method

```
                  imp_wt_ll_mean  imp_wt_ll_std  daily_ll_mean  worst_day_mean  brier_mean  ece_mean  n
method                                                                                                 
expanding                0.60807        0.00001        0.60834         0.61552     0.21007   0.01296  3
rolling_3                0.60741        0.00011        0.60765         0.61413     0.20979   0.01041  3
rolling_7                0.60729        0.00005        0.60755         0.61414     0.20973   0.01009  3
equal_ensemble           0.60716        0.00004        0.60741         0.61415     0.20968   0.01026  3
long_only                0.60718        0.00006        0.60743         0.61408     0.20968   0.00990  3
short_only               0.60788        0.00001        0.60813         0.61464     0.20996   0.00929  3
combined                 0.60711        0.00006        0.60737         0.61388     0.20965   0.00844  3
time_of_day              0.60710        0.00007        0.60734         0.61389     0.20964   0.01008  3
online_platt             0.60700        0.00007        0.60726         0.61383     0.20961   0.00688  3
oracle_intercept         0.60707        0.00005        0.60732         0.61386     0.20963   0.00853  3
```

## Decisive paired comparisons (Delta = method - baseline, <0 favours method)

```
              comparison  n_seeds  mean_delta  delta_sd  seeds_delta_neg  mean_days_won_frac  seeds_ci_below_zero  sign_test_p
   combined_vs_long_only        3    -0.00006   0.00003                3             0.70370                    0        0.125
  combined_vs_short_only        3    -0.00076   0.00005                3             1.00000                    3        0.125
   combined_vs_expanding        3    -0.00097   0.00005                3             1.00000                    3        0.125
 combined_vs_time_of_day        3     0.00003   0.00004                1             0.22222                    0        0.875
 short_only_vs_expanding        3    -0.00020   0.00001                3             0.81481                    3        0.125
  long_only_vs_expanding        3    -0.00090   0.00005                3             1.00000                    3        0.125
online_platt_vs_combined        3    -0.00011   0.00001                3             0.88889                    3        0.125
```

## Ablations (plan section 9)

```
          ablation    setting    mean     std  size
  calib_complexity  intercept 0.60711 0.00006     3
  calib_complexity      platt 0.60700 0.00007     3
     carryover_rho        0.0 0.60711 0.00006     3
     carryover_rho        0.5 0.60712 0.00005     3
     carryover_rho        1.0 0.60712 0.00005     3
        chronology       real 0.60711 0.00006     3
        chronology   shuffled 0.60710 0.00004     3
         delay_sec          0 0.60705 0.00005     3
         delay_sec       1800 0.60711 0.00006     3
         delay_sec       3600 0.60715 0.00006     3
         delay_sec        900 0.60709 0.00005     3
long_term_backbone   adaptive 0.60711 0.00006     3
long_term_backbone      equal 0.60709 0.00000     3
long_term_backbone  expanding 0.60788 0.00001     3
update_granularity block_1800 0.60722 0.00006     3
update_granularity block_3600 0.60713 0.00006     3
update_granularity  block_900 0.60711 0.00006     3
update_granularity impression 0.60703 0.00005     3
```

## Verdict (plan section 10)
- criterion 1 (combined beats long-only, paired CI<0): 0% of seeds
- criterion 2 (combined beats short-only): 100% of seeds
- interpretation: **long_only ~= combined: limited exploitable within-day calibration drift**
