# AMG-TP persistence hyperparameters — frozen on dev seeds

`amgtp_stage2_sweep.py`, dev seeds [0,1,2,3,4], regimes S0/S1/S3 (the three
that discriminate persistence strength). Dev-seed mean locked-test log loss:

| config | S0 none | S1 abrupt | S3 recurring |
|---|---|---|---|
| m5b_default (smooth 1e-3) | 0.3191 | 0.3388 | 0.4321 |
| m5b_high_smooth (smooth 0.1) | 0.3191 | **0.3506** | 0.4212 |
| **AMG-TP** (init_bias −1, rho 0.3, β-entropy 0) | 0.3202 | **0.3397** | 0.4250 |

All 8 sweep configs land within 0.0012 summed log loss of each other — the
persistence hyperparameters barely matter, so no real search is needed
(plan §16). Frozen choice, hard-coded as `AMGTP_CONFIG` in `amgtp_run.py`:

```python
{"init_bias": -1.0, "rho": 0.3, "beta_entropy_reg": 0.0}
```

Read on the dev seeds: AMG-TP nearly matches `m5b_default` under abrupt drift
(0.3397 vs 0.3388 — it does *not* inherit `m5b_high_smooth`'s +3.5% abrupt
regression) while landing ~64% of the way from `m5b_default` to
`m5b_high_smooth` on recurring. Stationary downside +0.001. The confirmation
battery (S0–S6 × 12 disjoint seeds) tests whether this holds.
