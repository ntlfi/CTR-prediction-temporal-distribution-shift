# Extension B -- per-example beta_t(x): variance penalty dev sweep

`amgtp_betax_sweep.py`, dev seeds [0, 1, 2, 3, 4], regimes ['s0_none', 's3_recurring', 's4_local', 's5_opposing_local', 's7_opposing_recurring']. Dev-seed mean locked-test log loss. `amgtp_global` = the global-beta_t AMG-TP that beta_t(x) must beat.

| config | s0_none | s3_recurring | s4_local | s5_opposing_local | s7_opposing_recurring | sum |
|---|---|---|---|---|---|---|
| m5b_default | 0.3191 | 0.4321 | 0.4561 | 0.4969 | 0.4308 | 2.1350 |
| m5b_high_smooth | 0.3191 | 0.4212 | 0.4637 | 0.5029 | 0.4276 | 2.1345 |
| amgtp_global | 0.3202 | 0.4250 | 0.4583 | 0.4991 | 0.4311 | 2.1338 |
| betax_vr0 | 0.3208 | 0.4255 | 0.4573 | 0.4992 | 0.4318 | 2.1346 |
| betax_vr0.0001 | 0.3208 | 0.4255 | 0.4573 | 0.4992 | 0.4318 | 2.1347 |
| betax_vr0.001 | 0.3208 | 0.4265 | 0.4573 | 0.4992 | 0.4318 | 2.1357 |
| betax_vr0.01 | 0.3206 | 0.4265 | 0.4572 | 0.4993 | 0.4318 | 2.1355 |
| betax_vr0.1 | 0.3206 | 0.4265 | 0.4572 | 0.4993 | 0.4321 | 2.1357 |

Per-example beta spread and A/B group-beta gap (S7 is where a real gap is the point):

| beta_var_reg | mean beta_std | mean |beta_A - beta_B| |
|---|---|---|
| 0 | 0.010 | 0.005 |
| 0.0001 | 0.010 | 0.005 |
| 0.001 | 0.010 | 0.005 |
| 0.01 | 0.010 | 0.005 |
| 0.1 | 0.009 | 0.004 |

**NEGATIVE RESULT.** No `beta_var_reg` makes per-example `beta_t(x)` beat the global `beta_t` AMG-TP on S7 -- its purpose-built target -- or anywhere: the best S7 config (`betax_vr0.001`) is +0.0007 vs global. The `|beta_A - beta_B|` column stays ~0.005 at every penalty: `g_xi` never learns the subgroup split that S7 is built around. Mechanistic read: the persistent state `m_{t-1}` it mixes toward is a single *global* EMA, so routing a stable-subgroup example to `m` still hands it the *blended* history, not that subgroup's own -- per-example persistence needs a per-example (or per-group) `m`, which PDF section 2.3 scopes out. Global `beta_t` is the right granularity given a global `m`. Closes plan section 3's open question.