# PersistenceNet hidden width -- dev-seed sweep (Extension A)

`amgtp_hidden_sweep.py`, dev seeds [0, 1, 2, 3, 4], regimes ['s0_none', 's1_abrupt', 's3_recurring', 's4_local', 's5_opposing_local', 's7_opposing_recurring']. Dev-seed mean locked-test log loss; `hidden0` = the frozen Stage 2 linear persistence net.

| config | s0_none | s1_abrupt | s3_recurring | s4_local | s5_opposing_local | s7_opposing_recurring | sum |
|---|---|---|---|---|---|---|---|
| m5b_default | 0.3191 | 0.3388 | 0.4321 | 0.4561 | 0.4969 | 0.4308 | 2.4738 |
| m5b_high_smooth | 0.3191 | 0.3506 | 0.4212 | 0.4637 | 0.5029 | 0.4276 | 2.4851 |
| hidden0 | 0.3202 | 0.3396 | 0.4250 | 0.4583 | 0.4991 | 0.4311 | 2.4734 |
| hidden4 | 0.3204 | 0.3398 | 0.4255 | 0.4588 | 0.4989 | 0.4307 | 2.4742 |
| hidden8 | 0.3204 | 0.3400 | 0.4258 | 0.4584 | 0.4987 | 0.4308 | 2.4741 |
| hidden16 | 0.3204 | 0.3400 | 0.4271 | 0.4582 | 0.4984 | 0.4307 | 2.4748 |

Delta vs `hidden0` (negative = the hidden layer helps):

| config | s0_none | s1_abrupt | s3_recurring | s4_local | s5_opposing_local | s7_opposing_recurring |
|---|---|---|---|---|---|---|
| hidden4 | +0.0002 | +0.0003 | +0.0004 | +0.0005 | -0.0002 | -0.0004 |
| hidden8 | +0.0002 | +0.0004 | +0.0008 | +0.0001 | -0.0005 | -0.0003 |
| hidden16 | +0.0002 | +0.0004 | +0.0021 | -0.0001 | -0.0007 | -0.0004 |

**Frozen choice: `persist_hidden=0`** (linear net retained -- no hidden layer helps on dev).

`amgtp_hidden8` / `amgtp_hidden16` still run in the Stage 4 battery (ablation A10) so the negative result is confirmed on the disjoint seeds; the deployed `amgtp` keeps `persist_hidden=0`.