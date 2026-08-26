# M1/M2/M5b/ensemble findings

- M1 (global adaptive mixing, val_window=3) and M2 (context-dependent gating) mix the shared rolling_3 (short) / expanding (long) candidate-bank predictions; M5b (context-dependent gating) mixes all 5 WINDOW_FAMILY candidates; the ensemble is a meta-gate blending M2's and M5b's final predictions per example.
- M1 locked-test log loss 0.4588; deployed alpha ranged 0.00-0.60 (mean 0.31); mean oracle headroom 0.0020.
- M2 locked-test log loss 0.4437; mean gate weight ranged 0.06-0.79.
- M5b locked-test log loss 0.4250; final-day mean weights [('expanding', 0.44), ('rolling_1', 0.01), ('rolling_3', 0.02), ('rolling_7', 0.05), ('rolling_14', 0.49)], top expert rolling_14.
- M2+M5b ensemble locked-test log loss 0.4261; mean meta-gate weight (0=trust M2, 1=trust M5b) ranged 0.43-0.98 (mean 0.74).
- Best method overall on locked test: m5_multiscale_gate (log loss 0.4250).
- Group breakdown (locked-test mean log loss, A=drifted subpopulation, B=stable): see group_breakdown_summary.csv.
