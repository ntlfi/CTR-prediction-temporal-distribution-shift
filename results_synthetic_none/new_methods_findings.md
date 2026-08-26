# M1/M2/M5b/ensemble findings

- M1 (global adaptive mixing, val_window=3) and M2 (context-dependent gating) mix the shared rolling_3 (short) / expanding (long) candidate-bank predictions; M5b (context-dependent gating) mixes all 5 WINDOW_FAMILY candidates; the ensemble is a meta-gate blending M2's and M5b's final predictions per example.
- M1 locked-test log loss 0.3107; deployed alpha ranged 0.00-0.00 (mean 0.00); mean oracle headroom 0.0000.
- M2 locked-test log loss 0.3117; mean gate weight ranged 0.04-0.17.
- M5b locked-test log loss 0.3119; final-day mean weights [('expanding', 0.72), ('rolling_1', 0.03), ('rolling_3', 0.05), ('rolling_7', 0.09), ('rolling_14', 0.11)], top expert expanding.
- M2+M5b ensemble locked-test log loss 0.3118; mean meta-gate weight (0=trust M2, 1=trust M5b) ranged 0.41-0.61 (mean 0.50).
- Best method overall on locked test: expanding (log loss 0.3107).
- Group breakdown (locked-test mean log loss, A=drifted subpopulation, B=stable): see group_breakdown_summary.csv.
