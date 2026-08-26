# M1/M2/M5b/ensemble findings

- M1 (global adaptive mixing, val_window=3) and M2 (context-dependent gating) mix the shared rolling_3 (short) / expanding (long) candidate-bank predictions; M5b (context-dependent gating) mixes all 5 WINDOW_FAMILY candidates; the ensemble is a meta-gate blending M2's and M5b's final predictions per example.
- M1 locked-test log loss 0.3998; deployed alpha ranged 0.60-0.80 (mean 0.67); mean oracle headroom 0.0004.
- M2 locked-test log loss 0.4002; mean gate weight ranged 0.50-0.87.
- M5b locked-test log loss 0.3554; final-day mean weights [('expanding', 0.01), ('rolling_1', 0.01), ('rolling_3', 0.05), ('rolling_7', 0.09), ('rolling_14', 0.84)], top expert rolling_14.
- M2+M5b ensemble locked-test log loss 0.3567; mean meta-gate weight (0=trust M2, 1=trust M5b) ranged 0.89-0.95 (mean 0.92).
- Best method overall on locked test: rolling_14 (log loss 0.3543).
- Group breakdown (locked-test mean log loss, A=drifted subpopulation, B=stable): see group_breakdown_summary.csv.
