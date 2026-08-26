# M1/M2/M5b/ensemble findings

- M1 (global adaptive mixing, val_window=3) and M2 (context-dependent gating) mix the shared rolling_3 (short) / expanding (long) candidate-bank predictions; M5b (context-dependent gating) mixes all 5 WINDOW_FAMILY candidates; the ensemble is a meta-gate blending M2's and M5b's final predictions per example.
- M1 locked-test log loss 0.4199; deployed alpha ranged 0.00-0.70 (mean 0.32); mean oracle headroom 0.0215.
- M2 locked-test log loss 0.4180; mean gate weight ranged 0.13-0.51.
- M5b locked-test log loss 0.4258; final-day mean weights [('expanding', 0.08), ('rolling_1', 0.01), ('rolling_3', 0.86), ('rolling_7', 0.01), ('rolling_14', 0.04)], top expert rolling_3.
- M2+M5b ensemble locked-test log loss 0.4187; mean meta-gate weight (0=trust M2, 1=trust M5b) ranged 0.06-0.59 (mean 0.24).
- Best method overall on locked test: m2_context_gate (log loss 0.4180).
- Group breakdown (locked-test mean log loss, A=drifted subpopulation, B=stable): see group_breakdown_summary.csv.
