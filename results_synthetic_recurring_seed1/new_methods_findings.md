# M1/M2/M5b/ensemble findings

- M1 (global adaptive mixing, val_window=3) and M2 (context-dependent gating) mix the shared rolling_3 (short) / expanding (long) candidate-bank predictions; M5b (context-dependent gating) mixes all 5 WINDOW_FAMILY candidates; the ensemble is a meta-gate blending M2's and M5b's final predictions per example.
- M1 locked-test log loss 0.4271; deployed alpha ranged 0.00-0.70 (mean 0.31); mean oracle headroom 0.0234.
- M2 locked-test log loss 0.4251; mean gate weight ranged 0.10-0.60.
- M5b locked-test log loss 0.4284; final-day mean weights [('expanding', 0.9), ('rolling_1', 0.0), ('rolling_3', 0.03), ('rolling_7', 0.02), ('rolling_14', 0.05)], top expert expanding.
- M2+M5b ensemble locked-test log loss 0.4262; mean meta-gate weight (0=trust M2, 1=trust M5b) ranged 0.08-0.44 (mean 0.26).
- Best method overall on locked test: m2_context_gate (log loss 0.4251).
- Group breakdown (locked-test mean log loss, A=drifted subpopulation, B=stable): see group_breakdown_summary.csv.
