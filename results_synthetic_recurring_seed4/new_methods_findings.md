# M1/M2/M5b/ensemble findings

- M1 (global adaptive mixing, val_window=3) and M2 (context-dependent gating) mix the shared rolling_3 (short) / expanding (long) candidate-bank predictions; M5b (context-dependent gating) mixes all 5 WINDOW_FAMILY candidates; the ensemble is a meta-gate blending M2's and M5b's final predictions per example.
- M1 locked-test log loss 0.4376; deployed alpha ranged 0.00-0.70 (mean 0.31); mean oracle headroom 0.0231.
- M2 locked-test log loss 0.4366; mean gate weight ranged 0.10-0.59.
- M5b locked-test log loss 0.4387; final-day mean weights [('expanding', 0.57), ('rolling_1', 0.01), ('rolling_3', 0.32), ('rolling_7', 0.02), ('rolling_14', 0.07)], top expert expanding.
- M2+M5b ensemble locked-test log loss 0.4375; mean meta-gate weight (0=trust M2, 1=trust M5b) ranged 0.06-0.59 (mean 0.30).
- Best method overall on locked test: m2_context_gate (log loss 0.4366).
- Group breakdown (locked-test mean log loss, A=drifted subpopulation, B=stable): see group_breakdown_summary.csv.
