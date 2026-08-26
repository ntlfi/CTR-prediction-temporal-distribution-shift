# M1/M2/M5b/ensemble findings

- M1 (global adaptive mixing, val_window=3) and M2 (context-dependent gating) mix the shared rolling_3 (short) / expanding (long) candidate-bank predictions; M5b (context-dependent gating) mixes all 5 WINDOW_FAMILY candidates; the ensemble is a meta-gate blending M2's and M5b's final predictions per example.
- M1 locked-test log loss 0.4372; deployed alpha ranged 0.00-0.70 (mean 0.32); mean oracle headroom 0.0220.
- M2 locked-test log loss 0.4353; mean gate weight ranged 0.12-0.56.
- M5b locked-test log loss 0.4404; final-day mean weights [('expanding', 0.89), ('rolling_1', 0.0), ('rolling_3', 0.03), ('rolling_7', 0.03), ('rolling_14', 0.05)], top expert expanding.
- M2+M5b ensemble locked-test log loss 0.4375; mean meta-gate weight (0=trust M2, 1=trust M5b) ranged 0.05-0.83 (mean 0.37).
- Best method overall on locked test: m2_context_gate (log loss 0.4353).
- Group breakdown (locked-test mean log loss, A=drifted subpopulation, B=stable): see group_breakdown_summary.csv.
