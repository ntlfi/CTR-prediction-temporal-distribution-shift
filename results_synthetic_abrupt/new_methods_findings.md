# M1/M2/M5b/ensemble findings

- M1 (global adaptive mixing, val_window=3) and M2 (context-dependent gating) mix the shared rolling_3 (short) / expanding (long) candidate-bank predictions; M5b (context-dependent gating) mixes all 5 WINDOW_FAMILY candidates; the ensemble is a meta-gate blending M2's and M5b's final predictions per example.
- M1 locked-test log loss 0.4077; deployed alpha ranged 0.00-1.00 (mean 0.60); mean oracle headroom 0.0071.
- M2 locked-test log loss 0.4059; mean gate weight ranged 0.06-1.00.
- M5b locked-test log loss 0.3731; final-day mean weights [('expanding', 0.01), ('rolling_1', 0.01), ('rolling_3', 0.04), ('rolling_7', 0.04), ('rolling_14', 0.9)], top expert rolling_14.
- M2+M5b ensemble locked-test log loss 0.3749; mean meta-gate weight (0=trust M2, 1=trust M5b) ranged 0.46-1.00 (mean 0.79).
- Best method overall on locked test: m5_multiscale_gate (log loss 0.3731).
- Group breakdown (locked-test mean log loss, A=drifted subpopulation, B=stable): see group_breakdown_summary.csv.
