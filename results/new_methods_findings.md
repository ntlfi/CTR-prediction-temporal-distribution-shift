# M1/M2/M5b/ensemble findings

- M1 (global adaptive mixing, val_window=3) and M2 (context-dependent gating) mix the shared rolling_3 (short) / expanding (long) candidate-bank predictions; M5b (context-dependent gating) mixes all 5 WINDOW_FAMILY candidates; the ensemble is a meta-gate blending M2's and M5b's final predictions per example.
- M1 locked-test log loss 0.6072; deployed alpha ranged 0.60-0.80 (mean 0.69); mean oracle headroom 0.0000.
- M2 locked-test log loss 0.6072; mean gate weight ranged 0.54-0.58.
- M5b locked-test log loss 0.6070; final-day mean weights [('expanding', 0.17), ('rolling_1', 0.19), ('rolling_3', 0.19), ('rolling_7', 0.21), ('rolling_14', 0.23)], top expert rolling_14.
- M2+M5b ensemble locked-test log loss 0.6070; mean meta-gate weight (0=trust M2, 1=trust M5b) ranged 0.52-0.54 (mean 0.53).
- Best method overall on locked test: m5_multiscale_gate (log loss 0.6070).
