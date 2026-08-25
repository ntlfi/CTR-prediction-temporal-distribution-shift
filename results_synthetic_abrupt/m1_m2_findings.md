# M1/M2 findings

- M1 (global adaptive mixing, val_window=3) and M2 (context-dependent gating) mix the shared rolling_3 (short) / expanding (long) candidate-bank predictions.
- M1 locked-test log loss 0.4077; deployed alpha ranged 0.00-1.00 (mean 0.60); mean oracle headroom 0.0071.
- M2 locked-test log loss 0.4088; mean gate weight ranged 0.10-0.99.
- Best method overall on locked test: han_arw (log loss 0.3936).
- Group breakdown (locked-test mean log loss, A=drifted subpopulation, B=stable): see group_breakdown_summary.csv.
