# M1/M2 findings

- M1 (global adaptive mixing, val_window=3) and M2 (context-dependent gating) mix the shared rolling_3 (short) / expanding (long) candidate-bank predictions.
- M1 locked-test log loss 0.3998; deployed alpha ranged 0.60-0.80 (mean 0.67); mean oracle headroom 0.0004.
- M2 locked-test log loss 0.4000; mean gate weight ranged 0.53-0.84.
- Best method overall on locked test: rolling_14 (log loss 0.3543).
- Group breakdown (locked-test mean log loss, A=drifted subpopulation, B=stable): see group_breakdown_summary.csv.
