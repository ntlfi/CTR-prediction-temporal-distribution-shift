# M1/M2 findings

- M1 (global adaptive mixing, val_window=3) and M2 (context-dependent gating) mix the shared rolling_3 (short) / expanding (long) candidate-bank predictions.
- M1 locked-test log loss 0.4372; deployed alpha ranged 0.00-0.70 (mean 0.32); mean oracle headroom 0.0220.
- M2 locked-test log loss 0.4353; mean gate weight ranged 0.12-0.56.
- Best method overall on locked test: m2_context_gate (log loss 0.4353).
- Group breakdown (locked-test mean log loss, A=drifted subpopulation, B=stable): see group_breakdown_summary.csv.
