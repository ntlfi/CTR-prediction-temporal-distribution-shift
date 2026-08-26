# M5c (periodicity-aware M5b) findings

- M5c is M5b (context-dependent gating over the full WINDOW_FAMILY, see m5_multiscale_gate.py) plus sin/cos periodicity phase features (periodicity.py), testing whether an explicit 'where in the cycle are we' signal closes the recurring-drift gap shared by every other method in this project.
- M5c (deployed, causally-detected period) locked-test log loss 0.3555; 66/116 eligible days had a detected period.
- Best method overall on locked test (including M5c): rolling_14 (log loss 0.3543).
