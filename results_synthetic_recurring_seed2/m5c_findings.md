# M5c (periodicity-aware M5b) findings

- M5c is M5b (context-dependent gating over the full WINDOW_FAMILY, see m5_multiscale_gate.py) plus sin/cos periodicity phase features (periodicity.py), testing whether an explicit 'where in the cycle are we' signal closes the recurring-drift gap shared by every other method in this project.
- M5c (deployed, causally-detected period) locked-test log loss 0.4280; 66/116 eligible days had a detected period.
- M5c-oracle (diagnostic, true period=14d, never a deployed prediction) locked-test log loss 0.4308 -- quantifies the ceiling if period detection were perfect.
- Best method overall on locked test (including M5c): m2_context_gate (log loss 0.4245).
