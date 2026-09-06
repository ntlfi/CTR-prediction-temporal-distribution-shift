"""AP-OPS: Adaptive-Persistence Online Platt Scaling.

Implements ``final_experiments/AP_OPS_Minimal_Experiment_Plan.pdf`` (the
"minimal" note, September 2026).  Keeps the repo's verified Online Platt
Scaling (``twoscale.calib.replay_day``, daily reset, projected gradient)
as one *reset anchor* expert R and runs two *persistent* full-Platt
experts S / L (discounted Online Newton Step, 4 h / 16 h half-life) in
parallel, combining their probabilities with a delayed fixed-share meta
rule.  Nothing upstream of the calibration layer changes: every expert
receives the identical frozen cross-day ``q_{d,i}`` the headline OPS row
already uses.

Modules:
  experts.py    -- discounted-ONS persistent expert stream replay
  aggregate.py  -- delayed fixed-share meta aggregation over {R, S, L}
  method.py     -- assembles the plan's four evaluation rows
"""
