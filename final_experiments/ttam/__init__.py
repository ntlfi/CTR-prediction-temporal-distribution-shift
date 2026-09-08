"""TTAM: the integrated AMG-TP + AP-OPS pipeline evaluated under one
corrected chronological protocol (``final_experiments/TTAM_Additional_Experiments_Plan.pdf``,
8 September 2026 -- "Execution plan for the revised Section 6").

The plan asks for three things this package supplies:

  1. A *corrected* fully-nested rolling-origin protocol in which **every
     data-tuned setting is chosen from data strictly before the evaluated
     origin** -- no configuration is inherited from a frozen file whose
     development window overlaps early outer days (plan section 2).

  2. A single ``TTAM`` method = AMG-TP historical module (sample-specific
     gating + learned persistence + deployed-weight memory) feeding AP-OPS
     calibration (reset anchor + persistent short/long Platt experts,
     delayed fixed-share aggregation), plus the eight comparison /
     ablation variants (plan sections 4-5).

  3. Paired day-level uncertainty, a two-panel figure, and a bidding
     replay that changes only the prediction input (plan sections 7-10).

Modules:
  bank.py      -- shared five-horizon expert bank {1, 3, 7, 14, expanding}
  fixed_mix.py -- constant validation-fitted simplex mixture (no-AMG-TP control)
  amgtp.py     -- causal AMG-TP on the five-horizon bank, maturation-aware,
                  deployed-weight memory update (plan section 4)
  method.py    -- the nine method variants on identical expert predictions
"""
