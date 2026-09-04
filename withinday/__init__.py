"""Within-day capacity-ladder adapters for CTR prediction.

Self-contained implementation of the *next-step* plan
``CTR_Within_Day_Capacity_Ladder_Experiment_Plan.pdf`` (added to the repo
root 2026-09-04). It builds directly on top of :mod:`twoscale` -- the same
frozen long-term predictor bank, day splits and adaptive-mixture ``q`` --
but adds a new causal per-block history representation and a ladder of
adapters (V1 Transformer down to V5 linear) that learn a *heterogeneous*
within-day residual correction the twoscale scalar calibrator cannot see.

Reuses ``twoscale.data`` / ``twoscale.splits`` / ``twoscale.longterm`` /
``twoscale.metrics`` as-is; everything specific to this plan (the context
sketch, causal block tokens and summaries, the capacity-ladder adapters,
training/ablation/selection) lives here.
"""
