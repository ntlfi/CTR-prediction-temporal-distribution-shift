"""Two-timescale CTR forecasting: long-term cross-day adaptation plus a
lightweight within-day online scalar calibration.

Self-contained implementation of ``CTR_Two_Timescale_Experiment_Plan.pdf``.
This package deliberately shares *no code* with the repo's earlier
adaptive-training / AMG-TP experiments -- only the raw Criteo / Avazu data
files on disk. Everything needed (data loading, the long-term predictor bank,
the adaptive mixture, the online calibrator, the causal within-day replay,
metrics and diagnostics) lives here.
"""
