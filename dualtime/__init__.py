"""DualTime-CTR: the frozen final method (adaptive cross-day historical
learning + contextual online residual learning within each day) and the
baselines it is compared against in the headline table.

Deliberately does not duplicate what already exists and already matches
this experiment's requirements:

* ``twoscale.data`` / ``twoscale.splits`` -- unchanged loaders and the
  chronological splits (both datasets already give exactly the required
  16/6/9 and 5/2/3 day counts).
* ``twoscale.longterm`` -- the shared three-expert bank (``HORIZONS =
  ("roll3", "roll7", "expanding")`` already, no five-expert bank in this
  repo to worry about) and the adaptive cross-day mixture (its
  ``eta``/``halflife`` grid already matches this experiment's Section 8
  grid exactly).
* ``twoscale.calib`` -- OPS (Gupta & Ramdas online Platt scaling) is
  already implemented (``CalibConfig(platt=True)``, block-based causal
  replay); its grid already matches this experiment's Section 10 grid.
* ``withinday.blocks`` / ``withinday.contextsketch`` / ``withinday.cache``
  -- the frozen feature architecture (context sketch, block token,
  deterministic summary) this experiment's Section 11 explicitly says to
  reuse rather than re-derive.

New in this package:

* ``dualtime.arw`` -- Adaptive Rolling Window (Han, Huang & Wang, 2024,
  *"Model Assessment and Selection under Temporal Distribution Shift"*,
  ICML 2024, arXiv:2402.08672), reconstructed from the paper's algorithm
  description (bias/variance-proxy window selection + Bernstein
  confidence bound + pairwise tournament) -- not verified against the
  authors' own source code, since none was consulted.
* ``dualtime.adamoe`` -- an EMA-weighted mixture-of-experts baseline in
  the spirit of Liu et al. (2022), *"On the Adaptation to Concept Drift
  for CTR Prediction"* (arXiv:2204.05101); implemented directly from this
  experiment's own precise specification (EMA momentum lambda, uniform
  init) rather than the paper's exact update rule, which was not
  independently verified.
* ``dualtime.online`` -- DualTime-CTR's within-day module: a projected
  online-gradient learner over the frozen phi features, reset to w=0
  every day, updated once per matured block -- structurally the
  "frozen-encoder online-regret" variant sketched (but never implemented)
  in the original within-day capacity-ladder plan, now built for real.
"""
