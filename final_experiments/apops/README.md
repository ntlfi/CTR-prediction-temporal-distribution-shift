# AP-OPS — Adaptive-Persistence Online Platt Scaling

Implementation of `final_experiments/AP_OPS_Minimal_Experiment_Plan.pdf`
(the "minimal" method-and-experiment note, September 2026).

## Idea

Keep the repo's verified Online Platt Scaling (OPS) as the calibration
primitive but make its *temporal memory* adaptive. Three experts run in
parallel on the identical frozen cross-day probability `q_{d,i}`:

| expert | state across days | update | role |
|---|---|---|---|
| **R** reset anchor | reset `(a,b)=(1,0)` each day | current projected gradient (`twoscale.calib.replay_day`) | protects the verified OPS behaviour |
| **S** short memory | carried continuously | discounted ONS, 4 h half-life | tracks fast day-boundary shifts |
| **L** long memory | carried continuously | discounted ONS, 16 h half-life | retains stable cross-day calibration |

Their probabilities are mixed `p^AP = Σ_k w_k p^(k)`; the weights start at
`(0.50, 0.25, 0.25)`, persist across days, and are updated by a **delayed
fixed-share** rule on matured block losses:

```
w~_k  ∝  w_k exp(-η_m L_{r,k}) ;   w_k = (1-α) w~_k + α/3
```

R run alone is exactly the current OPS row of the headline table
(`apops_tests.py` checks this bit-for-bit), so AP-OPS is anchored to the
strongest observed calibrator.

## Files

| file | role |
|---|---|
| `experts.py` | `replay_ons_stream` — one persistent discounted-ONS expert over the whole stream |
| `aggregate.py` | `aggregate` — delayed fixed-share meta-layer over `{R,S,L}` (absolute-time maturation queue) |
| `method.py` | `build_rows` — the plan's 4 rows: `current_ops`, `ap_ops`, `single_memory`, `no_slope` |

Runners (in `final_experiments/`): `run_apops_hpo.py` (dev tuning →
`apops_selected.json`), `run_apops_final.py` (locked fixed test + golden
check + decision rule), `run_apops_rolling.py` (rolling-origin
confirmation). Tests: `apops_tests.py`. SLURM: `apops_{hpo,final,rolling}_{criteo,avazu}.slurm`.

## What is tuned (dev days, seeds 0/1/2, one frozen config)

Only the ONS scale/regularization (`ons_lam ∈ {1e-3,1e-2,1e-1,1}`,
`ons_eta ∈ {0.25,1,4}`) and the two meta parameters (`eta_m ∈ {1,10,100}`,
switching half-life `τ ∈ {4,16,64,∞} h` → `α`). Everything upstream of
calibration — the shared adaptive cross-day mixture and the locked OPS
`(B, η0, schedule)` — is reused verbatim from the dataset's existing
`selected_configs.json`.

## Decision rule (plan section 4)

`δ_NI = 0.10 · |L_OPS,dev − L_base-q,dev|` (base-q = uncalibrated adaptive
mixture), recorded in `apops_selected.json`. AP-OPS **Improves** if the
paired daily CI vs OPS is `< 0`, **Retains performance** if its upper end
is `< δ_NI`, otherwise **Inconclusive/Failure** (→ keep OPS, report AP-OPS
as an ablation).
