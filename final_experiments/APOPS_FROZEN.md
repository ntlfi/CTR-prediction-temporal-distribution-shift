# AP-OPS — frozen algorithm, grid, and decision rules

Frozen 2026-09-07 after the fully-nested rolling-origin analysis
(`APOPS_FINDINGS.md` section 7). Per the additional-experiments spec's
"Final Confirmation" step: the nested analysis is itself the confirmation
(no genuinely-unseen temporal stream exists in this repo). This file
records exactly what to run, with **no further tuning**, if a new dataset
with second/hour timestamps becomes available.

Code commit at freeze: `e3f52b6` (+ analysis refinement `<this commit>`).

## Algorithm (frozen)

Base map, per impression `i` with frozen cross-day probability `q_i`:

```
z_i        = logit(clip(q_i, 1e-5, 1 - 1e-5))
p_i^(k)    = sigmoid(a^(k) z_i + b^(k)),   a in [0.2, 5],  b in [-0.25, 0.25]
p_i^AP     = w_R p_i^(R) + w_S p_i^(S) + w_L p_i^(L)
```

Three experts, all on the identical `q_i`:

| expert | update | day boundary | half-life |
|---|---|---|---|
| R | projected gradient (`twoscale.calib.replay_day`, `platt=True`) | reset `(a,b)=(1,0)` | — |
| S | discounted ONS | carried | **4 h** |
| L | discounted ONS | carried | **16 h** |

Discounted ONS, one delayed step per block once every label in the block
has matured (absolute-time queue; a block near midnight updates on the
following day):

```
gamma = 2 ** (-block_sec / half_life)
g_r   = mean_{i in block r} (p_i - y_i) (z_i, 1)
A_r   = gamma A_{r-1} + g_r g_r^T + (1 - gamma) lam I,    A_0 = lam I
theta_{r+1} = clip_box( theta_r - eta_ons A_r^{-1} g_r )
```

Delayed prior-centered fixed share over `(R, S, L)`:

```
pi     = (1 - lambda_AP, lambda_AP / 2, lambda_AP / 2)
w_1    = pi
w~_k   proportional to  w_k exp(-eta_m (L_{r,k} - min_k L_{r,k}))
w_{r+1}= (1 - alpha) w~ + alpha pi,     alpha = 1 - 2 ** (-block_sec / tau)
```

`lambda_AP = 0` pins `w = (1,0,0)` -> AP-OPS == OPS prediction by
prediction (the anchor / floor).

## Frozen hyper-parameters

Dev-only (never re-tuned):

| parameter | Criteo | Avazu |
|---|---|---|
| block_sec | 900 | 3600 |
| delay_sec | 1800 | 1800 |
| cross-day mixture (q) | adaptive roll3/roll7/expanding, `eta=150, halflife=3` | `eta=1e6, halflife=3` |
| OPS `(B, eta0, schedule)` | `(0.25, 0.3, const)` | `(0.25, 0.3, const)` |
| ONS `(lam, eta_ons)` | `(0.1, 0.25)` | `(1.0, 4.0)` |
| meta `eta_m` | 100 | 100 |
| S / L half-lives | 4 h / 16 h | 4 h / 16 h |

Selected per rolling origin (inner window = trailing 3 eligible days, one
common config across seeds 0/1/2 by mean inner-validation log loss):

| knob | grid | note |
|---|---|---|
| shared mixture `(eta, halflife)` | `eta in {10,30,60,150,1e6}` x `halflife in {3,5,10}` | on uncalibrated inner loss |
| `lambda_AP` | `{0, 0.25, 0.5, 0.75, 1.0}` | **grid contains 0** |
| `tau` (switch half-life) | `{4, 16, inf} h` | |
| persistent / reset ONS half-life | `{4, 16} h` | shared by Reset-ONS and Persistent-ONS |

## Decision rules (predeclared, spec section 4)

Materiality floor for "two rows differ": `2e-4` on the equal-day-weighted
mean paired log-loss delta (well above the D-day bootstrap noise here).

- **Cross-day persistence supported** iff Persistent-ONS materially beats
  Reset-ONS.
- **Adaptive aggregation supported** iff AP-OPS is non-inferior to
  Persistent-ONS on both datasets and materially better on >= 1.
- **AP-OPS succeeds** iff it beats OPS on both datasets in paired
  rolling-origin log loss (day-level CI excludes 0), shows no systematic
  early-day / worst-day regression, and selects `lambda_AP > 0` on a
  non-trivial fraction of origins.

## How to run on a new dataset (no tuning)

1. Add a loader to `twoscale/data.py` returning a `Dataset` (X hashed
   features, y, day, sec_in_day; rows day-then-sec sorted).
2. Pick `block_sec` = the data's native time resolution (900 s if
   second-resolution, else the hour), `delay_sec = 1800`.
3. Freeze the cross-day mixture + OPS + ONS hyper-parameters to the values
   above for the closest existing dataset (second-resolution -> Criteo
   column; hour-resolution -> Avazu column).
4. `PYTHONPATH=. .venv/bin/python final_experiments/run_apops_nested.py
   --source <name> --config <that dataset's apops_selected.json or a hand
   -written one with the frozen values> --out <dir>` -- the per-origin
   grid above is applied automatically; nothing else is tuned.
5. `run_apops_analysis.py --nested` -> apply the decision rules verbatim.

## Result at freeze (for reference; `APOPS_FINDINGS.md` section 7)

AP-OPS beats OPS on both datasets under the fully nested protocol:
Criteo -0.000172 [-0.000220, -0.000131], 15/15 origins; Avazu -0.000136
[-0.000181, -0.000098], 5/5 origins; `lambda_AP > 0` on 20/20 origins.
Mechanism is dataset-specific: Criteo = the discounted-ONS optimizer
(persistence / aggregation add nothing); Avazu = cross-day persistence +
adaptive aggregation (the optimizer alone does nothing). The learned
slope is essential on both.
