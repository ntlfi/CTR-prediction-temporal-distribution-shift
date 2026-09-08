# TTAM revised Section 6 — frozen protocol, grids, and decision rules

Frozen **before** inspecting any TTAM result, per `TTAM_Additional_Experiments_Plan.pdf`
(8 September 2026) section 6 ("Freeze candidate grids, search budgets, objective, and
deterministic tie-breaking before inspecting the new results") and section 11 order 1
("Freeze corrected protocol and implementation"). This file is the audit record; the
runnable source of truth is `final_experiments/run_ttam_nested.py` (the constants block)
and `final_experiments/ttam/`.

Code commit at freeze: see `git rev-parse HEAD` recorded inside every
`selected_configs.json` / `summary.json` the runner writes.

---

## 1. Datasets, splits, protocol

| dataset | full stream | outer test origins (0-based day idx) | inner validation | block | warmup |
|---|---|---|---|---|---|
| Criteo Attribution | 16,468,027 rows / 31 days | 16–30 (15 origins) | 3 latest eligible days before `d` | 900 s | 4 |
| Avazu | ~40.4M rows / 10 days | 5–9 (5 origins) | 3 latest eligible days before `d` | 3600 s | 3 |

- **Fully nested rolling origin.** For each outer origin `d`: inner validation = the
  three latest eligible days strictly before `d`; one configuration per knob is chosen,
  **shared across seeds 0/1/2**, by mean inner-validation impression-weighted log loss
  across the seeds; the chosen configuration is replayed over the whole prefix `≤ d` to
  initialise its online state; day `d` is scored exactly once; day `d` informs only
  later origins.
- **No frozen-file inheritance.** Every data-tuned setting (ONS regularisation/step,
  meta learning rate, memory/aggregation, OPS settings, gate/persistence, baseline
  knobs) is reselected at every origin from data strictly before `d`. The earlier
  rolling runs' tuning windows (Criteo 16–21, Avazu 5–6) are **not** reused.
- **Maturity rule.** A label is eligible only once its `delay_sec = 1800` s delay has
  elapsed. Applied to backbone training, gates, historical loss summaries, HPO
  selection, and calibration. At midnight, unmatured labels are held out of that
  period's update and folded into the next period (feature rows preserved).
- **Seeds** 0, 1, 2. Backbone `alpha = 1e-4`, fixed a priori (never tuned on any
  evaluation data, in this project or here).
- **Objective / tie-break.** Selection objective = mean inner-validation
  impression-weighted log loss across seeds. Deterministic tie-breaking: Best Fixed
  Window ties break toward the shorter nominal window (`HORIZONS5` order); grid
  `min(means, key=means.get)` returns the first key at equal value (insertion order of
  the frozen grid lists below).

## 2. Nine method variants (identical expert predictions, impressions, origins, seeds)

| variant | historical prediction | calibration | role |
|---|---|---|---|
| `expanding` | expanding-history expert | none | main |
| `best_fixed_window` | horizon `h*` on inner validation, frozen for day `d` | none | main |
| `arw` | causal single-elimination tournament over the 5 horizons | none | main |
| `adamoe` | causal EMA of inverse-loss softmax over mature past losses | none | main |
| `ops` | validation-fitted fixed simplex mixture `q` | daily-reset OPS (free `a`, `b`) | main |
| `ttam` | AMG-TP on the 5-horizon bank | AP-OPS | main + ablation |
| `without_both` | validation-fitted fixed simplex mixture `q` | none | ablation |
| `amgtp_only` | AMG-TP | none | ablation |
| `apops_only` | validation-fitted fixed simplex mixture `q` | AP-OPS | ablation |

Shared bank horizons `HORIZONS5 = (roll1, roll3, roll7, roll14, expanding)`; coinciding
truncated windows are fitted once and aliased (no double counting). `amgtp_only` /
`ttam` share their selected AMG-TP module; `apops_only` / `ttam` share the AP-OPS
algorithm and tuning rule applied to their own input stream.

## 3. Staged selection (affordable; documented, not claimed jointly optimal)

1. **Pass 1 — historical module** on *uncalibrated* inner loss: pick `h*` (BFW),
   `arw_delta`, `adamoe_lambda`, `amgtp_rho`, and the fixed simplex weight vector
   (exponentiated-gradient fit, deterministic).
2. **Pass 2 — calibration** on the chosen module's causal inner prediction stream:
   pick OPS `(B, eta0, schedule)` for the `ops` arm on the fixed-mixture stream; pick
   AP-OPS `(ons_lam, ons_eta, lambda_AP, tau)` separately on the fixed-mixture stream
   (`apops_only`) and on the AMG-TP stream (`ttam`).
3. **Pass 3** — replay the frozen winners over the prefix, score the origin day once,
   store per-origin predictions.

## 4. Frozen grids (insertion order = tie-break order)

```
# --- pass 1: historical module ---
ARW_DELTA_GRID        = [0.05, 0.10, 0.20]
ADAMOE_LAMBDA_GRID    = [0.0, 0.25, 0.50, 0.75, 0.99]
AMGTP_RHO_GRID        = [0.2, 0.3, 0.5]            # gate/persistence memory EMA rate
BFW horizons          = roll1, roll3, roll7, roll14, expanding
simplex weights       = exponentiated-gradient fit (n_iter 300, lr 4.0), not a grid

# --- pass 2: OPS arm (fixed-mixture stream) ---
OPS_B_GRID            = [0.25, 0.5, 1.0]
OPS_ETA0_GRID         = [0.03, 0.1, 0.3]
OPS_SCHED_GRID        = ["const", "inv_sqrt"]
a_bounds             = (0.2, 5.0)   (OPS free intercept; slope unclamped by OPS grid)

# --- pass 2: AP-OPS arms (both streams) ---
ONS_LAM_GRID          = [0.1, 1.0]                 # discounted-ONS ridge
ONS_ETA_GRID          = [0.25, 1.0, 4.0]           # discounted-ONS step
APOPS_LAMBDA_GRID     = [0.0, 0.25, 0.50, 0.75, 1.0]   # adaptive mass; 0.0 => AP-OPS == OPS
APOPS_TAU_GRID        = [4.0, 16.0, inf] hours     # delayed fixed-share switch half-life
```

Fixed a priori (dataset-independent, **not** searched):

```
APOPS_ETA_M           = 100.0        # meta learning rate (block-loss scale)
S_L_HALF_LIVES        = (4.0, 16.0) hours   # persistent short/long Platt expert memory
SEL_BOUNDS a_bounds   = [0.2, 5.0]   b_bounds = [-0.25, 0.25]   # persistent Platt (a,b) box
delay_sec             = 1800
ANCHOR_OPS_HP         = B 0.25 / eta0 0.3 / const / a_bounds (0.2, 5.0)
                        (AP-OPS reset anchor R = repo-verified OPS component setting, frozen)
AMG-TP net            = lr 0.05, l2 1e-3, entropy_reg 1e-3, epochs_per_day 3,
                        beta_0 = sigmoid(-1.0), adaptive_beta True, linear persistence net,
                        context sketch m = 32 (seeded signed-hash, L2-normalised, shared)
```

Reuse audit (plan section 6 "any backbone setting previously tuned on later data must
be reselected or fixed independently of those data"): the OPS `(B, eta0, schedule)`
grid, the ARW/AdaMoE grids, and the mixture fit are all reselected per origin here.
`APOPS_ETA_M`, the S/L half-lives, the `(a,b)` box, and `ANCHOR_OPS_HP` were fixed
before the outer days from the AP-OPS component study (`APOPS_FROZEN.md`) and are
carried as constants, not re-tuned.

## 5. Pre-run checks (must pass before full runs — `ttam_tests.py`, 10/10)

1. `lambda_AP = 0` ⇒ AP-OPS ≡ daily-reset OPS on that stream, prediction by prediction
   (both `q_fixed` and `q_amgtp`).
2. Fixed-mixture weights nonnegative, sum to 1; AP-OPS meta weights stay on the simplex.
3. Persistent Platt experts keep `(a, b) ∈ [0.2, 5] × [-0.25, 0.25]`.
4. Causal invariance: flipping labels on/after the last prefix day leaves every earlier
   TTAM prediction byte-identical.
5. Cross-midnight maturation: flipping an interior day's last-block labels leaves that
   day's predictions unchanged and changes a later block of the next day.
6. Deployed-weight memory update: AMG-TP `m_t = (1-rho) m_{t-1} + rho · mean_x pi_deployed_t`
   (using the weights actually deployed for day `t`, not re-derived post gate step).
7. Determinism: identical inputs reproduce identical variant outputs.

## 6. Statistics (frozen — `ttam_stats.py`)

- `L_{m,d,s}` = mean impression log loss of method `m` on origin `d`, seed `s`.
- Seeds averaged **first**: `A_{m,d} = mean_s L_{m,d,s}`; equal-day score
  `A_m = mean_d A_{m,d}`; daily gain `G_{m,d} = A_{m,d} - A_{TTAM,d}` (positive ⇒ TTAM
  lower loss); mean gain `G_m = mean_d G_{m,d}`.
- 95% **paired day-bootstrap** interval for `G_m`: 10,000 resamples, resampling seed
  **20260908**, the same sampled days applied to both methods after seed-averaging.
- Serial-dependence sensitivity: block-2 moving-block bootstrap, same seed.
- Report `A_m`, `G_m`, the interval, and origins won. Impressions and seed×day cells
  are never independent replicates. With D = 5 Avazu origins both interval estimates
  are descriptive — no significance stars, no universal-superiority language.
- Appendix only, never mixed with the equal-day gain: impression-weighted log loss,
  Brier score, calibration error.

## 7. Downstream bidding (frozen — `run_ttam_bidding.py`)

- Criteo is the **primary** downstream experiment. Price proxy = recorded per-impression
  display `cost` (the price to beat to win that logged impression; **not** treated as an
  observed clearing price). Auction: `b_i = scale · pctr_i`, win iff `b_i ≥ cost_i`, pay
  `cost_i` on a win, receive logged `click_i`.
- Budget = 25% of each origin day's fixed reference cost (`sum(cost)` over its eligible
  impressions); daily reset; identical across paired methods per origin. Appendix budget
  fractions: 10%, 50%, 75%.
- If realised spend ≠ target, interpolate clicks to the common spend on the method's own
  global-scale value-vs-spend frontier, bracket on **spend only**, label interpolated.
- Uncertainty: days paired after averaging training seeds; aggregate click-gain ratio of
  TTAM over each baseline recomputed in every one of 10,000 paired day resamples (seed
  20260908).
- **Avazu bidding: pending** — no recorded price field; a simulated price model needs a
  separate specification frozen on past data. Criteo replay is reported as the primary
  downstream result.

## 8. Completion criteria

Completion = the experiments and their limitations are reported, **regardless of whether
every comparison favours TTAM**. The figure retains unfavourable origins. Section 6.3
discussion reports whether the direction/magnitude of `A_{apops_only} - A_{ttam}` and
`A_{amgtp_only} - A_{ttam}` differ across datasets; synergy is not inferred merely
because the full method has the lowest mean.
