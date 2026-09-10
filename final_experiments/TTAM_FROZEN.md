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
| Criteo Attribution | 16,468,027 rows / 31 days | 16–30 (15 origins) | up to 3 latest eligible days before `d` | 900 s | 4 |
| Avazu | ~40.4M rows / 10 days | 5–9 (5 origins) | up to 3 latest eligible days before `d` | 3600 s | 3 |

- **Fully nested rolling origin.** For each outer origin `d`: inner validation = **up to
  the three latest eligible days** strictly before `d` (the first Avazu origin, `d = 5`,
  has only days 3 and 4); one configuration per knob is chosen, **shared across seeds
  0/1/2**, by mean inner-validation impression-weighted log loss across the seeds; the
  chosen configuration is replayed over the whole prefix `≤ d` to initialise its online
  state; day `d` is scored exactly once; day `d` informs only later origins.
- **No frozen-file inheritance.** Every data-tuned setting (ONS regularisation/step,
  meta learning rate `eta_m`, memory/aggregation, OPS settings, the AP-OPS **reset-anchor
  OPS** settings, gate/persistence, baseline knobs) is reselected at every origin from
  data strictly before `d`. The earlier rolling runs' tuning windows (Criteo 16–21,
  Avazu 5–6) are **not** reused. Constants left fixed a priori (independent of the
  evaluation days): backbone `alpha = 1e-4`, persistent-Platt `(a,b)` box
  `[0.2,5]×[-0.25,0.25]`, S/L expert half-lives `(4 h, 16 h)`, `delay_sec = 1800`.
- **Label maturity — enforced everywhere a cutoff reads labels**
  (`final_experiments/ttam/maturity.py`). At any training / selection / update cutoff
  `t` (an absolute time), a day-`k` impression at `sec_in_day = s` is usable iff
  `k·86400 + s + 1800 ≤ t`. This is applied to:
  - the **expert-bank fits** — for day `d`'s experts (cutoff `d·86400`), day `d-1`'s
    impressions after `s = 84600` are excluded; days `≤ d-2` are unaffected; the
    empty/single-class fallback uses the matured-history CTR, never `y` of the day being
    predicted;
  - the **inner-validation objective** (`iw_on`) — the latest inner day's post-84600 s
    tail is dropped at cutoff `d·86400`;
  - **AMG-TP's state summaries** (`_state_vector`: recent per-expert loss, CTR,
    disagreement, loss jump) and the **ARW / AdaMoE causal loss histories**
    (`_matured_day_logloss`: each day's summary is over labels matured by the next
    midnight);
  - the online calibrator queues (`apops` absolute-time queue, AMG-TP's period-update
    `matured` mask) — already had it.
  Pending (not-yet-matured) labels are never discarded from `bank[d].y`; they are simply
  excluded from any cutoff that precedes their arrival and picked up by the next cutoff
  that follows it. Expert-bank caches are keyed by `_d<delay>` so a pre-fix (leaky)
  cache is not reused.
- **Seeds** 0, 1, 2. Backbone `alpha = 1e-4`, fixed a priori (never tuned on any
  evaluation data, in this project or here).
- **Objective / tie-break.** Selection objective = mean inner-validation
  impression-weighted log loss across seeds. Deterministic tie-breaking: Best Fixed
  Window ties break toward the shorter nominal window (`HORIZONS5` order); grid
  `min(means, key=means.get)` returns the first key at equal value (insertion order of
  the frozen grid lists below).

## 2. Method variants (identical expert predictions, impressions, origins, seeds)

### 2a. Main table (6)

| variant | historical prediction | calibration |
|---|---|---|
| `expanding` | expanding-history expert | none |
| `best_fixed_window` | horizon `h*` on inner validation, frozen for day `d` | none |
| `arw` | causal single-elimination tournament over the 5 horizons | none |
| `adamoe` | causal EMA of inverse-loss softmax over mature past losses | none |
| `ops` | validation-fitted fixed simplex mixture `q` | daily-reset OPS (free `a`, `b`) |
| `ttam` | AMG-TP on the 5-horizon bank | AP-OPS |

### 2b. Ablation — 2×2 (revised 2026-09-09)

Cross **historical predictor ∈ {AMG-TP, expanding-history}** with
**calibration ∈ {AP-OPS, none}**. This measures each timescale's contribution
*with and without* the other, against a single pre-registered window baseline
rather than the validation-fitted mixture.

| | calibration = none | calibration = AP-OPS |
|---|---|---|
| **historical = expanding** | `expanding` (= main table) | `expanding_apops` |
| **historical = AMG-TP** | `amgtp_only` | `ttam` (= main table) |

- **The window baseline is expanding history**, chosen before examining the
  final results: the canonical B0 "train on all past data" predictor, zero
  tuning DOF, and the natural long-timescale counterpart to AMG-TP's adaptive
  memory.
- **Identical historical predictions are shared within each row**:
  `expanding` and `expanding_apops` both use `bank[d].preds["expanding"]`;
  `amgtp_only` and `ttam` both use the same selected AMG-TP q-stream. The
  AP-OPS algorithm + tuning rule is applied to each row's own input stream.
- Only `expanding_apops` and `amgtp_only` are ablation-only; `expanding` and
  `ttam` are shared with the main table.

Shared bank horizons `HORIZONS5 = (roll1, roll3, roll7, roll14, expanding)`; coinciding
truncated windows are fitted once and aliased (no double counting).

The validation-fitted simplex mixture is still fit in pass 1 and used only by
the main-table `ops` arm; it is no longer an ablation predictor.

## 3. Staged selection (affordable; documented, not claimed jointly optimal)

1. **Pass 1 — historical module** on *uncalibrated* inner loss: pick `h*` (BFW),
   `arw_delta`, `adamoe_lambda`, `amgtp_rho`, and the fixed simplex weight vector
   (exponentiated-gradient fit, deterministic).
2. **Pass 2a — reset-anchor OPS.** On each historical stream's causal inner predictions
   (fixed mixture / expanding history / AMG-TP), select daily-reset OPS `(B, eta0,
   schedule)` by inner loss. The fixed-mixture pick is the main-table `ops` arm; the
   expanding and AMG-TP picks are the **reset anchor R** for `expanding_apops` / `ttam`.
3. **Pass 2b — AP-OPS**, with the pass-2a anchor **fixed**: select `(ons_lam, ons_eta,
   lambda_AP, tau, eta_m)` — including the meta learning rate — separately on the
   expanding-history stream (`expanding_apops`) and the AMG-TP stream (`ttam`).
4. **Pass 3** — replay the frozen winners over the prefix, score the origin day once,
   store per-origin predictions.

Every candidate score and every selected value is written:
`pass1_module_inner.json`, `pass2a_anchor_inner.json` + `selected_anchors.json`,
`pass2b_apops_inner.json` + `selected_configs.json`.

## 4. Frozen grids (insertion order = tie-break order)

```
# --- pass 1: historical module ---
ARW_DELTA_GRID        = [0.05, 0.10, 0.20]
ADAMOE_LAMBDA_GRID    = [0.0, 0.25, 0.50, 0.75, 0.99]
AMGTP_RHO_GRID        = [0.2, 0.3, 0.5]            # gate/persistence memory EMA rate
BFW horizons          = roll1, roll3, roll7, roll14, expanding
simplex weights       = exponentiated-gradient fit (n_iter 300, lr 4.0), not a grid

# --- pass 2a: daily-reset OPS -- `ops` arm AND the AP-OPS reset anchor R,
#     one grid on each historical stream (fixed mixture / expanding / AMG-TP) ---
OPS_B_GRID            = [0.25, 0.5, 1.0]
OPS_ETA0_GRID         = [0.03, 0.1, 0.3]
OPS_SCHED_GRID        = ["const", "inv_sqrt"]
a_bounds             = (0.2, 5.0)   (OPS free intercept; slope unclamped by OPS grid)

# --- pass 2b: AP-OPS (anchor from 2a fixed), both streams ---
ONS_LAM_GRID          = [0.1, 1.0]                 # discounted-ONS ridge
ONS_ETA_GRID          = [0.25, 1.0, 4.0]           # discounted-ONS step
APOPS_LAMBDA_GRID     = [0.0, 0.25, 0.50, 0.75, 1.0]   # adaptive mass; 0.0 => AP-OPS == OPS
APOPS_TAU_GRID        = [4.0, 16.0, inf] hours     # delayed fixed-share switch half-life
APOPS_ETA_M_GRID      = [30.0, 100.0, 300.0]       # meta learning rate -- now SELECTED per origin
```

Fixed a priori (dataset-independent, chosen independently of the evaluation days,
**not** searched):

```
S_L_HALF_LIVES        = (4.0, 16.0) hours   # persistent short/long Platt expert memory
SEL_BOUNDS a_bounds   = [0.2, 5.0]   b_bounds = [-0.25, 0.25]   # persistent Platt (a,b) box
delay_sec             = 1800                # the label-availability rule itself
AMG-TP net            = lr 0.05, l2 1e-3, entropy_reg 1e-3, epochs_per_day 3,
                        beta_0 = sigmoid(-1.0), adaptive_beta True, linear persistence net,
                        context sketch m = 32 (seeded signed-hash, L2-normalised, shared)
backbone alpha        = 1e-4
```

Reuse audit ("any setting previously tuned on data overlapping the evaluation period
must be reselected"): the `ops` OPS grid, the ARW/AdaMoE/rho grids and the mixture fit
were already per-origin; **`eta_m` and the AP-OPS reset-anchor OPS settings are now also
per-origin** (previously carried as `eta_m = 100`, anchor `B 0.25 / eta0 0.3 / const`
from earlier tuning on overlapping days). The remaining constants above were fixed
independently of the outer days.

## 5. Pre-run checks (must pass before full runs — `ttam_tests.py`, 13/13)

1. `lambda_AP = 0` ⇒ AP-OPS ≡ daily-reset OPS on that stream, prediction by prediction
   (both `q_fixed` and `q_amgtp`).
2. Fixed-mixture weights nonnegative, sum to 1; AP-OPS meta weights stay on the simplex.
3. Persistent Platt experts keep `(a, b) ∈ [0.2, 5] × [-0.25, 0.25]`.
4. Causal invariance: flipping labels on/after the last prefix day leaves every earlier
   TTAM prediction byte-identical.
5. Cross-midnight maturation: flipping an interior day's last-block labels leaves that
   day's predictions unchanged and changes a later block of the next day.
5b. **Delayed label in the expert bank**: flipping day `k`'s unmatured tail
   (`sec_in_day > 84600`) leaves day `k` **and day `k+1`**'s expert predictions
   byte-identical (both fit cutoffs precede those labels' arrival) and changes day
   `k+2`'s.
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
  Brier score, calibration error. The **impression-weighted log loss** is one scalar per
  method — `Σ_{d,s} (mean_loss_{d,s} · n_{d,s}) / Σ_{d,s} n_{d,s}` over every (day, seed)
  cell — not a per-day impression-weighting re-averaged equally across days.

## 7. Downstream bidding (frozen — `run_ttam_bidding.py`)

- Criteo is the **primary** downstream experiment. Price proxy = recorded per-impression
  display `cost` (the price to beat to win that logged impression; **not** treated as an
  observed clearing price). Auction: `b_i = scale · pctr_i`, win iff `b_i ≥ cost_i`, pay
  `cost_i` on a win, receive logged `click_i`.
- Budget = 25% of each origin day's fixed reference cost (`sum(cost)` over its eligible
  impressions); daily reset; identical across paired methods per origin. Appendix budget
  fractions: 10%, 50%, 75%.
- If realised spend ≠ target, interpolate clicks to the common **per-origin target
  spend** on the method's own global-scale value-vs-spend frontier, bracket on **spend
  only** (never on test clicks). Every target must lie **inside a non-degenerate
  bracket** — `run_ttam_bidding.py` verifies this per cell and records it in
  `summary.json::bracket_check`; per-cell realised (paced) spend is in
  `bidding_cells.csv`. Results are labelled an **interpolated offline replay**, not a
  live policy.
- Report the result **as observed** — the current Criteo replay shows TTAM winning
  slightly *fewer* clicks at matched spend than every baseline despite its lower
  prediction loss; that is the finding, not a bug to tune away.
- Uncertainty: days paired after averaging training seeds; aggregate click-gain ratio of
  TTAM over each baseline recomputed in every one of 10,000 paired day resamples (seed
  20260908).
- **Avazu bidding is outside the main experiment** — no recorded price field; a simulated
  price model would need a separate specification frozen on past data. Criteo replay is
  the primary (and only) downstream result.

## 8. Completion criteria

Completion = the experiments and their limitations are reported, **regardless of whether
every comparison favours TTAM**. The figure retains unfavourable origins.

The 2×2 ablation is reported **relative to the Expanding control** as marginal effects
with paired day-bootstrap CIs: the AP-OPS (calibration) effect at each level of the
historical factor (`expanding_apops − expanding`, `ttam − amgtp_only`), the AMG-TP
(historical) effect at each level of the calibration factor (`amgtp_only − expanding`,
`ttam − expanding_apops`), and the **interaction** `(ttam − amgtp_only) −
(expanding_apops − expanding)`. A negative interaction CI would mean AMG-TP makes AP-OPS
help more (synergy); a CI containing zero means the two timescales are additive. Synergy
is not inferred merely because the full method has the lowest mean. The **different
interaction directions across the two datasets are retained as reported** (Criteo and
Avazu disagree in sign), and Avazu's five-origin intervals are treated as descriptive.
