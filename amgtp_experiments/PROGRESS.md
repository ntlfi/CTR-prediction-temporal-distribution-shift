# AMG-TP progress log

Running log so work-in-progress is not lost between sessions. Newest entry on
top. See `REPORT.md` in each stage dir for the aggregated numbers; this file is
the narrative of what was done, what is running, and what is next.

---

## 2026-09-01 — S7 opposing_recurring regime + Extension A (hidden persistence net)

**Status: implemented + all unit tests pass. No battery run yet.**
Also pushed earlier this session: `amgtp_experiments/findings_summary.tex`/`.pdf`
(7pp, commit 873e25e) — consolidated evidence that AMG-TP beats prior methods.

### S7 `opposing_recurring` — new synthetic drift mode (`synthetic_data.py`)
Motivation: S5 `opposing_local`'s per-group differential is a one-off transient
that lands entirely in the *dev* period (shifts at n/3, n/4 vs locked-test
window = last 30%), so it can't test per-example persistence in the locked
test. S7 fixes this: both subpopulations oscillate on the w0↔w1 axis with
period `period_days`, **group B a quarter period behind group A**
(`a_a = 0.5(1+sin ωt)`, `a_b = 0.5(1+sin(ωt+π/2))`). At every block one group
sits near a turning point (slow local change → high persistence optimal) while
the other is at max slope (fast → low persistence optimal); they swap over the
cycle. A single global β_t is provably wrong for one group at all times, and
q_t(x) alone can't fix it (routes experts, not persistence). No w2 drawn →
RNG stream for the original 7 modes byte-for-byte unchanged (verified by test).
- `DRIFT_MODES` += `opposing_recurring`; `SYNTH_REGIMES["s7_opposing_recurring"]`
  in `amgtp_config.py` → `synth_grid()` now 136 cells (8 regimes × 17 seeds);
  indices 0–118 unchanged, s7 is 119–135.
- Sanity (120d/3000rows/seed0): group A range 0.085, B range 0.102,
  corr(A CTR, B CTR) = −0.69 (plain recurring: +0.81), A peaks block 3 / B
  block 0.

### Extension A — hidden-layer PersistenceNet (`amgtp_method.py`)
`PersistenceNet(n_features, init_bias, hidden=0)`. `hidden=0` = the Stage 2
single linear layer, **bit-identical** (tested). `hidden>0` = one tanh hidden
layer of that width; **output layer still zero-init** (weight 0, bias
init_bias) so day-0 β = σ(init_bias) exactly and hidden>0 is a strict superset
that must learn any curvature. `run_amgtp(..., persist_hidden=H)`. Hidden params
get the existing L2 on `trainable`.
- `amgtp_run.py --stage2` now also runs `amgtp_hidden8`, `amgtp_hidden16`
  (ablation A10). `amgtp_aggregate.py`: A10 rows in `ABLATION_LADDER`,
  `amgtp_hidden8/16` in `KEY_REFERENCES`.
- `amgtp_tests.py`: +5 checks (hidden=0 identity, day-0 β, β∈[0,1], finite,
  reproducible) + S7 isolation + S7 RNG-untouched fingerprint. **All 24 pass.**
- Smoke (`amgtp_run.py` 60d/800rows/s7/stage2, 54s): full pipeline incl.
  hidden8/16 runs end-to-end; too small to show the S7 effect (all methods
  ~0.438, drift barely visible at that size — expected).

### DONE this session
- **Extension A hidden sweep** (job 12456521): H=0 wins, H∈{4,8,16} all
  4th-decimal worse. Frozen h=0. `stage4_hidden/_sweep/FROZEN.md`.
- **S7 battery** (job 12456557, 17 cells into `stage2_amgtp/`,
  ~5-6min/cell): unified S0–S7 REPORT regenerated. **S7 result: global-β
  AMG-TP does NOT adapt** — ties han_arw/expanding (−0.0001, p=0.73),
  WORSE than both M5b specialists (+0.0033 vs smooth0.1 which wins S7 at
  0.4326, +0.0003 vs smooth0.001). amgtp_global_q ≈ amgtp (context gate
  doesn't rescue it). oracle-persistence-flip 66% → real headroom for
  per-example β. A10 hidden8/16 confirmed negative on S7 confirm seeds.
- `findings_summary.tex`/`.pdf` updated (8pp): S7 row everywhere +
  §"S7: where a global β_t is not enough" + Extension A negative
  documented. Recompiles clean (pdflatex ×2).

## 2026-09-01 (cont.) — Extension B (per-example β_t(x)) = NEGATIVE

`amgtp_method.py`: `BetaContextHead` g_ξ (zero-init) → `β_t(x) =
σ(r_ψ(s_{t-1}) + g_ξ(feats_t(x)))`. `run_amgtp(beta_per_example,
beta_var_reg, beta_hidden, group)`, records `beta_std`, per-group
`beta_A/beta_B`. Global path bit-identical (tested). `amgtp_run.py
--beta-x` → A11-A13 + per-group β trace. 30/30 tests pass. Commits
10791d4, 4dedfe7.

**Dev sweep** (`amgtp_betax_sweep.py`, job 12457082, 25 cells, 5 dev
seeds × S0/S3/S4/S5/S7, `beta_var_reg` ∈ {0,1e-4,1e-3,1e-2,1e-1}):
**no config beats global β_t** — S7 (the target) betax +0.0007 vs global
(0.4318 vs 0.4311); worse on S3/S0, tiny edge S4 (−0.001), tie S5. **g_ξ
never learns the subgroup split**: |β_A − β_B| ≈ 0.005 at every λ, β_std
≈ 0.01 (β_t(x) ≈ global β_t(t)). Mechanism: `m_{t-1}` is a single global
EMA, so per-example β routing sends a stable-subgroup example to the
*blended* history, not that subgroup's own. Per-example persistence needs
per-example/per-group `m` — PDF §2.3 scopes that out.
`stage4_betax/_sweep/FROZEN.md`.

**Confirmed on 12 disjoint S7 seeds** (job 12457141): amgtp_bx −0.0007
vs global amgtp (i.e. amgtp is 0.0007 *better*), 12/12. Negative holds
out-of-sample.

### Per-group `m` diagnostic (the hypothesised fix) — ALSO NEGATIVE
`run_amgtp(persist_per_group=True)` = DIAGNOSTIC (uses the group label like
the amgtp_eval oracles): a separate persistent state m^(g) per subgroup,
EMA'd from that subgroup's own deployed weights. S7, 1 seed, 3000 rows:
- global β_t: 0.4229
- β_t(x) + global m: 0.4232
- **β_t(x) + per-group m: 0.4236** (worse still)
- global β + per-group m: 0.4229 (unchanged)
- |β_A − β_B| still ≈ 0.008 — **g_ξ never differentiates the subgroups
  even with an oracle group label AND a per-group m available.**

So "global m is the bottleneck" is **refuted**. Mechanistic read: the
optimal β_A-vs-β_B relationship *flips sign every quarter cycle* (whichever
group is at its turning point wants high β), so a time-invariant
g_ξ(context) cannot represent it — it needs a phase×context interaction,
which M5c already showed this architecture won't learn; and the direct
per-example volatility signal (|p_short(x) − p_long(x)|) is apparently
below the per-impression noise floor. Three variants of "smarter
per-example persistence" now fail with one coherent explanation.

**Verdict: the per-example-persistence direction is a validated dead end
for this problem.** Global linear β_t remains the deployed model. AMG-TP's
S7 boundary (global β can't beat the fixed high-persistence specialist
when two subpops want opposite persistence) stands as a documented limit,
not something a smarter β fixes.

---

## 2026-08-31 — Downstream autobidding eval (PDF §8, step 8)

**Status: DONE. 56 synthetic cells + 5 Criteo seeds, all COMPLETED (job
12455401 / 12455402). `amgtp_experiments/stage3_autobid/REPORT.md`.**

### RESULT — the prediction wins translate to bidding value

Value at matched spend (clicks won at 25% of historical budget), AMG-TP vs
Han ARW, paired over 8 seeds:

| regime | rel % | seeds | Wilcoxon p | verdict |
|---|---|---|---|---|
| S3 recurring | **+1.53%** | 8/8 | 0.008 | **AMG-TP wins** |
| S4 local | **+2.09%** | 8/8 | 0.008 | **AMG-TP wins** |
| S1 abrupt | +0.17% | 7/8 | 0.016 | marginal edge (beats m5b_smooth0.1 +2.1%) |
| S5 opposing | +0.12% | 5/8 | 0.38 | tie |
| S0 / S2 | −0.3% / −0.25% | 0/8 | 0.008 | small loss |
| S6 mixed | −1.02% | 0/8 | 0.008 | **loss** |
| Criteo (real) | +0.01% | 4/5 | 0.125 | flat, no separation |

- vs `ensemble3`: within ±0.25% everywhere — one model matches the 3-way
  ensemble downstream too.
- Anchors sane: `_oracle` ~+45% over no-skill, `_shuffled` well below.
- **The causal chain (PDF eq. 9) holds on synthetic: a CTR-prediction gain
  under recurring/local drift → more clicks per dollar at matched spend.** On
  real Criteo nothing separates from a cheapest-first bidder (PDF §9
  "insufficient real evidence" carries downstream).
- Caveat: synthetic `cost` is tied to *true* CTR, so the eval measures
  whether a prediction gain *translates*, not the absolute value level.

New "new experiment" chosen by the user. PDF §8: freeze the CTR models, feed
each into the *same* auction + pacing policy, compare realised value at
matched spend.

### Code (all committed)
- **`autobid.py`** — the simulator (pure, `autobid_tests.py` = 7 groups, all pass):
  - `linear_frontier` — global bid-scale sweep → the value–spend frontier.
  - `paced_auction` — fixed budget, per-block pacing (take impressions by
    decreasing `pctr/cost` until the block allowance is spent; unspent
    carries over). Deterministic, controller-tuning-free.
  - `value_at_matched_spend` — interpolate value onto a common spend grid.
  - `synthetic_cost` — a documented synthetic second-price landscape:
    `cost_i = floor + scale·p_true_i·lognormal`, tied to the *true* CTR (not
    to any model under test), so clicked-likely impressions are genuinely
    more expensive — the regime where prediction skill pays off.
  - `load_criteo_bidding` — Criteo log keeping `cost`/`conversion`/`cpo`.
  - reference bidders: `_oracle` (pctr = realised click), `_noskill`
    (constant pctr → cheapest-first), `_shuffled_amgtp`.
- **`run_autobid.py`** — driver. `--source {criteo, synthetic}`. Runs the
  frozen battery (expanding, rolling_7, han_arw, m2, m5b_smooth0.1,
  ensemble3, amgtp@AMGTP_CONFIG) → per-impression test preds → both policies.
  Writes `autobid_frontier.csv`, `autobid_matched_spend.csv`,
  `autobid_paced.csv`, `autobid_frontier.png`, `summary.json`.
- **`synthetic_data.py`** — added `return_p` to `generate_synthetic_ctr` and a
  `p_true` column to `generate_synthetic_raw` (backward compatible; RNG stream
  unchanged, `amgtp_tests.py` reproducibility test still passes).
- **`amgtp_config.py`** — `AUTOBID_DIR = stage3_autobid`, `AUTOBID_SEEDS 0-7`,
  `autobid-cell` / `autobid-ncells` (56 synthetic cells = 7 regimes × 8 seeds).
- **`autobid_synthetic.slurm`** (array 0-55%32), **`autobid_criteo.slurm`**
  (array 0-4, full dataset).
- **`autobid_aggregate.py`** — per-regime paired AMG-TP vs {han_arw, expanding,
  ensemble3, m5b_smooth0.1} on value-at-matched-spend, Wilcoxon over seeds →
  `stage3_autobid/REPORT.md` + tables + figure.

### Smoke findings (single seed, small — NOT the real result)
- **Criteo** (sample_frac 0.02): every deployable method ≈ `_noskill` at
  matched spend; only `_oracle` pulls clearly ahead. At *low* budget (2-10%
  spend) the CTR models beat `_noskill`/`_shuffled` by a few %, but the
  expanding/amgtp/m2/ensemble3 spread is <0.5% — i.e. the shallow-drift
  Criteo log-loss differences do **not** translate into bidding-value
  separation. Consistent with PDF §9 "insufficient real evidence".
- **Synthetic abrupt** (60d×2000): real separation — `ensemble3`/`han_arw`/
  `m5b_smooth0.1`/`amgtp` all ~4200 clicks @25% spend vs `expanding`/
  `_noskill` ~3830 vs `_shuffled` ~3325; `_oracle` ~6395. So better CTR
  prediction under drift **does** convert to value at matched spend. AMG-TP ≈
  Han ARW on abrupt (matches the Stage 2 prediction result).

### Next
1. Submit `autobid_synthetic.slurm` + `autobid_criteo.slurm` (after a
   full-size 120d×4000 timing check — running now, task bh1vc7xvp).
2. `autobid_aggregate.py` → does AMG-TP's S3/S4 *prediction* win show up as a
   *bidding-value* win? That's the headline question.
3. Fold into REPORT + README + push.

---

## 2026-08-31 — Avazu (second real dataset, PDF §5.3)

**Status: DONE. 8 seeds, all COMPLETED (jobs 12453463_0 + 12455275_[1-7]).
Both stage REPORTs regenerated.**

### RESULT — first real-data signal that AMG-TP > Han ARW

8-seed paired (Wilcoxon), locked-test log loss:
- **AMG-TP beats Han ARW: −0.32%, 8/8 seeds, p=0.0078** (the min p for n=8);
  CI [−0.0014, −0.0011] excludes 0.
- **AMG-TP beats expanding ERM: −0.49%, 8/8, p=0.0078**; beats rolling_7/14 too.
- Absolute: amgtp **0.3825** vs han_arw 0.3838 vs expanding 0.3844.
- Does *not* separate from the other mixture methods — adamoe / m5b_smooth* /
  ensemble3 / amgtp_no_state all cluster 0.3825–0.3827. It's the
  *multi-timescale family as a whole* that edges out the single-horizon
  methods, exploiting the real diurnal cycle (~12 blocks, inside the window
  family's reach).
- Effect is small in absolute log loss (~0.0012–0.0019) but statistically
  clean — better than Criteo's dead-flat. Weakly satisfies PDF H4 ("improves
  at least one real CTR benchmark over the best fixed- and adaptive-window
  baselines").

### Fixed at write-up
- `amgtp_aggregate.py`: stale "hourly blocks (~240)" boilerplate → "2-hour
  blocks (120-block horizon)"; grammar; and real regimes (criteo/avazu) now
  pair over *all* present seeds (were being truncated to DEV_SEEDS [0-4]).

### Done
- **Avazu integration** (`avazu_data.py`, `data_source.py`, `download_data.py`,
  `amgtp_config.py`, `amgtp_run.py`, `amgtp_aggregate.py`): `--source avazu`
  now selects the Kaggle avazu-ctr-prediction data via the `reczoo/Avazu_x4`
  BARS mirror (no Kaggle auth). `avazu_data.py` reassembles the chronological
  stream from the intact `hour` field (the mirror's 8:1:1 split is randomly
  ordered) and indexes time in **2-hour blocks** (120-block horizon, matching
  the synthetic suite). `id` and `hour` are dropped from the features so the
  temporal methods must discover the diurnal cycle from loss dynamics, not be
  handed the clock (same discipline as the M5c fair-test).
- Each seed draws a disjoint 20% row subsample of the 40.4M rows, so seeds are
  a genuine source of variation (unlike full Criteo, where 3 seeds are nearly
  degenerate).
- `amgtp_aggregate.py` generalised: `REAL_REGIMES = ["criteo", "avazu"]`,
  handled like synthetic regimes but with no injected change point.
- Data downloaded to `/insomnia001/home/tn2447/data/avazu/Avazu_x4.zip`
  (1,306,331,539 bytes).
- **Seed 0 validation run** (SLURM 12453463_0, ins080, 46:49, exit 0): full
  battery ran end-to-end, aggregation + all 4 figures generated without error.

### Seed-0 locked-test result (single seed — CIs all overlap, like Criteo)
| method | log loss |
|---|---|
| **amgtp** | **0.38235** |
| amgtp_no_state / amgtp_global_q | 0.38235 / 0.38236 |
| m5b_smooth0.001 | 0.38236 |
| ensemble3 / adamoe / m5b_smooth0 | ~0.3824 |
| m5b_smooth0.1 | 0.38248 |
| han_arw = rolling_14 (Han ARW picks h=14 every day) | 0.38398 |
| expanding | 0.38435 |
| rolling_1 / diff_forgetting / decay_hl* | 0.389–0.400 |

- AMG-TP nominally top, beats Han ARW by ~0.4% — within noise on one seed.
- `oracle_persistence_switch_frac = 0.44`: the real diurnal cycle **does** make
  the optimal fixed persistence flip day-to-day → adaptive β_t has real room,
  same story as synthetic S3.
- 8.09M rows @ 20% subsample, 120 blocks, 107 eligible → 75 dev / 32 locked test.

### Running
- **SLURM 12455275** `--array=1-7` (`amgtp_avazu.slurm`): seeds 1–7, same battery.
  Submitted 2026-08-31 ~17:07. → 8 seeds total, paired Wilcoxon can reach p≈0.008.
- Background queue poller: task `b68evf6x5`.

### Next
1. When 12455275 finishes: `amgtp_aggregate.py` for both stages, check
   `paired_avazu.csv` (AMG-TP vs Han ARW, 8 seeds).
2. Fix stale boilerplate in `amgtp_aggregate.py` build_report: says
   "hourly blocks (~240)" but runs use 2-hour blocks (120). Also fixes the
   "N seeds" grammar.
3. Write the Avazu section of both REPORT.md files + README results section.
4. Commit + push all Avazu results.

### Known gaps / TODO after Avazu
- Autobidding still gated: needs a real dataset where *something* separates from
  noise. Criteo flat; Avazu seed-0 also tight — 8-seed result will tell whether
  it clears the bar.
- **Example-specific β_t(x)** and a **hidden-layer persistence net** are still
  untried (PDF §3 defers β_t(x) until global β_t is understood — it now is).
  `PersistenceNet` is currently a single linear layer (zero-init weights,
  bias = init_bias).
