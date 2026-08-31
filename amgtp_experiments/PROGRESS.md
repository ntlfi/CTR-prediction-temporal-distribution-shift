# AMG-TP progress log

Running log so work-in-progress is not lost between sessions. Newest entry on
top. See `REPORT.md` in each stage dir for the aggregated numbers; this file is
the narrative of what was done, what is running, and what is next.

---

## 2026-08-31 — Downstream autobidding eval (PDF §8, step 8)

**Status: built + tested + smoke-run; batteries not yet submitted.**

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

**Status: multi-seed battery running.**

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
