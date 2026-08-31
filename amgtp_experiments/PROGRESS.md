# AMG-TP progress log

Running log so work-in-progress is not lost between sessions. Newest entry on
top. See `REPORT.md` in each stage dir for the aggregated numbers; this file is
the narrative of what was done, what is running, and what is next.

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
