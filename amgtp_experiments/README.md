# AMG-TP experiments

Complete experimental battery for the plan in
[`../AMG-TP_Academic_LaTeX.pdf`](../AMG-TP_Academic_LaTeX.pdf). Run in two stages.

## Stage 1 — `stage1_m5b_high_smooth/`

Benchmarks **M5b-high-smooth** (`m5_multiscale_gate.run_m5` with `smooth_reg=0.1`,
labelled `m5b_smooth0.1`) against the full baseline suite from PDF §5.1, over the
synthetic shift suite **S0–S6** (PDF Table 2, `synthetic_data.py` — S5
`opposing_local` and S6 `mixed` were added for this plan) plus the real Criteo
chronological benchmark.

- Methods per cell: expanding, rolling 1/3/7/14, validation-selected window,
  exponential forgetting (`decay_hl1/3/7`), Han ARW, Differentiable Forgetting,
  AdaMoE, uniform-5 average, M1, M2, the M5b `smooth_reg` grid
  `{0, 1e-3, 1e-2, 1e-1, 3e-1}` (which also serves as ablations A1/A2/A3 and the
  oracle-fixed-smoothness diagnostic), and `ensemble3` as a diagnostic ceiling.
- Metrics: log loss, Brier, PR-AUC, ROC-AUC, calibration error; per-day curves;
  per-subgroup losses; recovery time / peak post-shift excess / cumulative
  post-shift regret vs the per-day oracle horizon; stationary downside; gate
  movement and effective horizon; day-level bootstrap CIs.
- Oracle diagnostics (analysis only): best fixed horizon, per-day and per-group
  oracle horizon, per-day oracle persistence regime.
- Seeds: dev `[0,1,2,3,4]` (hyperparameters were frozen on these in earlier
  project work — see `../results/m5_analysis.md`), confirmation `[20..31]`
  (disjoint, 12 seeds).

```
.venv/bin/python amgtp_tests.py                       # leakage + edge-case checks
sbatch amgtp_stage1_synthetic.slurm                   # 119 (regime, seed) cells
sbatch amgtp_stage1_criteo.slurm                      # 3 Criteo seeds
.venv/bin/python amgtp_aggregate.py                   # tables/, figures/, REPORT.md
```

Layout: `stage1_m5b_high_smooth/<regime>/seed<N>/{summary.json,per_day_metrics.csv,
group_per_day_metrics.csv,gate_dynamics.csv,oracle_per_day.csv,comparison_table.csv}`.

## Stage 2 — `stage2_amgtp/`

Builds **AMG-TP** itself (adaptive global persistence `β_t` on top of M5b's
5-expert gate, PDF §2–3) and re-runs the same battery plus the AMG-TP-specific
ablations A4–A7, with M5b-high-smooth as the fixed-high-persistence reference
(A3). Started only after Stage 1 is complete and reviewed.
