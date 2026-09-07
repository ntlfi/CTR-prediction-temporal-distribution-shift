# AP-OPS additional experiments — resumption doc

Spec: the user's "Minimal Additional Experiments for AP-OPS" message
(2026-09-06, in the session that requested it — not saved as a repo file).
If this session ended: **read this first**, then `APOPS_FINDINGS.md`
section 7, then the code on branch `apops-additional-experiments`.

## Objective (3 questions the additional experiments must answer)

1. Does AP-OPS beat OPS under a **fully nested** rolling-origin protocol
   (nothing from the fixed-origin cut used in selection)?
2. Is the gain **cross-day persistence** or just the **projected-gradient →
   ONS optimizer** change?  → Persistent-ONS must beat Reset-ONS.
3. Does **adaptive aggregation** add value beyond one persistent
   calibrator?  → AP-OPS non-inferior to Persistent-ONS on both, better on ≥1.

No additional fixed-origin sweep. Final confirmation = the nested analysis
itself (user decision this session — no untouched stream exists in-repo;
Criteo 31d / Avazu 10d all used). Freeze algo + grid + rules anyway.

## Branch / commits

- Branch **`apops-additional-experiments`** off `main` (`ba83a19`).
- `73bfea6` — implementation corrections + nested runner (WIP, this doc's subject).
- NOT merged, NOT pushed. The first AP-OPS line (fixed test + dev-frozen
  rolling, "Improves" both datasets) is already on `main` (merge `cdb9d14`).

## DONE

### Implementation corrections (both required by the spec)

1. **Absolute-time feedback queue** — `apops/experts.py::replay_ons_stream`
   rewritten from a day-local maturation pointer to ONE chronological pass
   with maturation clock `d*86400 + (k+1)*block_sec + delay_sec`. A block
   near midnight now updates the persistent expert on the *next* day
   instead of being dropped. `OnsConfig.reset_daily` (False = persistent,
   True = Reset-ONS: re-init `theta=(1,0)`, `A=lam I` at each day boundary,
   queue still drained across the boundary → clean reset-vs-carry axis).
   Strict per-block delayed ONS step (one step per block, from that block's
   own matured impressions).
2. **`lambda_AP in [0,1]`** — `apops/aggregate.py::MetaConfig.lambda_ap`,
   `.prior()` = `(1-l, l/2, l/2)` over `(R,S,L)`. `w_1 = prior`;
   prior-centered fixed share `w = (1-alpha) w_tilde + alpha*prior`.
   `lambda_AP = 0` ⇒ `w=(1,0,0)` forever ⇒ AP-OPS == OPS prediction by
   prediction (bit-for-bit, tested).

### Methods — `apops/method.py::build_nested_rows`

`ops` (proj-grad, reset daily = `twoscale.calib.replay_day`),
`reset_ons` (discounted ONS, reset daily),
`persistent_ons` (discounted ONS, carried),
`ap_ops` (reset R + persistent S 4h / L 16h, aggregated with `lambda_AP`),
`no_slope` (AP-OPS with `a≡1`, supporting ablation only).
`reset_ons` & `persistent_ons` share ONE half-life (validation-selected).

### Tests — `apops_tests.py` (9/9 pass on 3% Criteo)

Added #7: cross-day-boundary maturation — a flipped last-block label
leaves its own day unchanged and changes the persistent expert on the
following day. Anchor test now exercises the `lambda_AP=0` path.

### Runner — `run_apops_nested.py`

Two seed-loops (memory: full-data Avazu ~40M rows × 3 seeds; never hold >1
seed's bank).
- **loop 1**: per seed, per origin — uncalibrated inner-val loss for the
  15 mixture configs. → pick common mixture per origin (mean over seeds).
- **loop 2**: per seed, per origin — with the common mixture fixed, replay
  `ops` / R / S / L / reset_ons(4h,16h) over days ≤ d; compute inner-val
  loss + day-d per-day metric for every candidate config
  (`lambda_AP∈{0,.25,.5,.75,1}` × `tau∈{4,16,inf}` = 15 for ap_ops/no_slope;
  `hl∈{4,16}` for reset/persistent). → freeze common `(lambda_AP,tau)` and
  `hl` per origin (mean inner loss over seeds), look up the stored day-d
  metric.
- Outer origins: Criteo 16–30, Avazu 5–9. Inner window: trailing 3 days.
- Frozen from `apops_selected.json` (dev only): `eta_m=100`, ONS `(lam,eta)`
  = Criteo (0.1, 0.25) / Avazu (1.0, 4.0), OPS `(B=0.25, eta0=0.3, const)`,
  block 900/3600, delay 1800, S/L half-lives 4h/16h.
- Output mirrors `run_rolling.py`: `seed<k>/per_day_metrics.csv`,
  `nested_origin_manifest.csv`, `nested_inner_losses.csv`,
  `nested_mixture_inner.csv`, `summary.json` (has
  `lambda_ap_selected_per_origin`, `lambda_ap_gt0_fraction`).

### Analysis — `run_apops_analysis.py --nested`

Seed-avg within day → bootstrap over days, every method paired vs `ops`.
`nested_decisions()` prints the 3 decision-rule verdicts + `lambda_AP>0`
fraction.

## IN PROGRESS / NEXT STEPS

1. **Smoke test** `run_apops_nested.py` on 3% Criteo — RUNNING as this doc
   was written (`/tmp/.../scratchpad/nested_smoke/`, loop 1 done ~7 min,
   loop 2 running). Verify: `per_day_metrics.csv` has 5 methods × 15 days ×
   3 seeds; `summary.json` `lambda_ap_selected_per_origin` populated;
   `run_apops_analysis.py --nested` produces sane tables. Then avazu smoke
   (`--source avazu --sample-frac 0.05`).
2. **Submit full jobs** (once smoke is clean, on branch):
   ```
   sbatch final_experiments/apops_nested_criteo.slurm   # ~1-3 h est
   sbatch final_experiments/apops_nested_avazu.slurm    # ~1.5-2 h est
   ```
   Both use `final_experiments/{ds}/apops/apops_selected.json` (already
   committed on main), write to `final_experiments/{ds}/apops/nested/`.
3. **Analyse** when both land:
   ```
   PYTHONPATH=. .venv/bin/python final_experiments/run_apops_analysis.py --nested \
     --dir final_experiments/criteo/apops/nested --label "Criteo -- nested rolling origin" \
     --dir final_experiments/avazu/apops/nested  --label "Avazu -- nested rolling origin" \
     --out final_experiments/APOPS_NESTED.md
   ```
4. **Fill `APOPS_FINDINGS.md` section 7 "Results"** with the 4-method
   table, the 3 decision-rule verdicts, `lambda_AP` selection frequency,
   temporal metrics. Update the Status table.
5. **Freeze** the algorithm + grid + decision rules in a short
   `APOPS_FROZEN.md` (for a future untouched dataset — the "final
   confirmation" the user deferred to the nested analysis).
6. **Commit** the nested outputs + findings. Ask the user before
   merging to `main` / pushing (previous AP-OPS line was merged only on
   explicit "merge push").

## Decision rules to apply (from the spec)

- **Persistence supported** iff `persistent_ons` improves on `reset_ons`
  (day-wt mean delta vs OPS). If `reset_ons ≈ ops`, neither optimizer nor
  the maturation fix matters and persistence is the whole story.
- **Adaptive aggregation supported** iff `ap_ops` non-inferior to
  `persistent_ons` on BOTH datasets and strictly better on ≥1. Else the
  final method simplifies to the single persistent calibrator.
- **AP-OPS succeeds** iff: beats OPS on both datasets in paired
  rolling-origin log loss; no systematic early-day / worst-day regression;
  `lambda_AP > 0` selected on a non-trivial fraction of origins.

## Prior expectation

§5 (preliminary) already showed AP-OPS ≈ single persistent expert on
Criteo, AP-OPS slightly ahead on Avazu, and no-slope worse than OPS on
both. So the likely nested outcome: **persistence supported** (Persistent-
ONS > Reset-ONS), **aggregation marginal** (AP-OPS ≈ Persistent-ONS on
Criteo, ahead on Avazu → keep AP-OPS for robustness + Avazu, or simplify
to Persistent-ONS if strict). `lambda_AP` likely > 0 on most origins but
not all.
