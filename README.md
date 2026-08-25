# CTR-prediction-temporal-distribution-shift

Chronological CTR benchmark comparing rules for using historical data
(expanding history, fixed rolling windows, exponential forgetting,
validation-selected window), before attempting any new temporal-adaptation
method. Scope and protocol are defined in [`first-step-baselines.pdf`](first-step-baselines.pdf);
this is the P0 stage of that plan.

## Setup

```bash
./setup.sh
source .venv/bin/activate
```

## Get the data

```bash
python download_data.py            # downloads to data/criteo_attribution_dataset.tsv.gz
```

Source: [Criteo Attribution Modeling for Bidding Dataset](https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/)
(30 days of impressions, click label, campaign + 9 anonymized contextual
categories). Only pre-bid context (campaign, cat1-cat9) is used as model
input; cost, conversion, and click-position fields are dropped since they
either happen after the impression or leak the outcome.

**The dataset's original host is currently down** (both `go.criteo.net` and
the S3 mirror return 404 as of 2026-08). If `download_data.py` fails, fetch
the file from a mirror (e.g. the Kaggle copy "criteo-attribution-modeling" by
sharatsachin) and place it at `data/criteo_attribution_dataset.tsv.gz`
yourself; the script checksums whatever is there
(sha256 `94ac7a46...2391a98`, 653,015,824 bytes) and tells you if it doesn't
match the known-good copy.

## Run the P0 baselines

```bash
python run_baselines.py --sample-frac 0.2   # quick preliminary pass
python run_baselines.py                     # full dataset
```

For each prediction day, a fresh L2-regularized logistic regression
(`SGDClassifier`, hashed categorical features) is trained on whichever rows
each rule selects, then scored on that day. All methods share the same model
and feature pipeline; they only differ in which historical rows/weights they
train on. See `baselines.py` for the rules and `data.py` for feature
hashing.

Outputs land in `results/`:
- `per_day_metrics.csv` — log loss / Brier / PR-AUC per method per day
- `comparison_table.csv` — aggregated locked-test comparison, including the
  validation-selected window baseline
- `hindsight_best_window.csv` / `.png` — which fixed window would have been
  best on each test day, in hindsight (diagnostic only, never fed back into
  a model)
- `per_day_logloss.png` — per-day log loss for all P0 methods
- `findings.md` — short auto-generated summary of the run

## Run the P1/P2 methods

Run `run_baselines.py` (P0) first — `run_advanced.py` reuses its saved
`results/per_day_metrics.csv` and `results/comparison_table.csv` to build
the combined comparison.

```bash
python run_advanced.py --sample-frac 0.2   # quick preliminary pass
python run_advanced.py                     # full dataset
```

Three methods, each reproducing a specific published mechanism rather than
a heuristic stand-in (PDF 3.5–3.7):

- **`han_arw`** (`han_arw.py`) — Han, Huang & Wang, *Model Assessment and
  Selection under Temporal Distribution Shift*, ICML 2024
  ([arXiv:2402.08672](https://arxiv.org/abs/2402.08672),
  [reference code](https://github.com/eliselyhan/ARW)). Reproduces their
  Algorithm 1 (Goldenshluger–Lepski bias/variance-adaptive windowed mean)
  and Algorithm 3 (single-elimination tournament of Algorithm 2 pairwise
  comparisons) to pick, fresh for every prediction day, among the P0
  window-family candidates (expanding + rolling 1/3/7/14) using only their
  historical per-sample loss trajectories. Unlike `validation_selected`
  (frozen once), the effective window can change every test day.
- **`diff_forgetting`** (`diff_forgetting.py`) — Bennett & Clarkson,
  *Differentiable Forgetting* ([reference
  code](https://github.com/jase-clarkson/pods_2022_icml_ts)). Learns a
  scalar exponential-decay rate η (their "GradExp" weighting,
  `α(τ)=exp(-ητ)`) via the paper's bilevel structure: an inner model fit on
  age-weighted history, an outer objective evaluated on a chronologically
  *later* held-out slice (never a random split) that picks η, then a final
  refit on all permissible history with the learned η. η is optimized by
  bounded derivative-free search (`scipy.optimize.minimize_scalar`) rather
  than the paper's implicit-function-theorem hypergradients, since those
  need a differentiable inner solver that `SGDClassifier` doesn't expose —
  the paper's own `GridSearchExp` ablation is exactly this substitute for a
  single decay parameter.
- **`adamoe`** (`adamoe.py`) — Liu et al., *On the Adaptation to Concept
  Drift for CTR Prediction* (AdaMoE). Reproduces AdaMoE's actual novelty —
  a closed-form, gradient-free EMA update of per-expert aggregation
  weights, based on each expert's per-sample correctness — using the P0
  window-family models as the experts (the paper's own backbone is
  explicitly swappable). Weights for day *t*'s prediction are the EMA as of
  day *t*-1, updated only after *t*'s true labels are observed, so nothing
  leaks.

Additional outputs in `results/`:
- `p1_p2_per_day_metrics.csv`, `all_methods_per_day_metrics.csv`,
  `all_methods_comparison_table.csv`, `all_methods_per_day_logloss.png` —
  P1/P2 combined with the P0 results
- `han_arw_selected_window.csv` — selected window per prediction day
- `diff_forgetting_eta.csv` — learned η / implied half-life per prediction day
- `adamoe_expert_weights.csv` — EMA expert-weight trajectory
- `advanced_memory_behavior.png` — Han ARW's selected window and
  Differentiable Forgetting's learned half-life over time
- `advanced_findings.md` — short auto-generated summary
- [`analysis.md`](results/analysis.md) — hand-written findings note (PDF
  deliverable in section 9): when recency helps, when history helps, and
  why none of the P1/P2 methods clearly beat the simplest P0 baseline on
  this dataset

## Synthetic drift-injection experiments

The real Criteo Attribution dataset (31 days) shows only shallow drift —
every P1/P2 method converges to roughly what a static 7-day window already
gives (see [`results/analysis.md`](results/analysis.md)). Neither the
papers being reproduced here nor the general CTR-benchmark literature
(AdaMoE's own public benchmark, Ali-CCP, Criteo 1TB, Avazu) offer a public,
CTR-native dataset with a genuinely long horizon and documented drift — the
two papers that needed real multi-month/multi-year drift to demonstrate
their own methods (Han et al., Differentiable Forgetting) had to leave the
CTR domain entirely (text-topic trends, real-estate prices, equities) to
find it.

`synthetic_data.py` follows the same workaround: a synthetic CTR generator
with a *known* ground-truth logistic model whose weights evolve over a much
longer horizon according to a chosen schedule, so every method's tracking
behavior can be checked against a true process instead of inferred
indirectly from held-out loss on data whose true process is unknown. Same
`(X, y, day)` interface as `data.load_dataset`, so every P0/P1/P2 method
runs against it unchanged via `--source synthetic`:

```bash
python run_baselines.py --source synthetic --synthetic-days 120 --synthetic-drift abrupt --synthetic-shift-day 95 --out results_synthetic_abrupt
python run_advanced.py  --source synthetic --synthetic-days 120 --synthetic-drift abrupt --synthetic-shift-day 95 --out results_synthetic_abrupt
```

Drift schedules (`--synthetic-drift`):
- `none` — stationary ground truth; sanity check that no method claims a
  spurious advantage when there is nothing to adapt to.
- `abrupt` — a single full regime swap at `--synthetic-shift-day`; tests
  recovery speed right after a sharp change (PDF section 10's "recovers
  slowly after abrupt shifts").
- `gradual` — the true weights interpolate linearly across the whole
  horizon; tests whether recency/decay/window choice keeps pace with
  continuous drift.
- `recurring` — the true weights oscillate with period
  `--synthetic-period-days`; tests whether recency-based adaptation
  (none of these methods model periodicity explicitly) can still track a
  cyclical regime.

Results for a 120-day run of all four modes are in
`results_synthetic_{none,abrupt,gradual,recurring}/`, analyzed in
[`results/synthetic_analysis.md`](results/synthetic_analysis.md). Headline:
`han_arw` decisively beats every static P0 baseline under `abrupt` drift
(tracking the regime change day-by-day, visible in
`han_arw_selected_window.csv`), but under `recurring` (cyclical) drift all
three P1/P2 methods default to "use everything" — recency-based adaptation
has no notion of periodicity, so it can't track a cycle even though it
clearly can track a one-off regime change.

## SFTL (`sftl.py`)

The other P2 alternative (PDF 3.7): Zhu et al.'s *Generalize for Future:
Slow and Fast Trajectory Learning for CTR Prediction* (SFTL), AAAI 2024.
Unlike every other method here, it isn't built on the `SGDClassifier`
window-family infrastructure — it needed a genuine neural model (embedding
+ MLP) and its own continuous streaming-training loop, since its
"trajectory loss" only makes sense as a training-time regularizer on a
differentiable model trained across the whole chronological stream:

```bash
python run_sftl.py --source synthetic --synthetic-days 120 --synthetic-drift abrupt --synthetic-shift-day 95 --epochs-per-domain 5 --out results_synthetic_abrupt
```

Three copies of one model — a working learner (trained every minibatch), a
slow learner (hard-copied from the working learner once per day), and a
fast learner (EMA of the working learner, served at inference) — coupled by
a bipartite-ranking loss that pushes the working learner to exceed the
slow/fast learner's own margin. Reproducing this surfaced a real bug: the
trajectory loss has no natural floor (the slow learner is a hard copy of an
already-more-confident past self every domain), and the paper doesn't
disclose its loss weights — a naive guess of 1.0 caused runaway confidence
escalation (log loss > 4 before evaluation even started). Lowered to 0.05
(found empirically), training merely *looks* stable over a short check —
**a full staged debugging pass (`debug_sftl.py`, following
[`sftl-debugging-plan.pdf`](sftl-debugging-plan.pdf)) found that 0.05 only
delays the same runaway**: logits reach the tens of thousands by day 90 of
the full 120-day run, days before any actual drift, while AUC stays fine
(ranking survives) and log loss is already far worse than every other
method — a clean calibration/confidence-runaway diagnosis, confirmed by
measuring gradient norms directly (the trajectory term's gradient is
3-4.5x larger than BCE's and grows over training, invisible if you only
watch the loss value). **SFTL underperforms every other method in every
drift mode tested**, worst of all under `abrupt` drift — the opposite of
`han_arw`'s behavior. Implementation bug is ruled out (Stage 1 passes
exactly); a properly small lambda is a real, quantified fix direction for
the calibration problem, but even a BCE-only model still trails the simple
window baselines at this benchmark's data scale, so the debugging plan's
own stop criterion applies. Full analysis in
[`results/sftl_analysis.md`](results/sftl_analysis.md) and the complete
staged diagnosis in
[`results/sftl_debug_findings.md`](results/sftl_debug_findings.md).

## Notes / limitations

- Dev/test split is a simple last-N-days holdout, not full rolling-origin
  cross-validation within development.
- No leakage/reproducibility test suite yet (PDF section 8 acceptance
  tests) — the day-based masks are the only leakage guard so far.
- The autobidding stage (PDF section 11) is out of scope until this
  prediction benchmark is frozen.
