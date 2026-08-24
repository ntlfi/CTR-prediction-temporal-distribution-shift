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

## Run the baselines

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

## Notes / limitations of this first pass

This is a preliminary baseline comparison, not the full protocol in the PDF:
- Dev/test split is a simple last-N-days holdout, not full rolling-origin
  cross-validation within development.
- Only the P0 baselines are implemented (expanding ERM, rolling windows,
  exponential forgetting, validation-selected window). Han et al. adaptive
  rolling window, Differentiable Forgetting, and the CTR-specific P2 model
  (AdaMoE/SFTL) are P1/P2 and not built yet.
- No leakage/reproducibility test suite yet (PDF section 8 acceptance
  tests) — the day-based masks are the only leakage guard so far.
