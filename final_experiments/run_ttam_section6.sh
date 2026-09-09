#!/bin/bash
# TTAM revised Section 6 -- run the downstream analysis over a finished nested
# run (or two). Idempotent; re-run after adding the Avazu directory.
#
#   final_experiments/run_ttam_section6.sh
#
# Expects (from run_ttam_nested.py):
#   final_experiments/ttam/criteo/nested/seed{0,1,2}/per_day_metrics.csv
#   final_experiments/ttam/criteo/nested/final_predictions/origin*_seed*.npz
# and, once built, the same under final_experiments/ttam/avazu/nested/.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
# .venv/bin/python on the cluster (matches the other final_experiments jobs);
# plain python3 locally.
PY=${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}

CRITEO=final_experiments/ttam/criteo/nested
AVAZU=final_experiments/ttam/avazu/nested

DIRS=(--dir "$CRITEO" --label Criteo)
[ -d "$AVAZU/seed0" ] && DIRS+=(--dir "$AVAZU" --label Avazu)

echo "== paired day-level statistics =="
$PY final_experiments/ttam_stats.py "${DIRS[@]}" \
    --out final_experiments/TTAM_SECTION6_STATS.md \
    --json-out final_experiments/ttam/section6_stats.json

echo "== Section 6.2 figure =="
$PY final_experiments/ttam_figure.py "${DIRS[@]}" \
    --out final_experiments/ttam/section6_figure.png

echo "== Criteo bidding replay =="
$PY final_experiments/run_ttam_bidding.py \
    --nested-out "$CRITEO" \
    --data data/criteo_attribution_dataset.tsv.gz \
    --out final_experiments/ttam/criteo/bidding

echo
echo "wrote:"
echo "  final_experiments/TTAM_SECTION6_STATS.md"
echo "  final_experiments/ttam/section6_figure.png"
echo "  final_experiments/ttam/criteo/bidding/{summary.json,bidding_matched_*pct.csv}"
