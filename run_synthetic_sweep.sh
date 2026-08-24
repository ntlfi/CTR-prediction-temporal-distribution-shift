#!/bin/bash
set -e
source .venv/bin/activate

N_DAYS=120
ROWS=3000

run_mode () {
  MODE=$1
  OUT=results_synthetic_${MODE}
  shift
  echo "=== $MODE -> $OUT ==="
  python run_baselines.py --source synthetic --synthetic-days $N_DAYS --synthetic-rows-per-day $ROWS \
    --synthetic-drift $MODE "$@" --out $OUT
  python run_advanced.py --source synthetic --synthetic-days $N_DAYS --synthetic-rows-per-day $ROWS \
    --synthetic-drift $MODE "$@" --n-jobs 2 --out $OUT
}

run_mode none
run_mode abrupt --synthetic-shift-day 95
run_mode gradual
run_mode recurring --synthetic-period-days 14

echo "ALL DONE"
