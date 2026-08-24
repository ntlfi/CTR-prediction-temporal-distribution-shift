#!/bin/bash
set -e
source .venv/bin/activate

N_DAYS=120
ROWS=3000

run_mode () {
  MODE=$1
  OUT=results_synthetic_${MODE}
  shift
  echo "=== sftl: $MODE -> $OUT ==="
  python run_sftl.py --source synthetic --synthetic-days $N_DAYS --synthetic-rows-per-day $ROWS \
    --synthetic-drift $MODE --epochs-per-domain 5 --batch-size 256 "$@" --out $OUT
}

run_mode none
run_mode abrupt --synthetic-shift-day 95
run_mode gradual
run_mode recurring --synthetic-period-days 14

echo "ALL DONE"
