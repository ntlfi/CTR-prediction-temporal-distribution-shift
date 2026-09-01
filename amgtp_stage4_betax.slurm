#!/bin/bash
#SBATCH --job-name=amgtp-s4-bx
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --array=119-135%20
#SBATCH --output=amgtp_experiments/logs/s4_bx_%A_%a.out

# Extension B confirmation: rerun S7 opposing_recurring (indices 119-135 =
# s7 x {dev 0-4, confirm 20-31}) with --beta-x, so amgtp_bx / A11-A13 land on
# the disjoint confirmation seeds. The dev sweep
# (amgtp_experiments/stage4_betax/_sweep/FROZEN.md) already found per-example
# beta_t(x) never beats the global beta_t; this confirms it out-of-sample.
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p amgtp_experiments/logs
PY=.venv/bin/python

CELL=$($PY amgtp_config.py cell "$SLURM_ARRAY_TASK_ID" amgtp_experiments/stage2_amgtp)
REGIME=$(echo "$CELL" | cut -f1)
SEED=$(echo "$CELL" | cut -f2)
OUT=$(echo "$CELL" | cut -f3)
RUN_ARGS=$(echo "$CELL" | cut -f4)

echo "cell $SLURM_ARRAY_TASK_ID -> regime=$REGIME seed=$SEED out=$OUT"
$PY amgtp_run.py $RUN_ARGS --seed "$SEED" --n-jobs 4 --beta-x --beta-var-reg 1e-3 --out "$OUT"
