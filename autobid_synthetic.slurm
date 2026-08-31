#!/bin/bash
#SBATCH --job-name=autobid-synth
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-55%32
#SBATCH --output=amgtp_experiments/logs/autobid_synth_%A_%a.out

# AMG-TP plan step 8: downstream autobidding eval on the synthetic shift suite
# S0-S6 with a synthetic second-price cost landscape (autobid.synthetic_cost).
# Tests whether the synthetic CTR-prediction wins (AMG-TP vs Han ARW on
# S1/S3/S4) translate into value at matched spend under a frozen bidder.
# 7 regimes x 8 seeds = 56 cells.
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p amgtp_experiments/logs
PY=.venv/bin/python

CELL=$($PY amgtp_config.py autobid-cell "$SLURM_ARRAY_TASK_ID")
REGIME=$(echo "$CELL" | cut -f1)
SEED=$(echo "$CELL" | cut -f2)
OUT=$(echo "$CELL" | cut -f3)
RUN_ARGS=$(echo "$CELL" | cut -f4)

echo "cell $SLURM_ARRAY_TASK_ID -> regime=$REGIME seed=$SEED out=$OUT"
$PY run_autobid.py $RUN_ARGS --seed "$SEED" --n-jobs 4 --out "$OUT"
