#!/bin/bash
#SBATCH --job-name=amgtp-s1-synth
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --array=0-118%40
#SBATCH --output=amgtp_experiments/logs/s1_synth_%A_%a.out

# AMG-TP plan Stage 1: M5b-high-smooth vs the baseline suite over the
# synthetic shift suite S0-S6, dev + confirmation seeds. One array task per
# (regime, seed) cell (7 regimes x 17 seeds = 119).
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p amgtp_experiments/logs
PY=.venv/bin/python

CELL=$($PY amgtp_config.py cell "$SLURM_ARRAY_TASK_ID" amgtp_experiments/stage1_m5b_high_smooth)
REGIME=$(echo "$CELL" | cut -f1)
SEED=$(echo "$CELL" | cut -f2)
OUT=$(echo "$CELL" | cut -f3)
RUN_ARGS=$(echo "$CELL" | cut -f4)

echo "cell $SLURM_ARRAY_TASK_ID -> regime=$REGIME seed=$SEED out=$OUT"
$PY amgtp_run.py $RUN_ARGS --seed "$SEED" --n-jobs 4 --out "$OUT"
