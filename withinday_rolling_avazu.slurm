#!/bin/bash
#SBATCH --job-name=withinday-rolling-avazu
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=withinday_experiments/logs/rolling_avazu_%j.out

# Day-level rolling-origin evaluation, Avazu (rolling-protocol section 2):
# 5 outer test days (5-9), the original dev+test boundary dissolved into
# one continuous walk-forward sequence. One fixed data-sampling seed
# (frozen at 0, same 20% subsample as the original seed0 run).
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p withinday_experiments/logs
.venv/bin/python withinday_rolling_run.py --source avazu --sample-frac 0.2 \
    --data-seed 0 --model-seed 0 --n-jobs 8 \
    --out withinday_experiments/rolling/avazu
