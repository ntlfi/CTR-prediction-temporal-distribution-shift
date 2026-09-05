#!/bin/bash
#SBATCH --job-name=withinday-rolling-criteo
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --output=withinday_experiments/logs/rolling_criteo_%j.out

# Day-level rolling-origin evaluation, Criteo (rolling-protocol sections
# 2-3 applied per ROLLING_PROTOCOL_FREEZE.md's "no untouched days" branch:
# post-selection rolling-origin analysis over the 6 original dev days,
# 16-21, NOT a second locked holdout -- the original 9-day locked test
# stays primary and untouched). Full data, one fixed data-sampling seed
# (frozen at 0); a small model-seed sensitivity check is a separate run.
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p withinday_experiments/logs
.venv/bin/python withinday_rolling_run.py --source criteo --sample-frac 1.0 \
    --data-seed 0 --model-seed 0 --n-jobs 8 \
    --out withinday_experiments/rolling/criteo
