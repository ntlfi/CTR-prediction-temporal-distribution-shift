#!/bin/bash
#SBATCH --job-name=dualtime-rolling-criteo
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --output=final_experiments/logs/rolling_criteo_%j.out

# Review comment 1: 6-method rolling-origin confirmation, Criteo outer
# days 16-30 (15 origins), 3 seeds, per-origin re-selection of the cheap
# knobs. Feeds final_experiments/day_level_stats.py.
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p final_experiments/logs
PYTHONPATH=. .venv/bin/python final_experiments/run_rolling.py --source criteo \
    --sample-frac 1.0 --n-jobs 8 \
    --config final_experiments/criteo/hpo/selected_configs.json \
    --out final_experiments/criteo/rolling
