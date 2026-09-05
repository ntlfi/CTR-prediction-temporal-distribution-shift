#!/bin/bash
#SBATCH --job-name=withinday-seedsens-avazu
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=withinday_experiments/logs/seedsens_avazu_%j.out
cd "$SLURM_SUBMIT_DIR"
set -e
.venv/bin/python withinday_rolling_seed_sensitivity.py --source avazu \
    --manifest withinday_experiments/rolling/avazu/rolling_origin_manifest.csv \
    --sample-frac 0.2 --data-seed 0 --model-seeds 0 1 2 --n-jobs 8 \
    --out withinday_experiments/rolling/avazu/model_seed_sensitivity.csv
