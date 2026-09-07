#!/bin/bash
#SBATCH --job-name=apops-rolling-avazu
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=230G
#SBATCH --time=12:00:00
#SBATCH --output=final_experiments/logs/apops_rolling_avazu_%j.out

# AP-OPS rolling-origin confirmation (primary temporal-adaptation
# evidence). Frozen apops_selected.json, no per-origin reselection.
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p final_experiments/logs
PYTHONPATH=. .venv/bin/python final_experiments/run_apops_rolling.py --source avazu \
    --sample-frac 1.0 --warmup 3 --n-jobs 8 \
    --config final_experiments/avazu/apops/apops_selected.json \
    --out final_experiments/avazu/apops/rolling
