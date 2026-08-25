#!/bin/bash
#SBATCH --job-name=ctr-local-pipeline
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

cd "$SLURM_SUBMIT_DIR"
set -e
.venv/bin/python run_baselines.py --source synthetic --synthetic-days 120 --synthetic-rows-per-day 3000 \
    --synthetic-drift local --synthetic-shift-day 95 --out results_synthetic_local
.venv/bin/python run_advanced.py --source synthetic --synthetic-days 120 --synthetic-rows-per-day 3000 \
    --synthetic-drift local --synthetic-shift-day 95 --n-jobs 4 --out results_synthetic_local
.venv/bin/python run_new_methods.py --source synthetic --synthetic-days 120 --synthetic-rows-per-day 3000 \
    --synthetic-drift local --synthetic-shift-day 95 --n-jobs 4 --out results_synthetic_local
