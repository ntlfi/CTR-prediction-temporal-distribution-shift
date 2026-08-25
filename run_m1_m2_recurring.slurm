#!/bin/bash
#SBATCH --job-name=ctr-m1m2-recurring
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

cd "$SLURM_SUBMIT_DIR"
.venv/bin/python run_new_methods.py --source synthetic --synthetic-days 120 --synthetic-rows-per-day 3000 \
    --synthetic-drift recurring --synthetic-period-days 14 --n-jobs 2 --out results_synthetic_recurring
