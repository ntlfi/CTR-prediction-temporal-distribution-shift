#!/bin/bash
#SBATCH --job-name=ctr-m5-smoothreg-sweep
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out

cd "$SLURM_SUBMIT_DIR"
set -e
.venv/bin/python sweep_m5_smooth_reg.py --source synthetic --synthetic-days 120 --synthetic-rows-per-day 3000 \
    --synthetic-drift recurring --synthetic-period-days 14 \
    --smooth-reg-grid 0.0 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 \
    --n-jobs 4 --out results_synthetic_recurring
