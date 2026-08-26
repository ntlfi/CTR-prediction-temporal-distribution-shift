#!/bin/bash
#SBATCH --job-name=ctr-m5-synthetic
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=slurm-%j.out

cd "$SLURM_SUBMIT_DIR"
set -e
N_DAYS=120
ROWS=3000

.venv/bin/python run_new_methods.py --source synthetic --synthetic-days $N_DAYS --synthetic-rows-per-day $ROWS \
    --synthetic-drift none --n-jobs 4 --out results_synthetic_none
.venv/bin/python run_new_methods.py --source synthetic --synthetic-days $N_DAYS --synthetic-rows-per-day $ROWS \
    --synthetic-drift abrupt --synthetic-shift-day 95 --n-jobs 4 --out results_synthetic_abrupt
.venv/bin/python run_new_methods.py --source synthetic --synthetic-days $N_DAYS --synthetic-rows-per-day $ROWS \
    --synthetic-drift gradual --n-jobs 4 --out results_synthetic_gradual
.venv/bin/python run_new_methods.py --source synthetic --synthetic-days $N_DAYS --synthetic-rows-per-day $ROWS \
    --synthetic-drift recurring --synthetic-period-days 14 --n-jobs 4 --out results_synthetic_recurring
.venv/bin/python run_new_methods.py --source synthetic --synthetic-days $N_DAYS --synthetic-rows-per-day $ROWS \
    --synthetic-drift local --synthetic-shift-day 95 --n-jobs 4 --out results_synthetic_local
