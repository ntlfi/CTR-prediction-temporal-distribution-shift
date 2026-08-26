#!/bin/bash
#SBATCH --job-name=ctr-m5c-synthetic
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

# run_new_methods.py must already have been run in each --out dir (M1/M2/M5b/
# ensemble comparison) for run_m5c.py's merge to include the full ladder --
# it already has been for all 5 results_synthetic_* dirs in this repo.
.venv/bin/python run_m5c.py --source synthetic --synthetic-days $N_DAYS --synthetic-rows-per-day $ROWS \
    --synthetic-drift none --n-jobs 4 --out results_synthetic_none
.venv/bin/python run_m5c.py --source synthetic --synthetic-days $N_DAYS --synthetic-rows-per-day $ROWS \
    --synthetic-drift abrupt --synthetic-shift-day 95 --n-jobs 4 --out results_synthetic_abrupt
.venv/bin/python run_m5c.py --source synthetic --synthetic-days $N_DAYS --synthetic-rows-per-day $ROWS \
    --synthetic-drift gradual --n-jobs 4 --out results_synthetic_gradual
.venv/bin/python run_m5c.py --source synthetic --synthetic-days $N_DAYS --synthetic-rows-per-day $ROWS \
    --synthetic-drift recurring --synthetic-period-days 14 --n-jobs 4 --out results_synthetic_recurring
.venv/bin/python run_m5c.py --source synthetic --synthetic-days $N_DAYS --synthetic-rows-per-day $ROWS \
    --synthetic-drift local --synthetic-shift-day 95 --n-jobs 4 --out results_synthetic_local
