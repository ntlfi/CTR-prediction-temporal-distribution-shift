#!/bin/bash
#SBATCH --job-name=ctr-m5-smoothreg-sweep2
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

cd "$SLURM_SUBMIT_DIR"
set -e
# Extend the recurring grid higher to find where the trend turns over.
.venv/bin/python sweep_m5_smooth_reg.py --source synthetic --synthetic-days 120 --synthetic-rows-per-day 3000 \
    --synthetic-drift recurring --synthetic-period-days 14 \
    --smooth-reg-grid 0.1 0.2 0.3 0.5 1.0 2.0 5.0 \
    --n-jobs 4 --out results_synthetic_recurring

# No-downside check: does smooth_reg=0.1 (the best recurring value so far)
# hurt the regimes where M5b currently wins outright?
N_DAYS=120
ROWS=3000
.venv/bin/python sweep_m5_smooth_reg.py --source synthetic --synthetic-days $N_DAYS --synthetic-rows-per-day $ROWS \
    --synthetic-drift abrupt --synthetic-shift-day 95 \
    --smooth-reg-grid 0.001 0.1 0.3 --n-jobs 4 --out results_synthetic_abrupt
.venv/bin/python sweep_m5_smooth_reg.py --source synthetic --synthetic-days $N_DAYS --synthetic-rows-per-day $ROWS \
    --synthetic-drift gradual \
    --smooth-reg-grid 0.001 0.1 0.3 --n-jobs 4 --out results_synthetic_gradual
.venv/bin/python sweep_m5_smooth_reg.py --source synthetic --synthetic-days $N_DAYS --synthetic-rows-per-day $ROWS \
    --synthetic-drift local --synthetic-shift-day 95 \
    --smooth-reg-grid 0.001 0.1 0.3 --n-jobs 4 --out results_synthetic_local
.venv/bin/python sweep_m5_smooth_reg.py --source synthetic --synthetic-days $N_DAYS --synthetic-rows-per-day $ROWS \
    --synthetic-drift none \
    --smooth-reg-grid 0.001 0.1 0.3 --n-jobs 4 --out results_synthetic_none
