#!/bin/bash
#SBATCH --job-name=ctr-recurring-seed
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:25:00
#SBATCH --array=1-4
#SBATCH --output=slurm-%A_%a.out

cd "$SLURM_SUBMIT_DIR"
set -e
SEED=$SLURM_ARRAY_TASK_ID
OUT=results_synthetic_recurring_seed${SEED}
.venv/bin/python run_baselines.py --source synthetic --synthetic-days 120 --synthetic-rows-per-day 3000 \
    --synthetic-drift recurring --synthetic-period-days 14 --seed $SEED --out $OUT
.venv/bin/python run_advanced.py --source synthetic --synthetic-days 120 --synthetic-rows-per-day 3000 \
    --synthetic-drift recurring --synthetic-period-days 14 --seed $SEED --n-jobs 4 --out $OUT
.venv/bin/python run_new_methods.py --source synthetic --synthetic-days 120 --synthetic-rows-per-day 3000 \
    --synthetic-drift recurring --synthetic-period-days 14 --seed $SEED --n-jobs 4 --out $OUT
