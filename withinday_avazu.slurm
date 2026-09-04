#!/bin/bash
#SBATCH --job-name=withinday-avazu
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --array=0-0
#SBATCH --output=withinday_experiments/logs/avazu_%A_%a.out

# Within-day capacity-ladder adapters -- second real dataset (Avazu,
# hour-resolution within-day arrival times). Single seed for a fast first
# read (twoscale used 8, each a disjoint 20% subsample); widen the array
# range back to 0-7 for the full multi-seed protocol once this looks worth
# confirming.
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p withinday_experiments/logs
PY=.venv/bin/python

SEED=$SLURM_ARRAY_TASK_ID
OUT=withinday_experiments/avazu/seed${SEED}
DATA=/insomnia001/home/tn2447/data/avazu/Avazu_x4.zip

$PY withinday_run.py --source avazu --data "$DATA" --sample-frac 0.2 \
    --seed "$SEED" --warmup 3 --n-jobs 8 --out "$OUT"
