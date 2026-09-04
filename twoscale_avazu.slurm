#!/bin/bash
#SBATCH --job-name=twoscale-avazu
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --array=0-7
#SBATCH --output=twoscale_experiments/logs/avazu_%A_%a.out

# Two-timescale CTR forecasting -- second real dataset (Avazu, hour-resolution
# within-day arrival times, plan section 11 step 10). 8 seeds, each drawing a
# disjoint 20% row subsample so seeds are a genuine source of variation.
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p twoscale_experiments/logs
PY=.venv/bin/python

SEED=$SLURM_ARRAY_TASK_ID
OUT=twoscale_experiments/avazu/seed${SEED}
DATA=/insomnia001/home/tn2447/data/avazu/Avazu_x4.zip

$PY twoscale_hpo.py --source avazu --data "$DATA" --sample-frac 0.2 \
    --seed "$SEED" --warmup 3 --n-jobs 8 --out "$OUT/_hpo"

$PY twoscale_run.py --source avazu --data "$DATA" --sample-frac 0.2 \
    --seed "$SEED" --warmup 3 --n-jobs 8 --config "$OUT/_hpo/FROZEN.json" --out "$OUT"
