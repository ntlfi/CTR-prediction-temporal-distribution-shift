#!/bin/bash
#SBATCH --job-name=twoscale-criteo
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=10:00:00
#SBATCH --array=0-2
#SBATCH --output=twoscale_experiments/logs/criteo_%A_%a.out

# Two-timescale CTR forecasting -- primary real dataset (Criteo Attribution,
# real second-resolution within-day arrival times). Per seed: pick the
# calibrator hyperparameters on the dev days (twoscale_hpo.py), freeze, then
# run the full locked-test protocol (twoscale_run.py).
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p twoscale_experiments/logs
PY=.venv/bin/python

SEEDS=(0 1 2)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=twoscale_experiments/criteo/seed${SEED}
DATA=/insomnia001/home/tn2447/data/criteo/criteo_attribution_dataset.tsv.gz

$PY twoscale_hpo.py --source criteo --data "$DATA" --sample-frac 1.0 \
    --seed "$SEED" --n-jobs 8 --out "$OUT/_hpo"

$PY twoscale_run.py --source criteo --data "$DATA" --sample-frac 1.0 \
    --seed "$SEED" --n-jobs 8 --config "$OUT/_hpo/FROZEN.json" --out "$OUT"
