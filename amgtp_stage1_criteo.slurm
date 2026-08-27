#!/bin/bash
#SBATCH --job-name=amgtp-s1-criteo
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --array=0-2
#SBATCH --output=amgtp_experiments/logs/s1_criteo_%A_%a.out

# AMG-TP plan Stage 1: real Criteo Attribution chronological benchmark
# (robustness / no-downside test -- natural drift is shallow). 3 seeds.
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p amgtp_experiments/logs
PY=.venv/bin/python

SEEDS=(0 1 2)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=amgtp_experiments/stage1_m5b_high_smooth/criteo/seed${SEED}

$PY amgtp_run.py --source criteo \
    --data /insomnia001/home/tn2447/data/criteo/criteo_attribution_dataset.tsv.gz \
    --sample-frac 1.0 --seed "$SEED" --n-jobs 8 --out "$OUT"
