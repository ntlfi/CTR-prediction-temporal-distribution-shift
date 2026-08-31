#!/bin/bash
#SBATCH --job-name=autobid-criteo
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --output=amgtp_experiments/logs/autobid_criteo_%A_%a.out

# AMG-TP plan step 8: downstream autobidding eval on the real Criteo
# Attribution log (recorded display cost + conversion). Natural drift is
# shallow -- this is the PDF section 8 real-data test / no-downside check.
# 5 seeds (full dataset, no subsample).
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p amgtp_experiments/logs
PY=.venv/bin/python

SEED=$SLURM_ARRAY_TASK_ID
$PY run_autobid.py --source criteo \
    --data /insomnia001/home/tn2447/data/criteo/criteo_attribution_dataset.tsv.gz \
    --sample-frac 1.0 --seed "$SEED" --n-jobs 8 \
    --out amgtp_experiments/stage3_autobid/criteo/seed${SEED}
