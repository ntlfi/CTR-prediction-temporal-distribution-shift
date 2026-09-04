#!/bin/bash
#SBATCH --job-name=withinday-criteo
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=withinday_experiments/logs/criteo_%A_%a.out

# Within-day capacity-ladder adapters -- primary real dataset (Criteo
# Attribution, real second-resolution within-day arrival times). Trains
# V1-V5 plus their 5 required ablations each, applies the plan's decision
# rules, and opens the locked test only if a candidate clears them.
#
# Uses the plan's fixed default hyperparameters (withinday.train.DEFAULT_CFG)
# rather than a tuned grid -- there is no withinday_hpo.py yet (see
# withinday_experiments/README.md).
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p withinday_experiments/logs
PY=.venv/bin/python

SEEDS=(0 1 2)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=withinday_experiments/criteo/seed${SEED}
DATA=/insomnia001/home/tn2447/data/criteo/criteo_attribution_dataset.tsv.gz

$PY withinday_run.py --source criteo --data "$DATA" --sample-frac 1.0 \
    --seed "$SEED" --n-jobs 8 --out "$OUT"
