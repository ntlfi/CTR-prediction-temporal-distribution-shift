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
# Attribution, real second-resolution within-day arrival times). Per seed:
# pick per-variant hyperparameters on the dev days (withinday_hpo.py),
# freeze, then train V1-V5 plus their 5 required ablations at that config,
# apply the plan's decision rules, and open the locked test only if a
# candidate clears them.
#
# Full 3-seed protocol (matching twoscale) -- the 1-seed HPO-tuned pass
# found v5_linear clearing every decision-rule gate and beating both
# long_only and online_platt significantly on the locked test (though
# below the materiality floor), so this checks stability across seeds
# (decision rule item 2).
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p withinday_experiments/logs
PY=.venv/bin/python

SEEDS=(0 1 2)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=withinday_experiments/criteo/seed${SEED}
DATA=/insomnia001/home/tn2447/data/criteo/criteo_attribution_dataset.tsv.gz

$PY withinday_hpo.py --source criteo --data "$DATA" --sample-frac 1.0 \
    --seed "$SEED" --n-jobs 8 --out "$OUT/_hpo"

$PY withinday_run.py --source criteo --data "$DATA" --sample-frac 1.0 \
    --seed "$SEED" --n-jobs 8 --config "$OUT/_hpo/FROZEN.json" --out "$OUT"
