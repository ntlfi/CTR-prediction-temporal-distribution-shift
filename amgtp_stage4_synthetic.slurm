#!/bin/bash
#SBATCH --job-name=amgtp-s4-s7
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --array=119-135%20
#SBATCH --output=amgtp_experiments/logs/s4_s7_%A_%a.out

# Stage 4: the new S7 opposing_recurring regime (per-example-persistence test
# bed) for the full method suite + AMG-TP + ablations, over dev + confirmation
# seeds. Also carries the A10 hidden-persistence-net ablation (amgtp_hidden8/16)
# -- the dev sweep (amgtp_experiments/stage4_hidden/_sweep/FROZEN.md) already
# found no hidden layer helps; this confirms it on the disjoint seeds for S7.
#
# synth_grid() indices 119-135 are exactly the 17 (s7_opposing_recurring, seed)
# cells. Cells land in stage2_amgtp/ so amgtp_aggregate.py produces one unified
# S0-S7 report.
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p amgtp_experiments/logs
PY=.venv/bin/python

CELL=$($PY amgtp_config.py cell "$SLURM_ARRAY_TASK_ID" amgtp_experiments/stage2_amgtp)
REGIME=$(echo "$CELL" | cut -f1)
SEED=$(echo "$CELL" | cut -f2)
OUT=$(echo "$CELL" | cut -f3)
RUN_ARGS=$(echo "$CELL" | cut -f4)

echo "cell $SLURM_ARRAY_TASK_ID -> regime=$REGIME seed=$SEED out=$OUT"
$PY amgtp_run.py $RUN_ARGS --seed "$SEED" --n-jobs 4 --stage2 --out "$OUT"
