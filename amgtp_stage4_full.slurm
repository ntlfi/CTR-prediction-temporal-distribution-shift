#!/bin/bash
#SBATCH --job-name=amgtp-s4-full
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-237%48
#SBATCH --output=amgtp_experiments/logs/s4_full_%A_%a.out

# Package 2 + 3: full method battery over ALL 14 synthetic regimes
# (S0-S7 + off-grid recurring S3b-e + irregular S8 + heterogeneous S9)
# x 17 seeds, now including Fixed Share / Learn-alpha as classical
# expert-tracking baselines. --stage2 = AMG-TP + ablations.
# synth_grid indices: 0-135 = the original 8 regimes, 136-237 = the 6 new.
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
