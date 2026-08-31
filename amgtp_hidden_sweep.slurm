#!/bin/bash
#SBATCH --job-name=amgtp-hid-sweep
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-29%15
#SBATCH --output=amgtp_experiments/logs/hid_sweep_%A_%a.out

# Extension A, step 1: freeze the PersistenceNet hidden width on the dev seeds.
# 6 regimes (S0/S1/S3/S4/S5/S7) x 5 dev seeds = 30 cells. After the array
# finishes:  .venv/bin/python amgtp_hidden_sweep.py --aggregate --out amgtp_experiments/stage4_hidden/_sweep
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p amgtp_experiments/logs
.venv/bin/python amgtp_hidden_sweep.py --cell "$SLURM_ARRAY_TASK_ID" --n-jobs 4 \
    --out amgtp_experiments/stage4_hidden/_sweep
