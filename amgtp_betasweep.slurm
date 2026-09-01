#!/bin/bash
#SBATCH --job-name=amgtp-betasweep
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-135%40
#SBATCH --output=amgtp_experiments/logs/betasweep_%A_%a.out

# Package 1: dense fixed-beta grid (0..1 step 0.05) x S0-S7 x 17 seeds.
# 136 cells. After:
#   .venv/bin/python amgtp_betasweep.py --aggregate --out amgtp_experiments/stage4_betasweep
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p amgtp_experiments/logs
.venv/bin/python amgtp_betasweep.py --cell "$SLURM_ARRAY_TASK_ID" --n-jobs 4 \
    --out amgtp_experiments/stage4_betasweep
