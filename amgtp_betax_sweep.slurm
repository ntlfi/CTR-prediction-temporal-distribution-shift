#!/bin/bash
#SBATCH --job-name=amgtp-bx-sweep
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-24%15
#SBATCH --output=amgtp_experiments/logs/bx_sweep_%A_%a.out

# Extension B, step 1: freeze the per-example beta_t(x) variance penalty on the
# dev seeds. 5 regimes (S0/S3/S4/S5/S7) x 5 dev seeds = 25 cells. After:
#   .venv/bin/python amgtp_betax_sweep.py --aggregate --out amgtp_experiments/stage4_betax/_sweep
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p amgtp_experiments/logs
.venv/bin/python amgtp_betax_sweep.py --cell "$SLURM_ARRAY_TASK_ID" --n-jobs 4 \
    --out amgtp_experiments/stage4_betax/_sweep
