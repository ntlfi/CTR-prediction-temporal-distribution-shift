#!/bin/bash
#SBATCH --job-name=ctr-local-m1m2-fix
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out

cd "$SLURM_SUBMIT_DIR"
set -e
# P0/P1/P2 for results_synthetic_local already exist from the earlier run --
# only re-run M1/M2 (now with the fixed context-carrying gate) and remerge.
.venv/bin/python run_new_methods.py --source synthetic --synthetic-days 120 --synthetic-rows-per-day 3000 \
    --synthetic-drift local --synthetic-shift-day 95 --n-jobs 4 --out results_synthetic_local
