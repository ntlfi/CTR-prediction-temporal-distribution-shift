#!/bin/bash
#SBATCH --job-name=apops-nested-criteo
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --output=final_experiments/logs/apops_nested_criteo_%j.out

# AP-OPS fully-nested rolling-origin (2026-09-06 additional-experiments
# spec). 4 headline methods (ops / reset_ons / persistent_ons / ap_ops) +
# no-slope ablation; per-origin common-config selection across seeds 0/1/2
# on trailing-3-day inner validation; lambda_AP grid contains 0. Two
# seed-loops so full-data Avazu (~40M rows) never holds >1 seed's bank.
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p final_experiments/logs
PYTHONPATH=. .venv/bin/python final_experiments/run_apops_nested.py --source criteo \
    --sample-frac 1.0 --n-jobs 8 \
    --config final_experiments/criteo/apops/apops_selected.json \
    --out final_experiments/criteo/apops/nested
