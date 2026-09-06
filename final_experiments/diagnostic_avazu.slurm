#!/bin/bash
#SBATCH --job-name=dualtime-diag-avazu
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=230G
#SBATCH --time=12:00:00
#SBATCH --output=final_experiments/logs/diagnostic_avazu_%j.out

# Review comment 3: frozen V5 vs online DualTime diagnostic, exact final
# protocol (full Avazu, 3 seeds, same frozen q). DEVELOPMENT EVIDENCE
# ONLY -- test days already inspected.
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p final_experiments/logs
PYTHONPATH=. .venv/bin/python final_experiments/run_diagnostic.py --source avazu \
    --sample-frac 1.0 --warmup 3 --n-jobs 8 \
    --config final_experiments/avazu/hpo/selected_configs.json \
    --out final_experiments/avazu/diagnostic
