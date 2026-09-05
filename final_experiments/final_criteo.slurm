#!/bin/bash
#SBATCH --job-name=dualtime-final-criteo
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --output=final_experiments/logs/final_criteo_%j.out

# DualTime-CTR final experiment -- Criteo primary 3-seed locked test
# (section 12), using the frozen final_experiments/criteo/hpo/selected_configs.json.
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p final_experiments/logs
PYTHONPATH=. .venv/bin/python final_experiments/run_final.py --source criteo --sample-frac 1.0 \
    --n-jobs 8 --config final_experiments/criteo/hpo/selected_configs.json \
    --out final_experiments/criteo/final
