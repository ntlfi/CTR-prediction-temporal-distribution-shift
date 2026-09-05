#!/bin/bash
#SBATCH --job-name=dualtime-hpo-criteo
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --output=final_experiments/logs/hpo_criteo_%j.out

# DualTime-CTR final experiment -- Criteo HPO (sections 6-10). Full data;
# each seed's dataset is loaded and dropped sequentially (twice -- once
# per HPO phase) rather than holding all 3 seeds simultaneously, since
# this cluster's nodes cap at ~250G and section 6 needs full-data Avazu
# (~40M rows) x 3 seeds to fit too.
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p final_experiments/logs
PYTHONPATH=. .venv/bin/python final_experiments/run_hpo.py --source criteo --sample-frac 1.0 \
    --n-jobs 8 --out final_experiments/criteo/hpo
