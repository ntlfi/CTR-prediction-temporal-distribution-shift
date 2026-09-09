#!/bin/bash
#SBATCH --job-name=ttam-nested-avazu
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=230G
#SBATCH --time=12:00:00
#SBATCH --output=final_experiments/logs/ttam_nested_avazu_%j.out

# TTAM revised Section 6 -- nested rolling-origin run on full-data Avazu
# (~40M rows), 9 variants, origins 5-9, seeds 0/1/2. Avazu is descriptive
# only (D = 5 origins). Bidding is left pending (no recorded price field).
# Run ttam_banks_avazu.slurm first. Each Avazu context sketch is large;
# keep --n-workers modest so N * (bank + ctx) stays inside --mem.
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p final_experiments/logs
PYTHONPATH=. .venv/bin/python final_experiments/run_ttam_nested.py \
    --source avazu --data data/avazu/Avazu_x4.zip \
    --sample-frac 1.0 --n-features 262144 \
    --n-workers 5 --n-jobs 8 \
    --cache-dir final_experiments/ttam/_bankcache \
    --out final_experiments/ttam/avazu/nested

# refresh the Section 6 stats + figure with the Avazu panel added
PYTHONPATH=. bash final_experiments/run_ttam_section6.sh
