#!/bin/bash
#SBATCH --job-name=ttam-banks-avazu
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=230G
#SBATCH --time=06:00:00
#SBATCH --output=final_experiments/logs/ttam_banks_avazu_%j.out

# Build + cache the shared five-horizon expert banks and context sketches
# for the TTAM nested runner on full-data Avazu (~40M rows). The seed loop
# in build_banks.py holds one seed's raw hashed matrix at a time; still
# large -- match the memory the AP-OPS Avazu jobs used (230G).
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p final_experiments/logs
PYTHONPATH=. .venv/bin/python final_experiments/ttam/build_banks.py \
    --source avazu --data data/avazu/Avazu_x4.zip \
    --cache-dir final_experiments/ttam/_bankcache --n-jobs 8
