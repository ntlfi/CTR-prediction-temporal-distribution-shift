#!/bin/bash
#SBATCH --job-name=ttam-banks-criteo
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=final_experiments/logs/ttam_banks_criteo_%j.out

# Build + cache the shared five-horizon expert banks and context sketches
# for the TTAM nested runner (final_experiments/ttam/build_banks.py).
# One-time; the nested passes then load from _bankcache/ instead of
# refitting. ~8 min per seed on full Criteo.
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p final_experiments/logs
PYTHONPATH=. .venv/bin/python final_experiments/ttam/build_banks.py \
    --source criteo --data data/criteo_attribution_dataset.tsv.gz \
    --cache-dir final_experiments/ttam/_bankcache --n-jobs 8
