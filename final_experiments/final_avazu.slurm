#!/bin/bash
#SBATCH --job-name=dualtime-final-avazu
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=230G
#SBATCH --time=12:00:00
#SBATCH --output=final_experiments/logs/final_avazu_%j.out

# DualTime-CTR final experiment -- Avazu primary 3-seed locked test
# (section 12), full data (sample_frac=1.0), using the frozen
# final_experiments/avazu/hpo/selected_configs.json. Do not submit before
# that file exists (hpo_avazu.slurm must finish first).
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p final_experiments/logs
PYTHONPATH=. .venv/bin/python final_experiments/run_final.py --source avazu --sample-frac 1.0 \
    --warmup 3 --n-jobs 8 --config final_experiments/avazu/hpo/selected_configs.json \
    --out final_experiments/avazu/final
