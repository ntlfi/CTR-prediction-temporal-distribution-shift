#!/bin/bash
#SBATCH --job-name=apops-final-avazu
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=230G
#SBATCH --time=12:00:00
#SBATCH --output=final_experiments/logs/apops_final_avazu_%j.out

# AP-OPS fixed-test: 4 rows once on the locked stream, 3 seeds, frozen
# apops_selected.json. --saved-ops-loss is this dataset's headline OPS
# mean_imp_wt_ll for the golden tolerance check.
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p final_experiments/logs
PYTHONPATH=. .venv/bin/python final_experiments/run_apops_final.py --source avazu \
    --sample-frac 1.0 --warmup 3 --n-jobs 8 \
    --config final_experiments/avazu/apops/apops_selected.json \
    --saved-ops-loss 0.3874432394592488 \
    --out final_experiments/avazu/apops/final
