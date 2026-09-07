#!/bin/bash
#SBATCH --job-name=apops-final-criteo
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --output=final_experiments/logs/apops_final_criteo_%j.out

# AP-OPS fixed-test: 4 rows once on the locked stream, 3 seeds, frozen
# apops_selected.json. --saved-ops-loss is this dataset's headline OPS
# mean_imp_wt_ll for the golden tolerance check.
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p final_experiments/logs
PYTHONPATH=. .venv/bin/python final_experiments/run_apops_final.py --source criteo \
    --sample-frac 1.0 --n-jobs 8 \
    --config final_experiments/criteo/apops/apops_selected.json \
    --saved-ops-loss 0.606958465967541 \
    --out final_experiments/criteo/apops/final
