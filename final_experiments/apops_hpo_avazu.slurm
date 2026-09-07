#!/bin/bash
#SBATCH --job-name=apops-hpo-avazu
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=230G
#SBATCH --time=12:00:00
#SBATCH --output=final_experiments/logs/apops_hpo_avazu_%j.out

# AP-OPS development tuning (AP_OPS_Minimal_Experiment_Plan.pdf section 3):
# ONS scale/regularization + the two meta parameters, dev days only, seeds
# 0/1/2, one frozen config. Reuses this dataset's existing
# selected_configs.json for the shared mixture + locked OPS hyperparams.
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p final_experiments/logs
PYTHONPATH=. .venv/bin/python final_experiments/run_apops_hpo.py --source avazu \
    --sample-frac 1.0 --warmup 3 --n-jobs 8 \
    --ops-config final_experiments/avazu/hpo/selected_configs.json \
    --out final_experiments/avazu/apops
