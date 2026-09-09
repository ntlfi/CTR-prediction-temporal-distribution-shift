#!/bin/bash
#SBATCH --job-name=ttam-nested-criteo
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=final_experiments/logs/ttam_nested_criteo_%j.out

# TTAM revised Section 6 -- corrected fully-nested rolling-origin run,
# 9 variants, Criteo origins 16-30, seeds 0/1/2
# (TTAM_Additional_Experiments_Plan.pdf; grids frozen in TTAM_FROZEN.md).
#
# The same run_ttam_nested.py is used locally (see final_experiments/
# ttam/README or PROGRESS.md); --n-workers fans the independent outer
# origins of each pass across cores. Run ttam_banks_criteo.slurm first so
# the passes load the bank + context sketch from _bankcache/.
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p final_experiments/logs
PYTHONPATH=. .venv/bin/python final_experiments/run_ttam_nested.py \
    --source criteo --data data/criteo_attribution_dataset.tsv.gz \
    --sample-frac 1.0 --n-features 262144 \
    --n-workers "${SLURM_CPUS_PER_TASK:-8}" --n-jobs 8 \
    --cache-dir final_experiments/ttam/_bankcache \
    --out final_experiments/ttam/criteo/nested

# downstream Section 6 outputs (stats + figure + Criteo bidding)
PYTHONPATH=. bash final_experiments/run_ttam_section6.sh
