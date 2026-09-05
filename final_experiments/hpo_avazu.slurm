#!/bin/bash
#SBATCH --job-name=dualtime-hpo-avazu
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=230G
#SBATCH --time=12:00:00
#SBATCH --output=final_experiments/logs/hpo_avazu_%j.out

# DualTime-CTR final experiment -- Avazu HPO (sections 6-10). Full data
# (sample_frac=1.0, ~40M rows) per this experiment's spec -- a real
# departure from every earlier run in this repo, which used a 20%
# subsample. This cluster's nodes cap at ~250G, so each seed's dataset is
# loaded and dropped sequentially (see run_hpo.py) rather than holding
# all 3 simultaneously. If 230G still OOMs, fall back to one fixed
# deterministic subsample shared by all methods/seeds (the spec's
# explicit fallback) rather than per-seed different subsamples.
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p final_experiments/logs
PYTHONPATH=. .venv/bin/python final_experiments/run_hpo.py --source avazu --sample-frac 1.0 \
    --warmup 3 --n-jobs 8 --out final_experiments/avazu/hpo
