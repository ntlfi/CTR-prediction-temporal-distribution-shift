#!/bin/bash
#SBATCH --job-name=withinday-manifest
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=withinday_experiments/logs/manifest_%j.out

# Day-usage manifest (rolling-protocol output 1) -- loads full Criteo +
# Avazu data just to classify days and report per-day impressions/CTR.
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p withinday_experiments/logs
.venv/bin/python withinday_manifest.py --out withinday_experiments/day_usage_manifest.csv
