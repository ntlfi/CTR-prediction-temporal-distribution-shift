#!/bin/bash
#SBATCH --job-name=ctr-baselines
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out

cd "$SLURM_SUBMIT_DIR"
.venv/bin/python run_baselines.py --sample-frac 1.0 --out results
