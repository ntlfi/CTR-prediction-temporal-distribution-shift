#!/bin/bash
#SBATCH --job-name=ctr-new-methods-criteo
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

cd "$SLURM_SUBMIT_DIR"
.venv/bin/python run_new_methods.py --sample-frac 1.0 --n-jobs 4 --out results
