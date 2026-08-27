#!/bin/bash
#SBATCH --job-name=amgtp-s2-sweep
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=amgtp_experiments/logs/s2_sweep_%j.out

cd "$SLURM_SUBMIT_DIR"
mkdir -p amgtp_experiments/logs
.venv/bin/python amgtp_stage2_sweep.py --n-jobs 4 --out amgtp_experiments/stage2_amgtp/_sweep
