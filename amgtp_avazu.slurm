#!/bin/bash
#SBATCH --job-name=amgtp-avazu
#SBATCH --partition=short
#SBATCH --account=cklab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=amgtp_experiments/logs/avazu_%A_%a.out

# AMG-TP plan Stage 1 + Stage 2 on the second real temporal dataset (Avazu,
# 2-hour blocks -> 120-block horizon matching the synthetic suite) -- PDF
# section 5.3 "insufficient real evidence" outcome. One pass runs the full
# baseline + M5b-smoothness suite AND AMG-TP + ablations (--stage2). 8 seeds,
# each drawing a disjoint 20% row-subsample of the 40.4M rows, so the seeds
# are a genuine source of variation (unlike full Criteo).
cd "$SLURM_SUBMIT_DIR"
set -e
mkdir -p amgtp_experiments/logs
PY=.venv/bin/python

SEEDS=(0 1 2 3 4 5 6 7)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=amgtp_experiments/stage2_amgtp/avazu/seed${SEED}

$PY amgtp_run.py --source avazu \
    --avazu-data /insomnia001/home/tn2447/data/avazu/Avazu_x4.zip \
    --avazu-block-hours 2 --sample-frac 0.2 --seed "$SEED" --n-jobs 8 --stage2 \
    --warmup-days 12 --test-frac 0.3 \
    --diff-forgetting-maxiter 8 \
    --out "$OUT"

# make the same cells visible to the Stage 1 aggregation too
mkdir -p amgtp_experiments/stage1_m5b_high_smooth/avazu
ln -sfn "../../stage2_amgtp/avazu/seed${SEED}" \
    "amgtp_experiments/stage1_m5b_high_smooth/avazu/seed${SEED}"
