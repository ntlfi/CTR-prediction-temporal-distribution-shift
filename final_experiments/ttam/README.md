# TTAM revised Section 6 — how to run

Integrated AMG-TP + AP-OPS pipeline ("TTAM") under one corrected
fully-nested rolling-origin protocol. Spec:
`final_experiments/TTAM_Additional_Experiments_Plan.pdf`. Frozen grids /
selection rule / statistics / bidding protocol:
`final_experiments/TTAM_FROZEN.md`.

The **same** `run_ttam_nested.py` drives both a local machine and a Slurm
node — `--n-workers` fans each pass's independent outer origins across
cores. Keep both entry points:

## 1. Local

```bash
# one-time: build + cache the expert banks and context sketches
PYTHONPATH=. python3 final_experiments/ttam/build_banks.py \
    --source criteo --data data/criteo_attribution_dataset.tsv.gz \
    --cache-dir final_experiments/ttam/_bankcache --n-jobs 4

# nested run (3 passes: module -> calibration -> score)
PYTHONPATH=. python3 final_experiments/run_ttam_nested.py \
    --source criteo --data data/criteo_attribution_dataset.tsv.gz \
    --out final_experiments/ttam/criteo/nested \
    --n-workers 4 --n-jobs 4

# Section 6 outputs: stats MD + figure + Criteo bidding
bash final_experiments/run_ttam_section6.sh
```

`--n-workers` is a memory/throughput knob only — results are identical to
`--n-workers 1`. Size it to `min(#origins, cores)` and to
`RAM / (~3.5 GB per worker for Criteo, more for Avazu)`. On a shared box,
watch free memory: each worker holds the seed's bank + context sketch.

**Resumable.** Each pass writes a per-seed checkpoint
(`pass{1,2}_seed{k}.json`, `seed{k}/per_day_metrics.csv`); re-running the
same command skips finished seeds. Delete the `nested/` dir to start over.

## 2. Slurm

```bash
sbatch final_experiments/ttam_banks_criteo.slurm     # then, when done:
sbatch final_experiments/ttam_nested_criteo.slurm    # runs the nested pass + section 6

sbatch final_experiments/ttam_banks_avazu.slurm      # ~40M rows, 230G
sbatch final_experiments/ttam_nested_avazu.slurm
```

The `.slurm` wrappers call the identical scripts with
`--n-workers ${SLURM_CPUS_PER_TASK}` and `.venv/bin/python`.

## Files

| module | role |
|---|---|
| `bank.py` | shared 5-horizon expert bank {1,3,7,14,expanding}, tie-aliased |
| `fixed_mix.py` | constant validation-fitted simplex mixture (no-AMG-TP control) |
| `amgtp.py` | causal AMG-TP on the 5-horizon bank (sample gate + learned persistence + deployed-weight memory) |
| `method.py` | the nine variants on identical expert predictions |
| `build_banks.py` | bank + context-sketch disk cache (`_bankcache/`, gitignored) |
| `../run_ttam_nested.py` | 3-pass nested runner (parallel origins, resumable) |
| `../ttam_stats.py` | paired day-level gain + 95% paired day-bootstrap + block-2 MBB |
| `../ttam_figure.py` | Section 6.2 two-panel origin-level gain figure |
| `../run_ttam_bidding.py` | Criteo bidding replay (recorded display cost as price proxy) |
| `../ttam_tests.py` | leakage / identity / determinism checks (10/10) |
