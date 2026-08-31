"""Shared configuration for the AMG-TP experimental plan (Stage 1 + Stage 2):
the synthetic-shift suite, the seed sets, and the output directory layout.
Imported by amgtp_run driver scripts and amgtp_aggregate.py so the SLURM
array indexing and the aggregation always agree.
"""

# Synthetic horizon shared by every regime (plan section 12: >= 120 blocks).
SYNTH_DAYS = 120
SYNTH_ROWS_PER_DAY = 3000
SYNTH_PERIOD_DAYS = 14

# AMG-TP_Academic_LaTeX.pdf Table 2 -- S0..S6. Each entry: extra CLI args for
# amgtp_run.py beyond --source synthetic --seed --out.
SYNTH_REGIMES = {
    "s0_none":            {"drift": "none",           "label": "S0 stationary"},
    "s1_abrupt":          {"drift": "abrupt",         "label": "S1 abrupt global",
                           "shift_day": SYNTH_DAYS // 2},
    "s2_gradual":         {"drift": "gradual",        "label": "S2 gradual"},
    "s3_recurring":       {"drift": "recurring",      "label": "S3 recurring"},
    "s4_local":           {"drift": "local",          "label": "S4 local/subpopulation",
                           "shift_day": SYNTH_DAYS // 2},
    "s5_opposing_local":  {"drift": "opposing_local", "label": "S5 opposing local"},
    "s6_mixed":           {"drift": "mixed",          "label": "S6 mixed unknown"},
}

# plan section 12/17: >= 10 seeds; hyperparameters were frozen on the dev seeds
# used in earlier project work (results/m5_analysis.md), confirmation reported
# on a disjoint fresh set.
DEV_SEEDS = [0, 1, 2, 3, 4]
CONFIRM_SEEDS = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
ALL_SEEDS = DEV_SEEDS + CONFIRM_SEEDS

CRITEO_SEEDS = [0, 1, 2]
CRITEO_DATA = "/insomnia001/home/tn2447/data/criteo/criteo_attribution_dataset.tsv.gz"

# Avazu: the second real temporal dataset (PDF section 5.3). 10-day mobile-ad
# click logs, indexed in 2-hour blocks (120-block horizon), each seed drawing
# a disjoint 20% row-subsample so seeds are a genuine source of variation.
AVAZU_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
AVAZU_DATA = "/insomnia001/home/tn2447/data/avazu/Avazu_x4.zip"
AVAZU_SAMPLE_FRAC = 0.2

STAGE1_DIR = "amgtp_experiments/stage1_m5b_high_smooth"
STAGE2_DIR = "amgtp_experiments/stage2_amgtp"

# Downstream autobidding eval (PDF section 8 / step 8). Same synthetic regimes
# and seed grid as the prediction battery; a separate dev seed set is enough
# for a Wilcoxon on the value-at-matched-spend gap.
AUTOBID_DIR = "amgtp_experiments/stage3_autobid"
AUTOBID_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
AUTOBID_SYNTH_ROWS_PER_DAY = 4000


def autobid_synth_grid():
    return [(rk, s) for rk in SYNTH_REGIMES for s in AUTOBID_SEEDS]


def synth_run_args(regime_key: str) -> list:
    r = SYNTH_REGIMES[regime_key]
    args = ["--source", "synthetic",
            "--synthetic-days", str(SYNTH_DAYS),
            "--synthetic-rows-per-day", str(SYNTH_ROWS_PER_DAY),
            "--synthetic-drift", r["drift"],
            "--synthetic-period-days", str(SYNTH_PERIOD_DAYS)]
    if "shift_day" in r:
        args += ["--synthetic-shift-day", str(r["shift_day"])]
    return args


# Flattened (regime, seed) grid for SLURM array indexing.
def synth_grid():
    return [(rk, s) for rk in SYNTH_REGIMES for s in ALL_SEEDS]


if __name__ == "__main__":
    import sys
    # `python amgtp_config.py cell <i> <stage_dir>` -> "<regime>\t<seed>\t<out>\t<run args...>"
    # `python amgtp_config.py ncells` -> number of synthetic cells
    if sys.argv[1] == "ncells":
        print(len(synth_grid()))
    elif sys.argv[1] == "cell":
        i = int(sys.argv[2])
        stage_dir = sys.argv[3] if len(sys.argv) > 3 else STAGE1_DIR
        regime, seed = synth_grid()[i]
        out = f"{stage_dir}/{regime}/seed{seed}"
        print("\t".join([regime, str(seed), out, " ".join(synth_run_args(regime))]))
    elif sys.argv[1] == "autobid-ncells":
        print(len(autobid_synth_grid()))
    elif sys.argv[1] == "autobid-cell":
        i = int(sys.argv[2])
        regime, seed = autobid_synth_grid()[i]
        r = SYNTH_REGIMES[regime]
        args = ["--source", "synthetic", "--synthetic-drift", r["drift"],
                "--synthetic-days", str(SYNTH_DAYS),
                "--synthetic-rows-per-day", str(AUTOBID_SYNTH_ROWS_PER_DAY),
                "--synthetic-period-days", str(SYNTH_PERIOD_DAYS)]
        if "shift_day" in r:
            args += ["--synthetic-shift-day", str(r["shift_day"])]
        print("\t".join([regime, str(seed), f"{AUTOBID_DIR}/{regime}/seed{seed}", " ".join(args)]))
    else:
        raise SystemExit(f"unknown command {sys.argv[1]!r}")
