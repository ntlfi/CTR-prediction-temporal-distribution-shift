"""Shared CLI args + dispatch for choosing the real Criteo dataset vs the
synthetic drift-injection generator (synthetic_data.py). Used by both
run_baselines.py and run_advanced.py so one --source flag selects the same
data for the whole P0/P1/P2 ladder.
"""
from data import CAT_COLUMNS, load_dataset, load_raw
from synthetic_data import DRIFT_MODES, generate_synthetic_ctr, generate_synthetic_raw


def add_data_source_args(parser):
    parser.add_argument("--source", choices=["criteo", "synthetic"], default="criteo",
                         help="Real Criteo Attribution data, or a synthetic drift-injection benchmark.")
    parser.add_argument("--data", default="data/criteo_attribution_dataset.tsv.gz",
                         help="[--source criteo] path to the tsv.gz file.")
    parser.add_argument("--sample-frac", type=float, default=1.0,
                         help="[--source criteo] subsample rows for a faster preliminary run.")
    parser.add_argument("--synthetic-days", type=int, default=180)
    parser.add_argument("--synthetic-rows-per-day", type=int, default=5000)
    parser.add_argument("--synthetic-drift", choices=DRIFT_MODES, default="gradual",
                         help="[--source synthetic] ground-truth drift schedule for the true CTR model.")
    parser.add_argument("--synthetic-drift-magnitude", type=float, default=1.0)
    parser.add_argument("--synthetic-shift-day", type=int, default=None,
                         help="[--source synthetic, drift=abrupt] day of the regime change (default: midpoint).")
    parser.add_argument("--synthetic-period-days", type=int, default=14,
                         help="[--source synthetic, drift=recurring] oscillation period in days.")


def load_data(args, n_features: int, seed: int):
    if args.source == "synthetic":
        print(f"Generating synthetic data: {args.synthetic_days} days, "
              f"{args.synthetic_rows_per_day} rows/day, drift={args.synthetic_drift} "
              f"(magnitude={args.synthetic_drift_magnitude}) ...")
        return generate_synthetic_ctr(
            n_days=args.synthetic_days, rows_per_day=args.synthetic_rows_per_day,
            drift_mode=args.synthetic_drift, drift_magnitude=args.synthetic_drift_magnitude,
            shift_day=args.synthetic_shift_day, period_days=args.synthetic_period_days,
            n_features=n_features, seed=seed)
    print(f"Loading data from {args.data} (sample_frac={args.sample_frac}) ...")
    return load_dataset(args.data, n_features=n_features, sample_frac=args.sample_frac, seed=seed)


def load_raw_df(args, seed: int):
    """Raw (categorical columns, click, day) DataFrame, for sftl.py's
    embedding-index representation. Returns (df, columns)."""
    if args.source == "synthetic":
        print(f"Generating synthetic data: {args.synthetic_days} days, "
              f"{args.synthetic_rows_per_day} rows/day, drift={args.synthetic_drift} "
              f"(magnitude={args.synthetic_drift_magnitude}) ...")
        return generate_synthetic_raw(
            n_days=args.synthetic_days, rows_per_day=args.synthetic_rows_per_day,
            drift_mode=args.synthetic_drift, drift_magnitude=args.synthetic_drift_magnitude,
            shift_day=args.synthetic_shift_day, period_days=args.synthetic_period_days, seed=seed)
    print(f"Loading data from {args.data} (sample_frac={args.sample_frac}) ...")
    return load_raw(args.data, sample_frac=args.sample_frac, seed=seed), CAT_COLUMNS
