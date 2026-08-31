"""Shared CLI args + dispatch for choosing a real dataset (Criteo Attribution
or Avazu) vs the synthetic drift-injection generator (synthetic_data.py).
Used by run_baselines.py, run_advanced.py and amgtp_run.py so one --source
flag selects the same data for the whole P0/P1/P2 ladder.
"""
from data import CAT_COLUMNS, hash_features, load_dataset, load_raw, raw_numeric_features
from synthetic_data import DRIFT_MODES, generate_synthetic_ctr, generate_synthetic_raw

AVAZU_DEFAULT_DATA = "data/avazu/Avazu_x4.zip"


def add_data_source_args(parser):
    parser.add_argument("--source", choices=["criteo", "avazu", "synthetic"], default="criteo",
                         help="Real Criteo Attribution data, real Avazu data (hourly blocks), "
                              "or a synthetic drift-injection benchmark.")
    parser.add_argument("--data", default="data/criteo_attribution_dataset.tsv.gz",
                         help="[--source criteo] path to the tsv.gz file.")
    parser.add_argument("--avazu-data", default=AVAZU_DEFAULT_DATA,
                         help="[--source avazu] path to Avazu_x4.zip or a dir of train/valid/test.csv.")
    parser.add_argument("--avazu-block-hours", type=int, default=2,
                         help="[--source avazu] hours per time block (2 -> 120 blocks, 1 -> 240).")
    parser.add_argument("--sample-frac", type=float, default=1.0,
                         help="[--source criteo/avazu] subsample rows for a faster preliminary run.")
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
    if args.source == "avazu":
        from avazu_data import load_dataset as load_avazu
        print(f"Loading Avazu from {args.avazu_data} (sample_frac={args.sample_frac}, "
              f"block_hours={args.avazu_block_hours}) ...")
        return load_avazu(args.avazu_data, n_features=n_features, sample_frac=args.sample_frac,
                          seed=seed, block_hours=args.avazu_block_hours)
    print(f"Loading data from {args.data} (sample_frac={args.sample_frac}) ...")
    return load_dataset(args.data, n_features=n_features, sample_frac=args.sample_frac, seed=seed)


def load_data_with_context(args, n_features: int, seed: int):
    """Like load_data, but also returns the S4 group-A/B split (None for
    --source criteo, where the local-drift concept doesn't apply) and a
    compact per-example context matrix (raw_numeric_features) for
    m2_context_gate.py's gate -- built from the same raw dataframe as X,
    so both are row-aligned with it (and with each other)."""
    if args.source == "synthetic":
        print(f"Generating synthetic data: {args.synthetic_days} days, "
              f"{args.synthetic_rows_per_day} rows/day, drift={args.synthetic_drift} "
              f"(magnitude={args.synthetic_drift_magnitude}) ...")
        df, columns = generate_synthetic_raw(
            n_days=args.synthetic_days, rows_per_day=args.synthetic_rows_per_day,
            drift_mode=args.synthetic_drift, drift_magnitude=args.synthetic_drift_magnitude,
            shift_day=args.synthetic_shift_day, period_days=args.synthetic_period_days, seed=seed)
        X = hash_features(df, columns=columns, n_features=n_features)
        context = raw_numeric_features(df, columns=columns)
        y = df["click"].to_numpy()
        day = df["day"].to_numpy()
        group = df["group"].to_numpy()
        return X, y, day, group, context
    if args.source == "avazu":
        from avazu_data import AVAZU_CAT_COLUMNS, load_raw as load_avazu_raw, numeric_context
        print(f"Loading Avazu from {args.avazu_data} (sample_frac={args.sample_frac}, "
              f"block_hours={args.avazu_block_hours}) ...")
        df = load_avazu_raw(args.avazu_data, sample_frac=args.sample_frac, seed=seed,
                            block_hours=args.avazu_block_hours)
        X = hash_features(df, columns=AVAZU_CAT_COLUMNS, n_features=n_features)
        context = numeric_context(df)
        return X, df["click"].to_numpy(), df["day"].to_numpy(), None, context
    print(f"Loading data from {args.data} (sample_frac={args.sample_frac}) ...")
    df = load_raw(args.data, sample_frac=args.sample_frac, seed=seed)
    X = hash_features(df, n_features=n_features)
    context = raw_numeric_features(df)
    y = df["click"].to_numpy()
    day = df["day"].to_numpy()
    return X, y, day, None, context


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
    if args.source == "avazu":
        from avazu_data import AVAZU_CAT_COLUMNS, load_raw as load_avazu_raw
        print(f"Loading Avazu from {args.avazu_data} (sample_frac={args.sample_frac}, "
              f"block_hours={args.avazu_block_hours}) ...")
        return load_avazu_raw(args.avazu_data, sample_frac=args.sample_frac, seed=seed,
                              block_hours=args.avazu_block_hours), AVAZU_CAT_COLUMNS
    print(f"Loading data from {args.data} (sample_frac={args.sample_frac}) ...")
    return load_raw(args.data, sample_frac=args.sample_frac, seed=seed), CAT_COLUMNS
