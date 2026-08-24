"""Run SFTL (Zhu et al., AAAI 2024) and merge it into the existing P0/P1/P2
comparison (PDF section 2, 5, 9). Run run_baselines.py (and, optionally,
run_advanced.py) first -- this script reuses their saved
results/per_day_metrics.csv, results/p1_p2_per_day_metrics.csv, and
results/*_comparison_table.csv.

Example:
    python run_sftl.py --source synthetic --synthetic-days 30   # quick pass
    python run_sftl.py                                          # full Criteo run
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from data import hash_indices
from data_source import add_data_source_args, load_raw_df
from metrics import day_metrics
from run_baselines import aggregate_table
from sftl import run_sftl
from splits import compute_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_source_args(parser)
    parser.add_argument("--warmup-days", type=int, default=3)
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--vocab-size", type=int, default=2**16, help="Embedding table size per categorical column.")
    parser.add_argument("--embed-dim", type=int, default=16, help="Matches the paper's DCN-Mix embedding rank.")
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64],
                         help="MLP hidden dims (paper uses 1024-512-256; sized down for CPU feasibility here).")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--ema-alpha", type=float, default=0.9,
                         help="Fast-learner EMA coefficient (undisclosed in the paper's accessible text).")
    parser.add_argument("--lambda-slow", type=float, default=0.05,
                         help="Undisclosed in the paper. Must stay small -- found empirically that larger "
                              "values (e.g. 1.0) cause runaway confidence escalation (see sftl.py docstring).")
    parser.add_argument("--lambda-fast", type=float, default=0.05, help="See --lambda-slow.")
    parser.add_argument("--sftl-warmup-domains", type=int, default=3,
                         help="Domains (days) before the trajectory loss activates.")
    parser.add_argument("--epochs-per-domain", type=int, default=1,
                         help="Passes over each day's data (paper's multi-epoch setting; use 1 for their "
                              "harder one-pass streaming setting). Raise this if the daily row count is small "
                              "enough that one pass gives too few gradient steps to leave random init.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, columns = load_raw_df(args, args.seed)
    day = df["day"].to_numpy()
    y = df["click"].to_numpy()
    print(f"Loaded {len(df)} rows, {day.max() + 1} days, click rate {y.mean():.3f}")

    eligible_days, dev_days, test_days = compute_splits(day, args.warmup_days, args.test_frac)
    print(f"Dev days: {sorted(dev_days)}")
    print(f"Test days (locked): {sorted(test_days)}")

    print(f"Hashing {len(columns)} categorical columns into embedding indices (vocab_size={args.vocab_size}) ...")
    x_idx = hash_indices(df, columns=columns, vocab_size=args.vocab_size)

    print("Training SFTL (one continuous streaming pass over the full day range) ...")
    start = time.time()
    sftl_rows = run_sftl(
        x_idx, y, day, eligible_days, n_columns=len(columns), vocab_size=args.vocab_size,
        embed_dim=args.embed_dim, hidden=tuple(args.hidden), lr=args.lr, weight_decay=args.weight_decay,
        ema_alpha=args.ema_alpha, lambda_slow=args.lambda_slow, lambda_fast=args.lambda_fast,
        warmup_domains=args.sftl_warmup_domains, epochs_per_domain=args.epochs_per_domain,
        batch_size=args.batch_size, seed=args.seed, device=args.device,
    )
    print(f"Done in {time.time() - start:.1f}s ({len(sftl_rows)} prediction days)")

    sftl_per_day = pd.DataFrame([
        {"method": "sftl", "day": r["day"], **day_metrics(r["y_true"], r["y_pred"]),
         "n_train": r["n_train"], "fit_time": r["fit_time"]}
        for r in sftl_rows
    ])
    sftl_per_day.to_csv(out_dir / "sftl_per_day_metrics.csv", index=False)

    # Merge with whatever's already in results/ (P0, and P1/P2 if present).
    per_day_frames, table_frames = [sftl_per_day], []
    for name in ["per_day_metrics.csv", "p1_p2_per_day_metrics.csv"]:
        p = out_dir / name
        if p.exists():
            per_day_frames.append(pd.read_csv(p))
    for name in ["comparison_table.csv", "all_methods_comparison_table.csv"]:
        p = out_dir / name
        if p.exists():
            table_frames.append(pd.read_csv(p))
    if not table_frames:
        print("Warning: no existing comparison table found in results/ -- run run_baselines.py "
              "(and run_advanced.py) first for a full comparison. Reporting SFTL alone.")

    all_per_day = pd.concat(per_day_frames, ignore_index=True)
    all_per_day.to_csv(out_dir / "all_methods_with_sftl_per_day_metrics.csv", index=False)

    sftl_test = sftl_per_day[sftl_per_day["day"].isin(test_days)]
    sftl_table = aggregate_table(sftl_test) if len(sftl_test) else pd.DataFrame(columns=["method"])
    table = pd.concat(table_frames + [sftl_table], ignore_index=True)
    table = table.drop_duplicates(subset="method", keep="last").sort_values("log_loss").reset_index(drop=True)
    table.to_csv(out_dir / "all_methods_with_sftl_comparison_table.csv", index=False)

    print("\n=== Locked-test comparison table (with SFTL) ===")
    print(table.to_string(index=False))
    print(f"\nAll outputs written to {out_dir}/")


if __name__ == "__main__":
    main()
