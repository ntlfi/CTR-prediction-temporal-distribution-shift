"""Downstream autobidding evaluation runner -- AMG-TP_Academic_LaTeX.pdf
section 8 / recommended-order step 8.

The CTR models are *frozen* (no tuning here) and each is fed, on the locked
test period, into the *same* auction + pacing policy (``autobid.py``). We then
compare realised value at matched spend.

    adaptive temporal training -> better CTR prediction
        -> same bidder -> better value at matched spend      (PDF eq. 9)

Two data sources:

``--source criteo``  the real Criteo Attribution log with its recorded display
    ``cost`` and ``conversion``. Natural drift is shallow (the rest of the
    project); this is the PDF section 8 real-data test and a no-downside check.

``--source synthetic``  the drift-injection benchmark, with a synthetic
    second-price ``cost`` landscape tied to the *true* click probability
    (``autobid.synthetic_cost``). This is where AMG-TP's synthetic
    log-loss wins over Han ARW (S1/S3/S4) can be checked for translation into
    bidding value -- the mechanism the real data cannot yet exercise.

Methods evaluated: the frozen AMG-TP config plus the reference baselines it
is meant to replace (expanding ERM, Han ARW, M2, M5b-high-smooth, ensemble3),
and non-deployable frontier anchors (oracle / no-skill / shuffled).

Outputs in ``--out``:
  autobid_frontier.csv       method x scale -> spend, clicks, conversions, ...
  autobid_matched_spend.csv  clicks/conversions interpolated to a common spend grid
  autobid_paced.csv          method x budget fraction -> paced-auction outcome
  autobid_frontier.png       value--spend frontier, all methods
  summary.json               headline: value at 25/50/75% of historical spend
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from amgtp_run import AMGTP_CONFIG
from autobid import (linear_frontier, load_criteo_bidding, noskill_pctr,
                     oracle_pctr, paced_auction, shuffled_pctr, synthetic_cost,
                     value_at_matched_spend)
from candidate_bank import build_candidate_bank
from ensemble3 import run_ensemble3
from han_arw import run_han_arw
from m2_context_gate import run_m2
from m5_multiscale_gate import run_m5
from splits import compute_splits

FROZEN_METHOD = "amgtp"
BUDGET_FRACS = [0.1, 0.25, 0.5, 0.75, 1.0]


def _rows_to_daymap(rows):
    return {r["day"]: np.asarray(r["y_pred"], float) for r in rows}


# --------------------------------------------------------------------------- #
#  data loading                                                               #
# --------------------------------------------------------------------------- #
def load_source(args):
    """Return (X, context, y, day, auction) where ``auction`` is a dict of
    full-length arrays: cost, conversion, attribution, cpo."""
    if args.source == "criteo":
        print(f"Loading Criteo bidding log from {args.data} "
              f"(sample_frac={args.sample_frac}) ...", flush=True)
        bd = load_criteo_bidding(args.data, sample_frac=args.sample_frac,
                                 seed=args.seed, n_features=args.n_features)
        auction = {"cost": bd.cost, "conversion": bd.conversion,
                   "attribution": bd.attribution, "cpo": bd.cpo}
        return bd.X, bd.context, bd.y, bd.day, auction

    from data import hash_features, raw_numeric_features
    from synthetic_data import generate_synthetic_raw
    print(f"Generating synthetic '{args.synthetic_drift}' drift "
          f"({args.synthetic_days}d x {args.synthetic_rows_per_day}) ...", flush=True)
    df, columns = generate_synthetic_raw(
        n_days=args.synthetic_days, rows_per_day=args.synthetic_rows_per_day,
        drift_mode=args.synthetic_drift, shift_day=args.synthetic_shift_day,
        period_days=args.synthetic_period_days, seed=args.seed)
    X = hash_features(df, columns=columns, n_features=args.n_features)
    context = raw_numeric_features(df, columns=columns)
    y = df["click"].to_numpy()
    day = df["day"].to_numpy()
    cost = synthetic_cost(df["p_true"].to_numpy(), seed=args.seed, competitiveness=args.competitiveness)
    # synthetic has no separate conversion event: treat every click as a unit-
    # value conversion so "conversions"/"cpo_value" == clicks for this source.
    auction = {"cost": cost, "conversion": y.copy(),
               "attribution": y.copy(), "cpo": np.ones(len(y))}
    return X, context, y, day, auction


# --------------------------------------------------------------------------- #
#  frozen CTR models -> per-impression test predictions                       #
# --------------------------------------------------------------------------- #
def build_predictions(X, context, y, day, auction, eligible_days, dev_days,
                      test_days, T, seed, n_jobs, alpha):
    print("Building WINDOW_FAMILY candidate bank ...", flush=True)
    t0 = time.time()
    bank = build_candidate_bank(X, y, day, eligible_days, alpha=alpha, seed=seed, n_jobs=n_jobs)
    print(f"  {time.time() - t0:.1f}s", flush=True)

    have_days = [t for t in test_days if t in bank["expanding"]]

    daymaps = {
        "expanding": {t: bank["expanding"][t]["y_pred"] for t in have_days},
        "rolling_7": {t: bank["rolling_7"][t]["y_pred"] for t in have_days},
    }
    print("Han ARW ...", flush=True)
    daymaps["han_arw"] = _rows_to_daymap(run_han_arw(bank, eligible_days, dev_days=set(dev_days)))
    print("M2 context gate ...", flush=True)
    m2_rows = run_m2(bank, eligible_days, T=T, context=context, day=day, seed=seed)
    daymaps["m2_context_gate"] = _rows_to_daymap(m2_rows)
    print("M5b (smooth 1e-3, 1e-1) ...", flush=True)
    m5_def = run_m5(bank, eligible_days, T=T, smooth_reg=1e-3, context=context, day=day, seed=seed)
    m5_hi = run_m5(bank, eligible_days, T=T, smooth_reg=1e-1, context=context, day=day, seed=seed)
    daymaps["m5b_smooth0.1"] = _rows_to_daymap(m5_hi)
    print("ensemble3 ...", flush=True)
    ens3 = run_ensemble3(m2_rows, m5_def, m5_hi, T=T, context=context, day=day, seed=seed)
    daymaps["ensemble3"] = _rows_to_daymap(ens3)
    print("AMG-TP (frozen config) ...", flush=True)
    from amgtp_method import run_amgtp
    amgtp_rows = run_amgtp(bank, eligible_days, T=T, context=context, day=day,
                           seed=seed, **AMGTP_CONFIG)
    daymaps["amgtp"] = _rows_to_daymap(amgtp_rows)

    # auction columns aligned to the flattened test rows (bank y_pred for day t
    # is y[day==t] in dataset order, so a day==t mask aligns exactly)
    cols = {"day": np.concatenate([np.full(int((day == t).sum()), t) for t in have_days]),
            "click": np.concatenate([y[day == t] for t in have_days])}
    for field, arr in auction.items():
        cols[field] = np.concatenate([arr[day == t] for t in have_days])

    n = len(cols["cost"])
    preds = {}
    for m, dm in daymaps.items():
        p = np.concatenate([dm[t] for t in have_days])
        assert len(p) == n, f"{m}: {len(p)} preds vs {n} auction rows"
        preds[m] = p

    preds["_oracle"] = oracle_pctr(cols["click"])
    preds["_noskill"] = noskill_pctr(cols["click"])
    preds["_shuffled_amgtp"] = shuffled_pctr(preds["amgtp"], seed=seed)
    return preds, cols, have_days


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["criteo", "synthetic"], default="criteo")
    ap.add_argument("--data", default="data/criteo_attribution_dataset.tsv.gz")
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--synthetic-drift", default="abrupt")
    ap.add_argument("--synthetic-days", type=int, default=120)
    ap.add_argument("--synthetic-rows-per-day", type=int, default=4000)
    ap.add_argument("--synthetic-shift-day", type=int, default=None)
    ap.add_argument("--synthetic-period-days", type=int, default=14)
    ap.add_argument("--competitiveness", type=float, default=1.0,
                    help="[synthetic] market price level vs mean true CTR (autobid.synthetic_cost).")
    ap.add_argument("--n-features", type=int, default=2 ** 18)
    ap.add_argument("--warmup-days", type=int, default=3)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    X, context, y, day, auction = load_source(args)
    T = int(day.max())
    print(f"Loaded {len(y)} rows, {T + 1} days, click rate {y.mean():.4f}, "
          f"total cost {auction['cost'].sum():.3f}", flush=True)

    eligible_days, dev_days, test_days = compute_splits(day, args.warmup_days, args.test_frac)
    eligible_days = list(eligible_days)
    test_days = sorted(test_days)
    print(f"{len(eligible_days)} eligible days, {len(dev_days)} dev, {len(test_days)} locked test", flush=True)

    preds, cols, have_days = build_predictions(
        X, context, y, day, auction, eligible_days, dev_days, test_days,
        T, args.seed, args.n_jobs, args.alpha)

    cost, click = cols["cost"], cols["click"]
    conv, attrib, cpo = cols["conversion"], cols["attribution"], cols["cpo"]
    day_flat = cols["day"]
    total_cost = float(cost.sum())
    total_clicks = int(click.sum())
    print(f"test period: {len(cost)} impressions over {len(have_days)} days, "
          f"{total_clicks} clicks, historical spend {total_cost:.3f}", flush=True)

    # ---- value--spend frontier (global scale sweep) --------------------
    frontiers, frontier_rows = {}, []
    for m, p in preds.items():
        fr = linear_frontier(p, click, cost, day_flat, conv=conv, attrib=attrib, cpo=cpo, n_scales=60)
        fr.insert(0, "method", m)
        frontiers[m] = fr
        frontier_rows.append(fr)
    pd.concat(frontier_rows, ignore_index=True).to_csv(out / "autobid_frontier.csv", index=False)

    spend_grid = np.linspace(0.02 * total_cost, 0.98 * total_cost, 60)
    ms_clicks = value_at_matched_spend(frontiers, spend_grid, value="clicks")
    ms_conv = value_at_matched_spend(frontiers, spend_grid, value="conversions")
    ms = ms_clicks.merge(ms_conv, on="spend", suffixes=("_clicks", "_conversions"))
    ms.to_csv(out / "autobid_matched_spend.csv", index=False)

    # ---- budgeted paced auction --------------------------------------
    paced_rows = []
    for frac in BUDGET_FRACS:
        for m, p in preds.items():
            s, _ = paced_auction(p, click, cost, day_flat, frac * total_cost,
                                 conv=conv, attrib=attrib, cpo=cpo)
            s["method"], s["budget_frac"] = m, frac
            paced_rows.append(s)
    pd.DataFrame(paced_rows).to_csv(out / "autobid_paced.csv", index=False)

    # ---- headline: value at matched spend ---------------------------
    def at(frac, value):
        row = ms.iloc[(ms["spend"] - frac * total_cost).abs().argmin()]
        return {m: float(row[f"{m}_{value}"]) for m in preds}

    summary = {
        "config": vars(args),
        "test_impressions": len(cost),
        "test_days": len(have_days),
        "total_historical_spend": total_cost,
        "total_clicks_available": total_clicks,
        "total_conversions_available": int(conv.sum()),
        "clicks_at_matched_spend": {f"{int(f*100)}pct": at(f, "clicks") for f in (0.1, 0.25, 0.5, 0.75)},
        "conversions_at_matched_spend": {f"{int(f*100)}pct": at(f, "conversions") for f in (0.1, 0.25, 0.5, 0.75)},
        "paced_by_budget": {
            f"{int(f*100)}pct": {r["method"]: {"clicks": r["clicks"], "conversions": r["conversions"],
                                               "spend": r["spend"]}
                                 for r in paced_rows if r["budget_frac"] == f}
            for f in BUDGET_FRACS},
        "runtime_s": time.time() - t_start,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    _plot_frontier(frontiers, total_cost, args.source, out / "autobid_frontier.png")

    label = "clicks" if args.source == "criteo" else "clicks (= conversions)"
    print(f"\n=== {label} won at matched spend (25% of historical) ===")
    hdr = at(0.25, "clicks")
    base = hdr.get("_noskill", 0.0)
    for m in sorted(hdr, key=hdr.get, reverse=True):
        print(f"  {m:22s} {hdr[m]:11.1f}   ({hdr[m]-base:+.1f} vs no-skill)")
    print(f"\ntotal runtime {summary['runtime_s']:.1f}s -> {out}/")


def _plot_frontier(frontiers, total_cost, source, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for value, ax in zip(("clicks", "conversions"), axes):
        for m, fr in frontiers.items():
            fr = fr.sort_values("spend")
            style = dict(lw=2.4) if m == FROZEN_METHOD else {}
            if m.startswith("_"):
                style = dict(lw=1.0, ls="--", alpha=0.7)
            ax.plot(fr["spend"] / total_cost, fr[value], label=m, **style)
        ax.set_xlabel("spend / historical spend")
        ax.set_ylabel(f"{value} won")
        ax.set_title(f"{value} vs spend ({source} test period)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
