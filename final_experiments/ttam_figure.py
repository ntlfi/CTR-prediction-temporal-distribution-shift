"""Section 6.2 figure (plan section 8): two dataset panels, origin-level
gains of TTAM over each comparator with the mean-gain 95% interval
overlaid.

Small-multiples layout: one row per comparator (Expanding, Best Fixed
Window, ARW, AdaMoE, OPS), one column per dataset.  Each cell shows every
origin's gain ``G_{m,d} = A_{m,d} - A_{TTAM,d}`` as a stem from zero
(positive = TTAM better), a solid line at the mean gain ``G_m`` and a
shaded band for its 95% paired day-bootstrap interval.  **Every origin is
shown, favourable or not** -- this is not a curated subset.

    PYTHONPATH=. python3 final_experiments/ttam_figure.py \
        --dir final_experiments/ttam/criteo/nested --label Criteo \
        --dir final_experiments/ttam/avazu/nested  --label Avazu \
        --out final_experiments/ttam/section6_figure.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ttam_stats import analyse, MAIN_METHODS, PRETTY, TTAM


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", action="append", dest="dirs", required=True)
    ap.add_argument("--label", action="append", dest="labels", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    analyses = [analyse(Path(d), lab) for d, lab in zip(args.dirs, args.labels)]
    comparators = [m for m in MAIN_METHODS if m != TTAM]
    nrow, ncol = len(comparators), len(analyses)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 1.9 * nrow),
                             squeeze=False, sharex="col")

    for j, a in enumerate(analyses):
        days = np.array(a["days"])
        sa = a["seed_averaged_log_loss"]
        a_ttam = np.array([sa[TTAM][d] for d in days])
        main = a["main"].set_index("method")
        for i, m in enumerate(comparators):
            ax = axes[i][j]
            pretty = PRETTY[m]
            if m not in sa or pretty not in main.index:
                ax.set_visible(False)
                continue
            a_m = np.array([sa[m][d] for d in days])
            gain = a_m - a_ttam
            row = main.loc[pretty]
            ax.axhline(0, color="0.6", lw=0.8, zorder=1)
            ax.axhspan(row["gain_ci95_lo"], row["gain_ci95_hi"], color="tab:blue",
                       alpha=0.15, zorder=1)
            ax.axhline(row["G_m (mean gain vs TTAM)"], color="tab:blue", lw=1.6, zorder=2)
            colors = np.where(gain > 0, "tab:green", "tab:red")
            ax.vlines(days, 0, gain, color=colors, lw=2, zorder=3)
            ax.scatter(days, gain, color=colors, s=16, zorder=4)
            ax.set_ylabel(f"{pretty}\ngain", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(f"{a['label']}  (D = {len(days)} origins)", fontsize=10)
            if i == nrow - 1:
                ax.set_xlabel("origin day", fontsize=8)

    fig.suptitle("Origin-level gain of TTAM over each comparator "
                 "(positive = TTAM lower log loss; band = 95% mean-gain CI)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
