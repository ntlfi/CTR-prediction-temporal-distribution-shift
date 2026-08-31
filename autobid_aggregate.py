"""Aggregate the downstream autobidding eval (run_autobid.py cells) into
tables + a REPORT -- AMG-TP_Academic_LaTeX.pdf section 8.

For every regime it computes, across seeds, the paired difference in
**value at matched spend** (clicks, and conversions) between AMG-TP and each
reference baseline, with a Wilcoxon signed-rank test -- the same discipline
as the prediction battery (PDF section 7).

    .venv/bin/python autobid_aggregate.py
    .venv/bin/python autobid_aggregate.py --stage amgtp_experiments/stage3_autobid
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from amgtp_config import SYNTH_REGIMES

DEPLOYABLE = ["expanding", "rolling_7", "han_arw", "m2_context_gate",
              "m5b_smooth0.1", "ensemble3", "amgtp"]
ANCHORS = ["_oracle", "_noskill", "_shuffled_amgtp"]
MUT = "amgtp"
REFS = ["han_arw", "expanding", "ensemble3", "m5b_smooth0.1"]
SPEND_POINTS = ["10pct", "25pct", "50pct", "75pct"]


def load_cells(stage: Path):
    """{regime: {seed: {'summary': dict, 'matched': DataFrame}}}"""
    out = {}
    regimes = list(SYNTH_REGIMES) + ["criteo"]
    for regime in regimes:
        rdir = stage / regime
        if not rdir.is_dir():
            continue
        by_seed = {}
        for sdir in sorted(rdir.glob("seed*")):
            sj = sdir / "summary.json"
            if not sj.exists():
                continue
            seed = int(sdir.name[4:])
            rec = {"summary": json.loads(sj.read_text())}
            ms = sdir / "autobid_matched_spend.csv"
            if ms.exists():
                rec["matched"] = pd.read_csv(ms)
            by_seed[seed] = rec
        if by_seed:
            out[regime] = by_seed
    return out


def _value_at(summary, metric, spend_key, method):
    """metric in {clicks, conversions}; spend_key like '25pct'."""
    block = summary.get(f"{metric}_at_matched_spend", {}).get(spend_key, {})
    return block.get(method, np.nan)


def paired_table(cells, regime, metric="clicks"):
    seeds = sorted(cells[regime])
    rows = []
    for spend_key in SPEND_POINTS:
        mut_v = np.array([_value_at(cells[regime][s]["summary"], metric, spend_key, MUT) for s in seeds])
        for ref in REFS:
            ref_v = np.array([_value_at(cells[regime][s]["summary"], metric, spend_key, ref) for s in seeds])
            d = mut_v - ref_v
            ok = np.isfinite(d)
            if ok.sum() < 2:
                continue
            rel = 100.0 * d[ok].sum() / max(ref_v[ok].sum(), 1e-9)
            try:
                p = stats.wilcoxon(d[ok]).pvalue if np.any(d[ok] != 0) else 1.0
            except ValueError:
                p = 1.0
            rows.append({"regime": regime, "metric": metric, "spend": spend_key,
                         "ref": ref, "n_seed": int(ok.sum()),
                         "amgtp_mean": float(mut_v[ok].mean()),
                         "ref_mean": float(ref_v[ok].mean()),
                         "mean_delta": float(d[ok].mean()),
                         "rel_pct": rel, "n_amgtp_wins": int((d[ok] > 0).sum()),
                         "wilcoxon_p": p})
    return pd.DataFrame(rows)


def headline_table(cells):
    rows = []
    for regime, by_seed in cells.items():
        seeds = sorted(by_seed)
        for method in DEPLOYABLE + ANCHORS:
            for metric in ("clicks", "conversions"):
                v = np.array([_value_at(by_seed[s]["summary"], metric, "25pct", method) for s in seeds])
                v = v[np.isfinite(v)]
                if not len(v):
                    continue
                rows.append({"regime": regime, "method": method, "metric": metric,
                             "spend": "25pct", "mean": v.mean(), "std": v.std(), "n_seed": len(v)})
    return pd.DataFrame(rows)


def build_report(stage, cells):
    lines = ["# Downstream autobidding eval -- AMG-TP plan step 8", "",
             "Frozen CTR models fed into the same auction + per-block pacing "
             "(`autobid.py`). Primary metric: **value at matched spend** "
             "(clicks; and conversions, which on the synthetic source equal "
             "clicks). `_oracle` / `_noskill` / `_shuffled_amgtp` are "
             "non-deployable frontier anchors.", ""]

    hl = headline_table(cells)
    hl.to_csv(stage / "tables" / "headline.csv", index=False)

    lines.append("## Clicks won at 25% of historical spend (mean over seeds)")
    lines.append("")
    piv = (hl[(hl.metric == "clicks")]
           .pivot_table(index="regime", columns="method", values="mean"))
    cols = [c for c in DEPLOYABLE + ANCHORS if c in piv.columns]
    piv = piv[cols]
    lines.append(piv.round(1).to_markdown())
    lines.append("")

    paired_all = []
    for regime in cells:
        for metric in ("clicks", "conversions"):
            pt = paired_table(cells, regime, metric)
            if not pt.empty:
                paired_all.append(pt)
    paired_df = pd.concat(paired_all, ignore_index=True) if paired_all else pd.DataFrame()
    paired_df.to_csv(stage / "tables" / "paired_amgtp.csv", index=False)

    lines.append("## AMG-TP minus baseline, value at matched spend (paired over seeds)")
    lines.append("")
    lines.append("Negative `rel_pct` = AMG-TP wins fewer; positive = AMG-TP wins more. "
                 "`wilcoxon_p` over the per-seed paired differences.")
    lines.append("")
    show = paired_df[(paired_df.metric == "clicks") & (paired_df.spend == "25pct")]
    if not show.empty:
        lines.append(show[["regime", "ref", "n_seed", "amgtp_mean", "ref_mean",
                           "rel_pct", "n_amgtp_wins", "wilcoxon_p"]]
                     .round({"amgtp_mean": 1, "ref_mean": 1, "rel_pct": 2, "wilcoxon_p": 4})
                     .to_markdown(index=False))
    lines.append("")

    # verdict per PDF section 9
    lines.append("## Read")
    lines.append("")
    vs_han = show[show.ref == "han_arw"] if not show.empty else pd.DataFrame()
    for _, r in vs_han.iterrows():
        verdict = ("AMG-TP > Han ARW" if r.rel_pct > 0.5 and r.wilcoxon_p < 0.1
                   else "AMG-TP < Han ARW" if r.rel_pct < -0.5 and r.wilcoxon_p < 0.1
                   else "tie")
        lines.append(f"- **{r.regime}**: {verdict} "
                     f"({r.rel_pct:+.2f}%, {int(r.n_amgtp_wins)}/{int(r.n_seed)} seeds, p={r.wilcoxon_p:.3f})")
    lines.append("")

    (stage / "REPORT.md").write_text("\n".join(lines))
    _plot_regime_summary(hl, stage / "figures" / "autobid_regime_summary.png")
    return paired_df


def _plot_regime_summary(hl, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = hl[(hl.metric == "clicks")]
    regimes = sorted(d.regime.unique())
    methods = [m for m in DEPLOYABLE if m in d.method.unique()]
    fig, ax = plt.subplots(figsize=(1.4 * len(regimes) + 3, 5))
    w = 0.8 / max(len(methods), 1)
    for i, m in enumerate(methods):
        vals = [d[(d.regime == r) & (d.method == m)]["mean"].mean() /
                max(d[(d.regime == r) & (d.method == "_noskill")]["mean"].mean(), 1e-9)
                for r in regimes]
        ax.bar(np.arange(len(regimes)) + i * w, vals, w, label=m)
    ax.set_xticks(np.arange(len(regimes)) + 0.4)
    ax.set_xticklabels(regimes, rotation=30, ha="right")
    ax.set_ylabel("clicks @25% spend / no-skill")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_title("Autobidding value at matched spend, normalised to the no-skill bidder")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="amgtp_experiments/stage3_autobid")
    args = ap.parse_args()
    stage = Path(args.stage)
    cells = load_cells(stage)
    if not cells:
        raise SystemExit(f"no completed cells under {stage}")
    (stage / "tables").mkdir(parents=True, exist_ok=True)
    (stage / "figures").mkdir(parents=True, exist_ok=True)
    build_report(stage, cells)
    print(f"wrote {stage}/REPORT.md, tables/, figures/")
    print(f"regimes: {[(r, sorted(s)) for r, s in cells.items()]}")


if __name__ == "__main__":
    main()
