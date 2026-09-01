"""Package 4 -- unified Stage 4 evaluation with paired-seed uncertainty.

Pulls together:
  * the full method battery   amgtp_experiments/stage2_amgtp/<regime>/seed<s>/summary.json
  * the dense fixed-beta sweep amgtp_experiments/stage4_betasweep/betasweep_raw.csv
and produces, for the revised claim ("context-dependent multiscale gating
gives the predictive gains; adaptive persistence gives robustness across
regimes that favour different persistence levels"):

  tables/excess_vs_fixed_beta_oracle.csv   per method: seed-level mean and
        worst-regime excess log loss vs the per-regime best fixed beta
        (hindsight), with paired bootstrap 95% CIs.
  tables/per_regime_amgtp_vs.csv           AMG-TP minus {global best fixed
        beta, ensemble3, learn_alpha, fixed_share} per regime, Wilcoxon with
        Holm multiplicity correction across regimes.
  tables/cost.csv                          measured train+inference cost.
  figures/fixed_beta_curve.png             excess vs beta, per regime.
  figures/beta_trace_heterogeneous.png     beta_t through the S9 stream.
  figures/per_group_weights_local.png      per-subgroup temporal weights, S4.
  REPORT.md                                the numbers in prose.
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from amgtp_config import ALL_SEEDS, SYNTH_REGIMES

BATTERY = Path("amgtp_experiments/stage2_amgtp")
BSWEEP = Path("amgtp_experiments/stage4_betasweep")
OUT = Path("amgtp_experiments/stage4")
BETA_GRID = [round(b, 2) for b in np.arange(0.0, 1.0001, 0.05)]

METHODS = ["expanding", "han_arw", "adamoe", "fixed_share", "learn_alpha",
           "m2_context_gate", "m5b_smooth0.001", "m5b_smooth0.1", "ensemble3", "amgtp"]


def load_battery():
    """{regime: {seed: {method: log_loss}}} from the battery summaries."""
    out = {}
    for rk in SYNTH_REGIMES:
        rd = BATTERY / rk
        if not rd.is_dir():
            continue
        out[rk] = {}
        for sd in sorted(rd.glob("seed*")):
            sj = sd / "summary.json"
            if not sj.exists():
                continue
            s = int(sd.name[4:])
            m = json.loads(sj.read_text())["methods"]
            out[rk][s] = {k: m[k]["log_loss"] for k in m}
    return out


def load_bsweep():
    if not (BSWEEP / "betasweep_raw.csv").exists():
        return None
    return pd.read_csv(BSWEEP / "betasweep_raw.csv")


def boot_ci(v, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    bs = (rng.choice(v, (n, len(v)), replace=True)).mean(axis=1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run_max = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * p[idx]
        run_max = max(run_max, val)
        adj[idx] = min(run_max, 1.0)
    return adj


def main():
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    bat = load_battery()
    bs = load_bsweep()
    regimes = [r for r in SYNTH_REGIMES if r in bat and bat[r]]
    seeds = ALL_SEEDS

    # ---- fixed-beta oracle per regime, and per-(regime,seed) loss for every method ----
    # betasweep gives fixed_beta_* + amgtp_adaptive + ensemble3 per (regime, seed).
    fb_cols = [f"fixed_beta_{b:.2f}" for b in BETA_GRID]
    oracle_beta, oracle_seed_loss = {}, {}
    bs_piv = None
    if bs is not None:
        bs_piv = bs.pivot_table(index=["regime", "seed"], columns="config", values="ll")
        seed_mean = bs.groupby(["regime", "config"])["ll"].mean().unstack("config")
        for rk in regimes:
            if rk not in seed_mean.index:
                continue
            avail = [c for c in fb_cols if c in seed_mean.columns and np.isfinite(seed_mean.loc[rk, c])]
            b_star = seed_mean.loc[rk, avail].idxmin()
            oracle_beta[rk] = float(b_star.replace("fixed_beta_", ""))
            oracle_seed_loss[rk] = {s: bs_piv.loc[(rk, s), b_star]
                                    for s in seeds if (rk, s) in bs_piv.index}

    # globally best fixed beta = argmin over b of mean across (regime, seed)
    glob_beta = None
    if bs is not None:
        gm = bs[bs.config.isin(fb_cols)].groupby("config")["ll"].mean()
        glob_beta = float(gm.idxmin().replace("fixed_beta_", ""))

    def method_loss(method, rk, s):
        if method == "global_fixed_beta" and bs_piv is not None:
            key = f"fixed_beta_{glob_beta:.2f}"
            return bs_piv.loc[(rk, s), key] if (rk, s) in bs_piv.index and key in bs_piv.columns else np.nan
        if method in bat.get(rk, {}).get(s, {}):
            return bat[rk][s][method]
        if bs_piv is not None and method == "amgtp" and (rk, s) in bs_piv.index and "amgtp_adaptive" in bs_piv.columns:
            return bs_piv.loc[(rk, s), "amgtp_adaptive"]
        return np.nan

    # ---- excess vs per-regime fixed-beta oracle: seed-level mean & worst ----
    methods_report = METHODS + (["global_fixed_beta"] if glob_beta is not None else [])
    rows = []
    per_seed = {m: {"mean": [], "worst": []} for m in methods_report}
    for m in methods_report:
        for s in seeds:
            exc = []
            for rk in regimes:
                if rk not in oracle_seed_loss or s not in oracle_seed_loss[rk]:
                    continue
                ml = method_loss(m, rk, s)
                if np.isfinite(ml):
                    exc.append(ml - oracle_seed_loss[rk][s])
            if exc:
                per_seed[m]["mean"].append(float(np.mean(exc)))
                per_seed[m]["worst"].append(float(np.max(exc)))
        mmean, mlo, mhi = boot_ci(per_seed[m]["mean"])
        wmean, wlo, whi = boot_ci(per_seed[m]["worst"])
        rows.append(dict(method=m, mean_excess=mmean, mean_ci_lo=mlo, mean_ci_hi=mhi,
                         worst_excess=wmean, worst_ci_lo=wlo, worst_ci_hi=whi,
                         n_seeds=len(per_seed[m]["mean"])))
    exc_df = pd.DataFrame(rows).sort_values("worst_excess")
    exc_df.to_csv(OUT / "tables" / "excess_vs_fixed_beta_oracle.csv", index=False)

    # ---- per-regime AMG-TP vs alternatives, Holm-corrected ----
    comp = [("global_fixed_beta", glob_beta), ("ensemble3", None),
            ("learn_alpha", None), ("fixed_share", None), ("m5b_smooth0.001", None)]
    prr = []
    for ref, _ in comp:
        pv, dd = [], []
        for rk in regimes:
            a = np.array([method_loss("amgtp", rk, s) for s in seeds], float)
            b = np.array([method_loss(ref, rk, s) for s in seeds], float)
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            diff = a - b
            p = stats.wilcoxon(diff).pvalue if len(diff) >= 6 and np.any(diff != 0) else np.nan
            pv.append(p)
            dd.append(diff.mean())
        adj = holm([x if np.isfinite(x) else 1.0 for x in pv])
        for rk, d, p, pa in zip(regimes, dd, pv, adj):
            prr.append(dict(reference=ref, regime=rk, amgtp_minus_ref=d, wilcoxon_p=p, holm_p=pa))
    pd.DataFrame(prr).to_csv(OUT / "tables" / "per_regime_amgtp_vs.csv", index=False)

    # ---- cost microbench (one bank, one seed) ----
    cost_rows = _cost_bench()
    pd.DataFrame(cost_rows).to_csv(OUT / "tables" / "cost.csv", index=False)

    # ---- figures ----
    _fig_fixed_beta_curve(bs, regimes, oracle_beta)
    _fig_beta_trace_heterogeneous()
    _fig_per_group_weights_local()

    _write_report(exc_df, pd.DataFrame(prr), pd.DataFrame(cost_rows), regimes,
                  oracle_beta, glob_beta)
    print(f"wrote {OUT}/REPORT.md, tables/, figures/")


def _cost_bench():
    from candidate_bank import build_candidate_bank
    from synthetic_data import generate_synthetic_raw
    from data import hash_features, raw_numeric_features
    from splits import compute_splits
    from amgtp_method import run_amgtp
    from amgtp_run import AMGTP_CONFIG
    from ensemble3 import run_ensemble3
    from m2_context_gate import run_m2
    from m5_multiscale_gate import run_m5
    from expert_tracking import run_fixed_share, run_learn_alpha
    from han_arw import run_han_arw
    df, cols = generate_synthetic_raw(n_days=120, rows_per_day=3000, drift_mode="recurring", seed=0)
    X = hash_features(df, columns=cols, n_features=2**18)
    ctx = raw_numeric_features(df, columns=cols)
    y, day, grp = df.click.to_numpy(), df.day.to_numpy(), df.group.to_numpy()
    elig, dev, test = compute_splits(day, 3, 0.3)
    T = int(day.max())
    t0 = time.time()
    bank = build_candidate_bank(X, y, day, list(elig), seed=0, n_jobs=4)
    t_bank = time.time() - t0

    def timeit(fn):
        t = time.time(); fn(); return time.time() - t
    out = [dict(component="candidate_bank (shared by all)", seconds=round(t_bank, 2), note="5 SGD experts")]
    out.append(dict(component="han_arw (selection)", seconds=round(timeit(
        lambda: run_han_arw(bank, elig, dev_days=set(dev))), 3), note="over the shared bank"))
    out.append(dict(component="fixed_share", seconds=round(timeit(
        lambda: run_fixed_share(bank, elig)), 3), note="EW + share, no training"))
    out.append(dict(component="learn_alpha", seconds=round(timeit(
        lambda: run_learn_alpha(bank, elig)), 3), note="7 Fixed-Share sub-algs"))
    m2 = run_m2(bank, elig, T=T, context=ctx, day=day, seed=0)
    m5lo = run_m5(bank, elig, T=T, smooth_reg=1e-3, context=ctx, day=day, seed=0)
    m5hi = run_m5(bank, elig, T=T, smooth_reg=0.1, context=ctx, day=day, seed=0)
    out.append(dict(component="m5b gate (1x)", seconds=round(timeit(
        lambda: run_m5(bank, elig, T=T, smooth_reg=0.1, context=ctx, day=day, seed=0)), 3),
        note="online context gate"))
    out.append(dict(component="ensemble3", seconds=round(timeit(
        lambda: run_ensemble3(m2, m5lo, m5hi, T=T, context=ctx, day=day, seed=0)), 3),
        note="meta-gate ONLY (needs m2 + 2x m5b first: ~3x gate cost)"))
    out.append(dict(component="amgtp (adaptive)", seconds=round(timeit(
        lambda: run_amgtp(bank, elig, T=T, context=ctx, day=day, seed=0, **AMGTP_CONFIG)), 3),
        note="one gate + one persistence net"))
    return out


def _fig_fixed_beta_curve(bs, regimes, oracle_beta):
    if bs is None:
        return
    fbm = bs[bs.config.str.startswith("fixed_beta_")].copy()
    fbm["beta"] = fbm.config.str.replace("fixed_beta_", "").astype(float)
    g = fbm.groupby(["regime", "beta"])["ll"].mean().reset_index()
    ncol = 4
    nrow = int(np.ceil(len(regimes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), squeeze=False)
    for i, rk in enumerate(regimes):
        ax = axes[i // ncol][i % ncol]
        sub = g[g.regime == rk].sort_values("beta")
        base = sub.ll.min()
        ax.plot(sub.beta, sub.ll - base, marker="o", ms=3)
        if rk in oracle_beta:
            ax.axvline(oracle_beta[rk], color="green", ls="--", lw=1, label=f"beta*={oracle_beta[rk]:.2f}")
        ax.set_title(SYNTH_REGIMES[rk]["label"], fontsize=8)
        ax.set_xlabel("fixed beta", fontsize=7)
        ax.set_ylabel("excess log loss", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
    for j in range(len(regimes), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Fixed-beta performance curve (excess over the regime's best fixed beta)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fixed_beta_curve.png", dpi=140)
    plt.close(fig)


def _fig_beta_trace_heterogeneous():
    traces = sorted((BATTERY / "s9_heterogeneous").glob("seed*/amgtp_beta_trace.csv"))
    if not traces:
        return
    from synthetic_data import _heterogeneous_plan
    _, _, segs = _heterogeneous_plan(120, np.random.default_rng(0))
    fig, ax = plt.subplots(figsize=(11, 4))
    for p in traces[:8]:
        d = pd.read_csv(p)
        ax.plot(d.day, d.beta, lw=1, alpha=0.5)
    dm = pd.concat([pd.read_csv(p) for p in traces]).groupby("day")["beta"].mean()
    ax.plot(dm.index, dm.values, color="crimson", lw=2.5, label="mean beta_t")
    for name, lo, hi in segs:
        ax.axvspan(lo, hi, alpha=0.06, color="k")
        ax.text((lo + hi) / 2, 1.02, name, ha="center", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("prediction block")
    ax.set_ylabel("deployed beta_t")
    ax.set_title("AMG-TP beta_t through the S9 heterogeneous stream")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "beta_trace_heterogeneous.png", dpi=140)
    plt.close(fig)


def _fig_per_group_weights_local():
    gp = sorted((BATTERY / "s4_local").glob("seed*/group_per_day_metrics.csv"))
    # use the gate_dynamics weights split by group is not stored; fall back to
    # amgtp beta_A/beta_B trace if present, else per-group loss curves.
    bx = sorted((BATTERY / "s4_local").glob("seed*/amgtp_bx_beta_trace.csv"))
    fig, ax = plt.subplots(figsize=(10, 4))
    if bx:
        d = pd.concat([pd.read_csv(p) for p in bx]).groupby("day")[["beta_A", "beta_B"]].mean()
        ax.plot(d.index, d.beta_A, label="group A (shifts at block 60)", lw=2)
        ax.plot(d.index, d.beta_B, label="group B (never shifts)", lw=2)
        ax.axvline(60, color="k", ls=":", lw=1)
        ax.set_ylabel("mean deployed beta per subgroup")
        ax.set_title("S4 local drift: per-subgroup persistence beta_t(x)")
    elif gp:
        d = pd.concat([pd.read_csv(p) for p in gp])
        d = d[d.method == "amgtp"].groupby(["day", "group"])["log_loss"].mean().unstack("group")
        for c in d.columns:
            ax.plot(d.index, d[c], label=f"group {c}", lw=1.5)
        ax.axvline(60, color="k", ls=":", lw=1)
        ax.set_ylabel("AMG-TP per-subgroup log loss")
        ax.set_title("S4 local drift: per-subgroup AMG-TP loss")
    ax.set_xlabel("prediction block")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "per_group_weights_local.png", dpi=140)
    plt.close(fig)


def _write_report(exc_df, prr, cost, regimes, oracle_beta, glob_beta):
    L = ["# Stage 4 -- revised-claim evaluation", "",
         f"Regimes ({len(regimes)}): {regimes}. Seeds: {ALL_SEEDS} ({len(ALL_SEEDS)}). ",
         "Excess = locked-test log loss minus the per-regime best fixed beta (chosen "
         "in hindsight from the dense 0..1 sweep). Paired over seeds.", "",
         f"Globally best single fixed beta (over all regimes x seeds): **{glob_beta}**.", "",
         "## Excess vs the per-regime fixed-beta oracle (seed-level, paired bootstrap 95% CI)", "",
         "| method | mean excess [95% CI] | worst-regime excess [95% CI] |", "|---|---|---|"]
    for _, r in exc_df.iterrows():
        L.append(f"| {r['method']} | {r['mean_excess']:+.4f} [{r['mean_ci_lo']:+.4f}, {r['mean_ci_hi']:+.4f}] "
                 f"| {r['worst_excess']:+.4f} [{r['worst_ci_lo']:+.4f}, {r['worst_ci_hi']:+.4f}] |")
    L += ["", "## Per-regime AMG-TP minus reference (Wilcoxon, Holm-corrected across regimes)",
          "Negative = AMG-TP better.", "",
          "| reference | regime | mean Δ | Wilcoxon p | Holm p |", "|---|---|--:|--:|--:|"]
    for _, r in prr.iterrows():
        L.append(f"| {r['reference']} | {r['regime']} | {r['amgtp_minus_ref']:+.4f} | "
                 f"{r['wilcoxon_p']:.3g} | {r['holm_p']:.3g} |")
    L += ["", "## Measured cost (one 120-block, 3000-row/block bank)", "",
          "| component | seconds | note |", "|---|--:|---|"]
    for _, r in cost.iterrows():
        L.append(f"| {r['component']} | {r['seconds']} | {r['note']} |")
    L += ["", "## Figures", "- `figures/fixed_beta_curve.png` -- excess vs beta per regime",
          "- `figures/beta_trace_heterogeneous.png` -- beta_t through S9",
          "- `figures/per_group_weights_local.png` -- per-subgroup beta under S4"]
    (OUT / "REPORT.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
