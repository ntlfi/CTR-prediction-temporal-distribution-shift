"""Plan section 3 inference rule, applied to the AP-OPS fixed-test and
rolling-origin outputs: average the 3 seeds within each calendar day
first, then bootstrap / sign-test **across days** (the exchangeable unit
for a temporal claim), differencing every row against ``current_ops``.

Also folds in the temporal metrics (pre-feedback / first-quarter /
worst-day pooled log loss) from each seed's ``summary.json`` and the
AP-OPS mechanism trace (final weights, memory-scale block dominance) from
``seed*/mechanism.json``, and re-states the section-4 decision from the
frozen ``delta_NI``.

Emits a markdown block ready to paste into ``APOPS_FINDINGS.md``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from withinday.daystats import day_summary

LABEL = {"current_ops": "Current OPS", "ops": "OPS", "ap_ops": "AP-OPS",
         "single_memory": "Single-memory", "no_slope": "No-slope",
         "reset_ons": "Reset-ONS", "persistent_ons": "Persistent-ONS"}
FIXED_ROWS = ["current_ops", "ap_ops", "single_memory", "no_slope"]
NESTED_ROWS = ["ops", "reset_ons", "persistent_ons", "ap_ops", "no_slope"]

ROWS = FIXED_ROWS
BASELINE = "current_ops"


def seed_avg_by_day(result_dir: Path) -> pd.DataFrame:
    frames = []
    for sd in sorted(result_dir.glob("seed*")):
        csv = sd / "per_day_metrics.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df["seed"] = int(sd.name.replace("seed", ""))
            frames.append(df[["seed", "method", "day", "n", "log_loss"]])
    if not frames:
        raise FileNotFoundError(f"no seed*/per_day_metrics.csv under {result_dir}")
    raw = pd.concat(frames, ignore_index=True)
    return raw.groupby(["method", "day"]).agg(lbar=("log_loss", "mean"),
                                              n=("n", "mean")).reset_index()


def analyse(result_dir: Path, label: str, rows=None, baseline=None) -> dict:
    rows = rows or ROWS
    baseline = baseline or BASELINE
    sa = seed_avg_by_day(result_dir)
    piv = sa.pivot_table(index="day", columns="method", values="lbar").sort_index()
    n_by_day = sa.groupby("day")["n"].first().reindex(piv.index).to_numpy()
    days = [int(d) for d in piv.index]

    table = []
    for m in rows:
        if m not in piv:
            continue
        row = {"method": LABEL.get(m, m), "seed_avg_mean_ll": float(piv[m].mean())}
        if m != baseline and baseline in piv:
            deltas = (piv[m] - piv[baseline]).to_numpy()
            s = day_summary(deltas, seed=0)
            row.update({
                "mean_delta_vs_ops": s["mean_delta"],
                "imp_wt_delta_vs_ops": float(np.sum(deltas * n_by_day) / np.sum(n_by_day)),
                "ci95_lo": s["ci95_lo"], "ci95_hi": s["ci95_hi"],
                "ci_excludes_0": bool(s["ci95_hi"] < 0 or s["ci95_lo"] > 0),
                "n_days_won": s["n_days_won"], "n_days": s["n_days"],
                "sign_test_p": s["sign_test_p"], "worst_day_delta": s["worst_day_delta"],
                "loo_reverses_sign": s["loo_reverses_sign"],
            })
        table.append(row)

    # temporal metrics + mechanism from the per-seed json
    temporal, mech = {}, {}
    seed_summ = []
    for sd in sorted(result_dir.glob("seed*")):
        sj = sd / "summary.json"
        if sj.exists():
            seed_summ.append(json.loads(sj.read_text()).get("methods", {}))
        mj = sd / "mechanism.json"
        if mj.exists():
            mech[sd.name] = json.loads(mj.read_text())
    if seed_summ:
        for m in rows:
            if all(m in s for s in seed_summ):
                temporal[LABEL.get(m, m)] = {
                    k: float(np.mean([s[m][k] for s in seed_summ]))
                    for k in ("pre_feedback_ll", "first_quarter_ll", "worst_day_ll")
                    if k in seed_summ[0][m]}

    return {"label": label, "dir": str(result_dir), "days": days,
            "table": table, "temporal": temporal, "mechanism": mech}


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    return f"{p:.3f}" if p >= 1e-3 else f"{p:.1e}"


def md_section(a: dict, delta_ni: float | None, decision: dict | None) -> str:
    L = [f"### {a['label']}  (D = {len(a['days'])} days: {a['days']})", ""]
    L += ["| row | seed-avg mean log loss | mean d vs OPS (day-wt) | imp-wt d | 95% CI (day bootstrap) | CI excl 0 | days won | sign p | worst-day d |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in a["table"]:
        if "mean_delta_vs_ops" not in r:
            L.append(f"| {r['method']} | {r['seed_avg_mean_ll']:.6f} | — | — | — | — | — | — | — |")
            continue
        L.append(f"| {r['method']} | {r['seed_avg_mean_ll']:.6f} | {r['mean_delta_vs_ops']:+.6f} | "
                 f"{r['imp_wt_delta_vs_ops']:+.6f} | [{r['ci95_lo']:+.6f}, {r['ci95_hi']:+.6f}] | "
                 f"{'yes' if r['ci_excludes_0'] else 'no'} | {r['n_days_won']}/{r['n_days']} | "
                 f"{fmt_p(r['sign_test_p'])} | {r['worst_day_delta']:+.6f} |")
    L.append("")
    if delta_ni is not None:
        L.append(f"Frozen `delta_NI = {delta_ni:.3e}`.")
    if decision is not None:
        L.append(f"**Decision: {decision.get('verdict', '?')}** "
                 f"(CI upper {decision.get('ci95_hi', decision.get('ap_ops_ci95', [None, None])[1])}).")
    L.append("")
    if a["temporal"]:
        L += ["Temporal (pooled log loss, mean over seeds):", "",
              "| row | pre-feedback | first quarter | worst day |", "|---|---|---|---|"]
        for name, t in a["temporal"].items():
            L.append(f"| {name} | {t.get('pre_feedback_ll', float('nan')):.6f} | "
                     f"{t.get('first_quarter_ll', float('nan')):.6f} | "
                     f"{t.get('worst_day_ll', float('nan')):.6f} |")
        L.append("")
    if a["mechanism"]:
        s0 = next(iter(a["mechanism"].values()))
        ap = s0.get("ap_ops", {})
        fw = ap.get("final_weights", {})
        dom = ap.get("dominant_block_counts", {})
        L += [f"AP-OPS mechanism (seed 0): final weights R/S/L = "
              f"{fw.get('R', 0):.3f} / {fw.get('S', 0):.3f} / {fw.get('L', 0):.3f}; "
              f"blocks where each dominates = {dom.get('R', 0)} / {dom.get('S', 0)} / {dom.get('L', 0)}; "
              f"weight range [{ap.get('weight_min', 0):.3f}, {ap.get('weight_max', 0):.3f}] "
              f"over {ap.get('n_meta_updates', 0)} updates.", ""]
    return "\n".join(L)


def nested_decisions(a: dict, summary: dict | None) -> str:
    """The 2026-09-06 spec's three decision rules."""
    by = {r["method"]: r for r in a["table"]}
    def d(name):  # mean day-wt delta vs OPS
        return by.get(name, {}).get("mean_delta_vs_ops")
    FLOOR = 2e-5   # materiality floor for "these two rows differ"
    L = ["**Decision rules:**", ""]
    do = d("OPS") if d("OPS") is not None else 0.0
    dr, dp, da = d("Reset-ONS"), d("Persistent-ONS"), d("AP-OPS")
    r_ci, p_ci, a_ci = by.get("Reset-ONS", {}), by.get("Persistent-ONS", {}), by.get("AP-OPS", {})
    if dr is not None and dp is not None:
        opt_helps = r_ci.get("ci95_hi", 1) < 0
        if dp - dr < -FLOOR:
            v = "SUPPORTED -- Persistent-ONS materially beats Reset-ONS"
        elif abs(dp - dr) <= FLOOR:
            v = ("indistinguishable from Reset-ONS -- the ONS optimizer, not persistence, carries the gain"
                 if opt_helps else "indistinguishable from Reset-ONS, and neither beats OPS")
        else:
            v = "NOT supported -- Reset-ONS is better"
        L.append(f"- Cross-day persistence: Persistent-ONS {dp:+.6f} vs Reset-ONS {dr:+.6f} "
                 f"(vs OPS; Reset-ONS CI excl 0: {opt_helps}) -> {v}.")
    if da is not None and dp is not None:
        if da - dp < -FLOOR and a_ci.get("ci95_hi", 1) < 0:
            v = "SUPPORTED -- AP-OPS materially beats the single persistent expert"
        elif abs(da - dp) <= FLOOR:
            v = "matches the single persistent expert (no material difference here)"
        else:
            v = "single persistent expert is at least as good"
        L.append(f"- Adaptive aggregation: AP-OPS {da:+.6f} vs Persistent-ONS {dp:+.6f} -> {v}.")
    if da is not None:
        beats_ops = a_ci.get("ci95_hi", 1) < 0
        L.append(f"- AP-OPS vs OPS: {da:+.6f}, CI [{a_ci.get('ci95_lo'):+.6f}, {a_ci.get('ci95_hi'):+.6f}], "
                 f"{a_ci.get('n_days_won')}/{a_ci.get('n_days')} origins -> {'beats OPS (CI excl 0)' if beats_ops else 'not significant'}.")
    if summary:
        f = summary.get("lambda_ap_gt0_fraction")
        sel = summary.get("lambda_ap_selected_per_origin", [])
        L.append(f"- Extension used: lambda_AP > 0 on {sum(1 for x in sel if x > 0)}/{len(sel)} origins "
                 f"(fraction {f:.2f}); per-origin lambda_AP = {sel}.")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", action="append", dest="dirs", required=True)
    ap.add_argument("--label", action="append", dest="labels", required=True)
    ap.add_argument("--selected", action="append", dest="selected", default=[],
                    help="apops_selected.json per dir (for delta_NI), same order; optional")
    ap.add_argument("--decision", action="append", dest="decision", default=[],
                    help="decision.json per dir, same order; optional")
    ap.add_argument("--nested", action="store_true",
                    help="4-method nested comparison (ops/reset_ons/persistent_ons/ap_ops), baseline=ops")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = NESTED_ROWS if args.nested else FIXED_ROWS
    baseline = "ops" if args.nested else "current_ops"
    sels = [json.loads(Path(p).read_text()) if p and Path(p).exists() else None for p in args.selected]
    decs = [json.loads(Path(p).read_text()) if p and Path(p).exists() else None for p in args.decision]
    hdr = "## Nested rolling-origin results" if args.nested else "## Results"
    blocks = [f"{hdr} (day-level inference: seeds averaged within day, then bootstrap over days)", ""]
    for i, (d, lab) in enumerate(zip(args.dirs, args.labels)):
        a = analyse(Path(d), lab, rows=rows, baseline=baseline)
        dni = sels[i]["decision_rule"]["delta_NI"] if i < len(sels) and sels[i] else None
        dec = decs[i] if i < len(decs) else None
        blocks.append(md_section(a, dni, dec))
        if args.nested:
            sm = Path(d) / "summary.json"
            blocks.append(nested_decisions(a, json.loads(sm.read_text()) if sm.exists() else None))
        pd.DataFrame(a["table"]).to_csv(Path(d) / "apops_day_level.csv", index=False)
    Path(args.out).write_text("\n".join(blocks) + "\n")
    print(f"wrote {args.out}")
    print("\n".join(blocks))


if __name__ == "__main__":
    main()
