"""Integrated-pipeline leakage and identity checks for TTAM (plan section 6:
"Before full runs, verify causal invariance to future-label changes,
cross-midnight label maturation, simplex and parameter constraints, and
the lambda = 0 AP-OPS-to-OPS identity. Verify the deployed-weight memory
update.").

Runs on a small Criteo sample so it is fast; every check is structural
(exact equalities / bounds), not statistical, so the sample size does not
matter.  Writes ``final_experiments/ttam_tests.txt``.

  1. lambda_AP = 0  =>  AP-OPS on a stream == daily-reset OPS on that
     stream, prediction by prediction (both q_fixed and q_amgtp streams)
  2. simplex: the fitted fixed-mixture weight vector is nonnegative and
     sums to one; AP-OPS meta weights stay on the simplex throughout
  3. parameter bounds: the persistent Platt experts keep (a, b) inside
     [0.2, 5] x [-0.25, 0.25]
  4. causal invariance: flipping labels on / after the last prefix day
     leaves every earlier TTAM prediction byte-identical
  5. cross-midnight maturation: flipping the last-block labels of an
     interior day (a) leaves that day's predictions unchanged and
     (b) changes a later block of the following day
  6. deployed-weight memory update: AMG-TP's ``m_t`` equals
     ``(1 - rho) m_{t-1} + rho * mean_x pi_deployed_t`` using the pi that
     was actually deployed for day t (recomputed here from the trace),
     i.e. m is *not* re-derived from the post-update gate
  7. determinism: identical inputs reproduce identical variant outputs
"""
from __future__ import annotations

import argparse
from dataclasses import replace as dc_replace
from pathlib import Path

import numpy as np

from twoscale.data import load

from final_experiments.ttam.bank import build_bank5, HORIZONS5
from final_experiments.ttam.build_banks import save_context, load_context
from final_experiments.ttam.fixed_mix import fit_simplex_weights, fixed_mixture_q
from final_experiments.ttam.amgtp import AMGTPConfig, run_amgtp5
from final_experiments.ttam import method as V
from final_experiments.ttam.method import apops_config, apops_on

LINES = []
OPS_HP = {"B": 0.25, "eta0": 0.3, "schedule": "const", "a_bounds": (0.2, 5.0)}
SEL = {"a_bounds": [0.2, 5.0], "b_bounds": [-0.25, 0.25]}


def check(name, cond):
    LINES.append(f"[{'PASS' if cond else 'FAIL'}] {name}")
    print(f"  {'PASS' if cond else 'FAIL'}  {name}", flush=True)
    return bool(cond)


def _cat(recs):
    return np.concatenate([np.asarray(r["p"], float) for r in sorted(recs, key=lambda r: r["day"])])


def run(data_path, sample_frac, n_features, n_jobs):
    block_sec, delay_sec = 900, 1800
    ds = load("criteo", data_path, n_features=n_features, sample_frac=sample_frac, seed=0)
    days = list(range(1, 13))
    bank = build_bank5(ds, days, seed=0, n_jobs=n_jobs, verbose=False)
    tmp = Path("/tmp/_ttam_ctx.npz")
    save_context(tmp, ds, days)
    ctx = load_context(tmp)
    hist = [d for d in days if d < 12]
    inner = hist[-3:]

    w = fit_simplex_weights(bank, inner)
    q_fixed = fixed_mixture_q(bank, days, w)
    q_expanding = {int(d): np.asarray(bank[d].preds["expanding"], float) for d in days}
    acfg = AMGTPConfig(rho=0.3, delay_sec=delay_sec, epochs_per_day=2)
    q_amgtp, trace = run_amgtp5(bank, ctx, days, acfg, seed=0)

    # 1. lambda_AP = 0 identity -- on every frozen calibration input stream
    #    (ops arm: q_fixed ; 2x2 ablation: q_expanding and q_amgtp)
    for label, q in (("q_fixed", q_fixed), ("q_expanding", q_expanding), ("q_amgtp", q_amgtp)):
        ops_recs = V.ops(bank, days, q, OPS_HP, block_sec, delay_sec)
        ac0 = apops_config(SEL, block_sec, delay_sec, ons_lam=0.1, ons_eta=0.25, eta_m=100.0,
                           lambda_ap=0.0, tau_h=4.0, persist_hl_h=4.0)
        ap0, _ = apops_on(bank, days, q, ac0, OPS_HP)
        check(f"lambda_AP=0 -> AP-OPS == daily-reset OPS on {label}",
              np.array_equal(_cat(ap0), _cat(ops_recs)))

    # 2. simplex
    check("fixed-mixture weights nonnegative and sum to 1",
          abs(w.sum() - 1.0) < 1e-9 and w.min() >= -1e-12)
    ac = apops_config(SEL, block_sec, delay_sec, ons_lam=0.1, ons_eta=0.25, eta_m=100.0,
                      lambda_ap=0.5, tau_h=4.0, persist_hl_h=4.0)
    _, info = apops_on(bank, days, q_amgtp, ac, OPS_HP)
    wp = info["weight_path"]
    check(f"AP-OPS meta weights on the simplex across {len(wp)} updates",
          all(abs(r["R"] + r["S"] + r["L"] - 1.0) < 1e-9 and min(r["R"], r["S"], r["L"]) >= 0.0
              for r in wp))

    # 3. parameter bounds
    from apops.method import build_ap_experts
    _, traces = build_ap_experts(bank, days, q_amgtp, ac, OPS_HP, learn_slope=True)
    ok = True
    for nm in ("S", "L"):
        for t in traces[nm]:
            if not (0.2 - 1e-9 <= t["a_end"] <= 5.0 + 1e-9 and -0.25 - 1e-9 <= t["b_end"] <= 0.25 + 1e-9):
                ok = False
    check("persistent Platt experts keep (a, b) in [0.2,5] x [-0.25,0.25]", ok)

    # 4. causal invariance to future-label changes
    last = max(days)
    bank2 = {d: bank[d] for d in bank}
    db = bank[last]
    y2 = 1.0 - np.asarray(db.y, float)
    bank2[last] = dc_replace(db, y=y2.astype(db.y.dtype))
    ttam_a, _ = apops_on(bank, days, q_amgtp, ac, OPS_HP)
    # q_amgtp does not depend on day-`last` labels until after it is emitted;
    # rebuild it on bank2 to be faithful, then compare all days < last
    qa2, _ = run_amgtp5(bank2, ctx, days, acfg, seed=0)
    ttam_b, _ = apops_on(bank2, days, qa2, ac, OPS_HP)
    pre = all(np.array_equal(np.asarray(ra["p"]), np.asarray(rb["p"]))
              for ra, rb in zip(sorted(ttam_a, key=lambda r: r["day"]),
                                sorted(ttam_b, key=lambda r: r["day"])) if ra["day"] < last)
    check("flipping the last prefix day's labels leaves every earlier TTAM prediction identical", pre)

    # 5. cross-midnight maturation (persistent ONS expert stream)
    from apops.experts import replay_ons_stream
    d0 = days[len(days) // 2]
    d1 = days[days.index(d0) + 1]
    n_blocks = int(np.ceil(86400 / block_sec))
    last_blk = np.minimum((bank[d0].sec_in_day // block_sec).astype(int), n_blocks - 1) == (n_blocks - 1)
    if last_blk.any():
        b3 = {d: bank[d] for d in bank}
        y3 = np.asarray(bank[d0].y, float).copy()
        y3[last_blk] = 1.0 - y3[last_blk]
        b3[d0] = dc_replace(bank[d0], y=y3.astype(bank[d0].y.dtype))
        cfg_ons = ac.ons_cfg(4.0)
        sa, _ = replay_ons_stream(q_amgtp, bank, days, cfg_ons)
        sb, _ = replay_ons_stream(q_amgtp, b3, days, cfg_ons)
        pa = {r["day"]: np.asarray(r["p"]) for r in sa}
        pb = {r["day"]: np.asarray(r["p"]) for r in sb}
        blk1 = np.minimum((bank[d1].sec_in_day // block_sec).astype(int), n_blocks - 1)
        late_d1 = blk1 > int(np.ceil(delay_sec / block_sec))
        check("cross-midnight label leaves its own day's ONS predictions unchanged",
              np.array_equal(pa[d0], pb[d0]))
        check("cross-midnight label updates the persistent expert on the next day",
              late_d1.any() and not np.allclose(pa[d1][late_d1], pb[d1][late_d1], atol=1e-12))

    # 6. deployed-weight memory update
    rho = acfg.rho
    m_seq = [np.array([trace[0]["m_state"][h] for h in HORIZONS5])]
    ok = True
    for i, row in enumerate(trace):
        pi_dep = np.array([row["mean_pi"][h] for h in HORIZONS5])
        m_next = (1 - rho) * m_seq[-1] + rho * pi_dep
        m_seq.append(m_next)
        if i + 1 < len(trace):
            m_reported_next = np.array([trace[i + 1]["m_state"][h] for h in HORIZONS5])
            if not np.allclose(m_next, m_reported_next, atol=1e-6):
                ok = False
    check("AMG-TP memory m_t = (1-rho) m_{t-1} + rho * mean_x pi_deployed_t", ok)

    # 7. determinism
    qa_again, _ = run_amgtp5(bank, ctx, days, acfg, seed=0)
    ttam_again, _ = apops_on(bank, days, qa_again, ac, OPS_HP)
    check("identical inputs reproduce identical TTAM outputs",
          np.array_equal(_cat(ttam_a), _cat(ttam_again)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/criteo_attribution_dataset.tsv.gz")
    ap.add_argument("--sample-frac", type=float, default=0.03)
    ap.add_argument("--n-features", type=int, default=2 ** 16)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--out", default="final_experiments/ttam_tests.txt")
    args = ap.parse_args()
    run(args.data, args.sample_frac, args.n_features, args.n_jobs)
    n_pass = sum(l.startswith("[PASS]") for l in LINES)
    n_fail = sum(l.startswith("[FAIL]") for l in LINES)
    LINES.append(f"\n{n_pass} passed, {n_fail} failed")
    Path(args.out).write_text("\n".join(LINES) + "\n")
    print(f"\n{n_pass} passed, {n_fail} failed -> {args.out}", flush=True)
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
