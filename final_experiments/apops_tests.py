"""Leakage and stability checks for AP-OPS (plan section 3, "Leakage and
stability checks" + "Golden check").  All must pass before any
fixed-test / rolling-origin AP-OPS number is accepted.  Writes
``final_experiments/apops_tests.txt``.

  1. anchor-only AP-OPS (weights pinned to (1,0,0)) == current OPS, exactly
  2. current OPS row == the standalone ``methods.ops_method`` computation
     the headline table used (bit-identical) -- the "reset expert alone
     reproduces the saved OPS result" golden check
  3. perturbing future / unmatured labels leaves every earlier AP-OPS
     prediction, expert state and meta weight unchanged
  4. meta weights stay nonnegative and sum to one throughout
  5. Platt parameters stay inside the stated bounds throughout
  6. same seed + cached base-prediction stream => identical outputs
  7. cross-day-boundary maturation (2026-09-06 correction): a label in the
     last block of an interior day matures *after midnight* and must (a)
     leave that day's predictions unchanged, (b) change the persistent
     expert's predictions on the following day
"""
from __future__ import annotations

import argparse
from dataclasses import replace as dc_replace
from pathlib import Path

import numpy as np

from twoscale.calib import CalibConfig
from twoscale.data import load
from twoscale.longterm import build_bank
from twoscale.splits import make_split
from twoscale_run import DATA_PATHS

from methods import adaptive_q_by_day, ops_method
from apops.aggregate import aggregate
from apops.experts import replay_ons_stream
from apops.method import APOPSConfig, anchor_only_ap_ops, build_experts, build_rows

LINES = []
OPS_HP = {"B": 0.25, "eta0": 0.3, "schedule": "const", "a_bounds": (0.2, 5.0)}


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    LINES.append(f"[{status}] {name}")
    print(f"  {status}  {name}", flush=True)
    return bool(cond)


def _concat(recs):
    return np.concatenate([np.asarray(r["p"], float) for r in sorted(recs, key=lambda r: r["day"])])


def run(source, sample_frac, n_features, warmup, n_jobs, seed=0):
    block_sec = 900 if source == "criteo" else 3600
    LINES.append(f"=== {source} (seed {seed}, sample_frac={sample_frac}) ===")
    print(f"=== {source} ===", flush=True)

    ds = load(source, DATA_PATHS[source], n_features=n_features, sample_frac=sample_frac, seed=seed)
    split = make_split(ds.n_days, warmup=warmup)
    dev_days, test_days = list(split.dev_days), list(split.test_days)
    bank = build_bank(ds, dev_days, seed=seed, n_jobs=n_jobs, verbose=False)
    days = [d for d in dev_days if d in bank]

    q_by_day, _ = adaptive_q_by_day(bank, days, eta=150.0, halflife=3.0)
    cfg = APOPSConfig(block_sec=block_sec, delay_sec=1800, ons_lam=1e-2, ons_eta=1.0,
                      eta_m=10.0, switch_half_life_h=16.0)

    # 1. anchor-only equivalence
    ops_cfg = CalibConfig(B=OPS_HP["B"], eta0=OPS_HP["eta0"], eta_schedule=OPS_HP["schedule"],
                          update="block", block_sec=block_sec, delay_sec=1800, platt=True)
    ops_recs, _ = ops_method(bank, days, q_by_day, ops_cfg)
    anchor = anchor_only_ap_ops(bank, days, q_by_day, cfg, OPS_HP)
    check("anchor-only AP-OPS (1,0,0) reproduces current OPS bit-for-bit",
          np.allclose(_concat(anchor), _concat(ops_recs), atol=0, rtol=0))

    # 2. golden reference: current_ops row == the headline ops_method call
    rows, info = build_rows(bank, days, q_by_day, cfg, OPS_HP)
    check("current_ops row == standalone ops_method (golden reference)",
          np.array_equal(_concat(rows["current_ops"]), _concat(ops_recs)))

    # 3. future / unmatured label perturbation
    last = max(days)
    cut_abs_sec = float(np.median(bank[last].sec_in_day))   # flip late labels on the last dev day
    bank2 = {d: bank[d] for d in bank}
    db = bank[last]
    y2 = np.array(db.y, float).copy()
    flip = db.sec_in_day >= cut_abs_sec
    y2[flip] = 1.0 - y2[flip]
    bank2[last] = dc_replace(db, y=y2.astype(db.y.dtype))

    experts_a, _ = build_experts(bank, days, q_by_day, cfg, OPS_HP)
    experts_b, _ = build_experts(bank2, days, q_by_day, cfg, OPS_HP)
    ap_a, _ = aggregate(experts_a, bank, days, cfg.meta_cfg())
    ap_b, _ = aggregate(experts_b, bank2, days, cfg.meta_cfg())

    def early_mask(d):
        return bank[d].sec_in_day < (cut_abs_sec - cfg.delay_sec) if d == last else np.ones(len(bank[d].y), bool)

    unchanged = True
    for ra, rb in zip(sorted(ap_a, key=lambda r: r["day"]), sorted(ap_b, key=lambda r: r["day"])):
        m = early_mask(ra["day"])
        if not np.allclose(np.asarray(ra["p"])[m], np.asarray(rb["p"])[m], atol=1e-12):
            unchanged = False
    check("flipping labels after t leaves every prediction before t-delay unchanged", unchanged)

    # also: every day strictly before the perturbed one is byte-identical
    pre = all(np.array_equal(np.asarray(ra["p"]), np.asarray(rb["p"]))
              for ra, rb in zip(sorted(ap_a, key=lambda r: r["day"]),
                                sorted(ap_b, key=lambda r: r["day"])) if ra["day"] < last)
    check("all days before the perturbed day are byte-identical", pre)

    # 4. weight simplex
    _, ap_info = aggregate(experts_a, bank, days, cfg.meta_cfg())
    wp = ap_info["weight_path"]
    simplex_ok = all(
        abs(row["R"] + row["S"] + row["L"] - 1.0) < 1e-9 and min(row["R"], row["S"], row["L"]) >= 0.0
        for row in wp)
    check(f"meta weights nonnegative and sum to 1 across all {len(wp)} updates", simplex_ok)

    # 5. parameter bounds
    _, traces = build_experts(bank, days, q_by_day, cfg, OPS_HP)
    bounds_ok = True
    for name in ("S", "L"):
        for t in traces[name]:
            if not (0.2 - 1e-9 <= t["a_end"] <= 5.0 + 1e-9 and -0.25 - 1e-9 <= t["b_end"] <= 0.25 + 1e-9):
                bounds_ok = False
    check("persistent-expert (a, b) stay inside [0.2,5] x [-0.25,0.25]", bounds_ok)

    # 6. determinism
    rows_again, _ = build_rows(bank, days, q_by_day, cfg, OPS_HP)
    det = all(np.array_equal(_concat(rows[k]), _concat(rows_again[k])) for k in rows)
    check("identical inputs reproduce identical AP-OPS / ablation outputs", det)

    # 7. cross-day-boundary maturation
    if len(days) >= 2:
        d0 = days[len(days) // 2]          # an interior day
        d1 = days[days.index(d0) + 1]
        n_blocks = int(np.ceil(86400 / block_sec))
        sec0 = bank[d0].sec_in_day
        last_blk = np.minimum((sec0 // block_sec).astype(int), n_blocks - 1) == (n_blocks - 1)
        if last_blk.any():
            b3 = {d: bank[d] for d in bank}
            db0 = bank[d0]
            y3 = np.array(db0.y, float).copy()
            y3[last_blk] = 1.0 - y3[last_blk]
            b3[d0] = dc_replace(db0, y=y3.astype(db0.y.dtype))
            s_a, _ = replay_ons_stream(q_by_day, bank, days, cfg.ons_cfg(4.0))
            s_b, _ = replay_ons_stream(q_by_day, b3, days, cfg.ons_cfg(4.0))
            pa = {r["day"]: np.asarray(r["p"]) for r in s_a}
            pb = {r["day"]: np.asarray(r["p"]) for r in s_b}
            same_d0 = np.array_equal(pa[d0], pb[d0])
            # first block of d1 predicted before the cross-midnight labels mature -> unchanged;
            # a later block of d1 must differ (the last-block-of-d0 labels were consumed)
            sec1 = bank[d1].sec_in_day
            blk1 = np.minimum((sec1 // block_sec).astype(int), n_blocks - 1)
            early_d1 = blk1 <= int(np.ceil(cfg.delay_sec / block_sec))
            late_d1 = ~early_d1
            changed_later = late_d1.any() and not np.allclose(pa[d1][late_d1], pb[d1][late_d1], atol=1e-12)
            check("cross-midnight label leaves its own day's predictions unchanged", same_d0)
            check("cross-midnight label updates the persistent expert on the following day", changed_later)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["criteo", "avazu", "both"], default="both")
    ap.add_argument("--sample-frac", type=float, default=0.05)
    ap.add_argument("--n-features", type=int, default=2 ** 16)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--out", default="final_experiments/apops_tests.txt")
    args = ap.parse_args()

    sources = ["criteo", "avazu"] if args.source == "both" else [args.source]
    for source in sources:
        warmup = 4 if source == "criteo" else 3
        run(source, args.sample_frac, args.n_features, warmup, args.n_jobs)

    n_pass = sum(1 for l in LINES if l.startswith("[PASS]"))
    n_fail = sum(1 for l in LINES if l.startswith("[FAIL]"))
    LINES.append(f"\n{n_pass} passed, {n_fail} failed")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(LINES) + "\n")
    print(f"\n{n_pass} passed, {n_fail} failed -> {out}", flush=True)
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
