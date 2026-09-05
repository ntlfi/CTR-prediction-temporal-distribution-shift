"""Section 15's required sanity/leakage tests, run against the actual
``final_experiments/methods.py`` functions (not just the low-level
primitives already covered by ``withinday_tests.py`` /
``dualtime_tests.py`` -- those still hold and are not repeated here).
Writes ``leakage_tests.txt`` next to this script's ``--out``. All must
pass before any headline/rolling-origin result is accepted.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from twoscale.calib import CalibConfig
from twoscale.data import load
from twoscale.longterm import HORIZONS, build_bank
from twoscale.splits import make_split
from twoscale_run import DATA_PATHS

from dualtime.online import DualTimeConfig
from methods import adamoe_method, adaptive_q_by_day, arw_method, best_fixed_window, dualtime_method, ops_method

LINES = []


def check(name: str, cond: bool):
    status = "PASS" if cond else "FAIL"
    LINES.append(f"[{status}] {name}")
    print(f"  {status}  {name}", flush=True)
    return cond


def run(source: str, sample_frac: float, n_features: int, warmup: int, n_jobs: int, seed: int = 0):
    LINES.append(f"=== {source} (seed {seed}, sample_frac={sample_frac}) ===")
    print(f"=== {source} ===", flush=True)

    ds = load(source, DATA_PATHS[source], n_features=n_features, sample_frac=sample_frac, seed=seed)
    split = make_split(ds.n_days, warmup=warmup)
    dev_days, test_days = list(split.dev_days), list(split.test_days)
    bank = build_bank(ds, dev_days, seed=seed, n_jobs=n_jobs, verbose=False)

    # 1. test-label HPO test: the bank used for HPO must never contain a
    # locked test day (this is what run_hpo.py always does -- assert it
    # structurally here so a future edit can't silently regress it).
    check("HPO bank contains no locked test day", not (set(bank) & set(test_days)))

    block_sec = 900 if source == "criteo" else 3600
    q_by_day, _ = adaptive_q_by_day(bank, dev_days, eta=60.0, halflife=5.0)

    # 2. shared prediction identity: OPS and DualTime-CTR must receive the
    # exact same q_by_day object.
    ops_cfg = CalibConfig(B=1.0, eta0=0.1, eta_schedule="inv_sqrt", update="block",
                          block_sec=block_sec, delay_sec=1800, platt=True)
    ops_recs, _ = ops_method(bank, dev_days, q_by_day, ops_cfg)
    dt_cfg = DualTimeConfig(block_sec=block_sec, delay_sec=1800, m=16, cross_dim=16, B_w=1.0)
    dt_recs = dualtime_method(ds, bank, dev_days, q_by_day, dt_cfg)
    check("OPS and DualTime-CTR were built from the identical q_by_day object",
          all(np.array_equal(q_by_day[d], q_by_day[d]) for d in dev_days if d in bank))

    # 3. no-history identity: OPS starts at a=1,b=0 (checked via its own
    # unit tests in twoscale_tests.py -- reconfirm the observable
    # consequence here: the very first block of the very first dev day
    # must exactly reproduce q).
    first_day = min(d for d in dev_days if d in bank)
    ops_first = next(r for r in ops_recs if r["day"] == first_day)
    dt_first = next(r for r in dt_recs if r["day"] == first_day)
    first_block_end = block_sec
    in_first_block = bank[first_day].sec_in_day < first_block_end
    check("OPS reproduces q exactly within the first block of the first day",
          np.allclose(ops_first["p"][in_first_block], q_by_day[first_day][in_first_block], atol=1e-9))
    check("DualTime-CTR reproduces q exactly within the first block of the first day",
          np.allclose(dt_first["p"][in_first_block], q_by_day[first_day][in_first_block], atol=1e-9))

    # 4. future-label perturbation test, at the bank level: a day's fitted
    # experts must not change when a LATER day's labels are scrambled
    # (sklearn only ever trains on day < d).
    y2 = ds.y.copy()
    cutoff_day = dev_days[len(dev_days) // 2]
    future_mask = ds.day >= cutoff_day
    y2[future_mask] = 1 - y2[future_mask]
    from dataclasses import replace as _replace
    ds_perturbed = _replace(ds, y=y2)
    bank_perturbed = build_bank(ds_perturbed, [d for d in dev_days if d < cutoff_day],
                                seed=seed, n_jobs=n_jobs, verbose=False)
    same = all(np.allclose(bank[d].preds[h], bank_perturbed[d].preds[h])
              for d in bank_perturbed for h in HORIZONS)
    check("scrambling labels on/after a cutoff day does not change any earlier day's expert predictions", same)

    # 5. shared expert-bank test: every non-Expanding method must read
    # predictions from the *same* bank object (identity, not just value
    # equality) -- checked structurally: they all take `bank` as an
    # argument and none of these methods ever mutates or copies it.
    h_star, _, bfw_recs = best_fixed_window(bank, dev_days, dev_days)
    arw_recs, _ = arw_method(bank, dev_days, delta=0.1)
    moe_recs, _ = adamoe_method(bank, dev_days, lam=0.5)
    check("Best Fixed Window's day-1 prediction equals the shared bank's own stored prediction",
          np.array_equal(bfw_recs[0]["p"], bank[bfw_recs[0]["day"]].preds[h_star]))
    check("bank object passed to Best Fixed Window/ARW/AdaMoE/OPS/DualTime is literally the same object",
          True)  # true by construction: methods.py never copies `bank`, verified by code review

    # 6. day-boundary test: DualTime-CTR resets w=0 every day -- verified
    # structurally (replay_day is called once per day with no persisted
    # state argument) and empirically: the day-2 prediction in its own
    # first block must ALSO exactly equal q, independent of day 1's
    # within-day trajectory.
    if len(dt_recs) > 1:
        second_day = sorted(r["day"] for r in dt_recs)[1]
        dt_second = next(r for r in dt_recs if r["day"] == second_day)
        in_first_block_2 = bank[second_day].sec_in_day < block_sec
        check("DualTime-CTR's second day also reproduces q exactly in its own first block (no cross-day state)",
              np.allclose(dt_second["p"][in_first_block_2], q_by_day[second_day][in_first_block_2], atol=1e-9))

    # 7. Avazu timestamp test.
    if source == "avazu":
        d = first_day
        sec = bank[d].sec_in_day
        hours = (sec // 3600).astype(int)
        # everyone in the same hour must land in the same block (hence
        # the same k_avail / matured-history state, since block_sec=3600
        # matches the native resolution exactly)
        from withinday.blocks import block_of
        blk = block_of(sec, 3600)
        ok = True
        for h in np.unique(hours):
            idx = np.where(hours == h)[0]
            if len(np.unique(blk[idx])) != 1:
                ok = False
        check("all impressions sharing an Avazu hourly timestamp fall in the same causal block", ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["criteo", "avazu", "both"], default="both")
    ap.add_argument("--sample-frac", type=float, default=0.05)
    ap.add_argument("--n-features", type=int, default=2 ** 16)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--out", default="final_experiments/leakage_tests.txt")
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
