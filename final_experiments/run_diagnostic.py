"""Decisive diagnostic (review comment 3): frozen V5 vs online DualTime,
under the *exact* final protocol.

Same data (full), same 3 seeds, same shared three-expert bank, and the
**same** frozen adaptive cross-day mixture ``q_{d,i}`` that ``run_final.py``
feeds OPS and DualTime-CTR. Four arms, all scored on the locked test days:

  long_only     the adaptive mixture q itself, no within-day correction
  ops           OPS on top of q            (identical to run_final's OPS row)
  dualtime      online DualTime-CTR on q   (identical to run_final's row)
  frozen_v5     the capacity-ladder V5 linear-interaction adapter, trained
                OFFLINE on the dev days and FROZEN for the whole test period
                -- only its history features h move within a day, not w.

``frozen_v5`` reuses ``withinday.adapters.V5Linear`` / ``withinday.train``
and the per-seed hyperparameters already frozen in
``withinday_experiments/<source>/seed<k>/_hpo/FROZEN.json`` (dev-only
selection, never re-tuned here), on a ``withinday.cache`` built from this
protocol's q -- so the only thing that differs between ``frozen_v5`` and
``dualtime`` is offline-frozen w vs daily-reset online w over
near-identical phi features.

**This is development evidence, not a fresh locked test.** The test days
have already been inspected for the headline table; the point of this run
is to localise *why* online DualTime-CTR underperforms, not to produce a
new confirmatory number. Any method change suggested by the result has to
be confirmed on a genuinely untouched stream (see comment 3's caution).

Also dumps, per seed, the realised adaptive-mixture weights per day for
eta in {selected, 0 (equal), 1e6 (follow-the-leader)} -- the evidence for
review comment 2 (what eta=1e6 actually does).

Writes ``final_experiments/<source>/diagnostic/`` with the same
``seed<k>/per_day_metrics.csv`` schema ``run_final.py`` uses, so
``day_level_stats.py`` runs on it directly.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from twoscale.calib import CalibConfig
from twoscale.data import load
from twoscale.longterm import build_bank
from twoscale.metrics import per_day_frame
from twoscale.splits import make_split
from twoscale_run import DATA_PATHS

from dualtime.online import DualTimeConfig
from methods import adaptive_q_by_day, dualtime_method, ops_method
from withinday.blocks import summary_dim, token_dim
from withinday.cache import build_cache
from withinday.daystats import day_summary
from withinday.train import DEFAULT_CFG, predict_records, train_variant

SEEDS = (0, 1, 2)
V5_VARIANT = "v5_linear"
# fallback if a per-seed withinday FROZEN.json is not found (it is the same
# for both datasets: DEFAULT_CFG with lr lowered to 3e-4, dev-selected)
V5_CFG_FALLBACK = {**DEFAULT_CFG, "lr": 3e-4, "weight_decay": 1e-5, "cross_dim": 32, "K": 16}
ADAPTER_TRAIN_FRAC = 0.7


def v5_cfg_for_seed(source: str, seed: int) -> dict:
    p = Path(f"withinday_experiments/{source}/seed{seed}/_hpo/FROZEN.json")
    if p.exists():
        d = json.loads(p.read_text())
        pv = d.get("per_variant", {}).get(V5_VARIANT, d)
        return {**DEFAULT_CFG, **{k: v for k, v in pv.items() if k in DEFAULT_CFG}}
    return dict(V5_CFG_FALLBACK)


def split_dev_days(dev_days: list[int]) -> tuple[list[int], list[int]]:
    """Same rule as withinday_run.py: first ~70% of dev days train the
    adapter, the rest early-stop it (at least one day each)."""
    n = max(1, int(round(len(dev_days) * ADAPTER_TRAIN_FRAC)))
    n = min(n, len(dev_days) - 1) if len(dev_days) > 1 else n
    return dev_days[:n], dev_days[n:] or dev_days[-1:]


def frozen_v5_records(ds, bank, q_by_day, eval_days, dev_days, test_days,
                      block_sec, delay_sec, m, seed, v5_cfg):
    cache = build_cache(ds, bank, q_by_day, eval_days, block_sec=block_sec,
                        delay_sec=delay_sec, m=m, seed=seed)
    adtr_days, addev_days = split_dev_days([d for d in dev_days if d in cache])
    a_dim, tok_dim, s_dim = m + 2, token_dim(m), summary_dim(m)
    cfg = {**v5_cfg, "seed": seed}
    model, dev_ll = train_variant(V5_VARIANT, [cache[d] for d in adtr_days],
                                  [cache[d] for d in addev_days],
                                  a_dim, tok_dim, s_dim, cfg=cfg)
    recs = predict_records(V5_VARIANT, model, [cache[d] for d in test_days if d in cache],
                           K=cfg["K"])
    return recs, {"adapter_train_days": adtr_days, "adapter_dev_days": addev_days,
                  "dev_logloss": float(dev_ll), "v5_cfg": cfg}


def weight_trace(bank, eval_days, etas: dict) -> list[dict]:
    """Realised mixture weights per day for several eta settings -- the
    empirical answer to 'what does eta=1e6 do' (comment 2)."""
    rows = []
    for tag, (eta, hl) in etas.items():
        _, w = adaptive_q_by_day(bank, eval_days, eta=eta, halflife=hl)
        for d in sorted(w):
            wd = w[d]
            top = max(wd, key=wd.get)
            rows.append({"eta_tag": tag, "eta": eta, "halflife": hl, "day": int(d),
                         "w_roll3": wd["roll3"], "w_roll7": wd["roll7"],
                         "w_expanding": wd["expanding"],
                         "argmax_horizon": top, "max_weight": wd[top]})
    return rows


def run_seed(source, data_path, sample_frac, n_features, warmup, n_jobs, seed, selected, out_dir):
    t = time.time()
    ds = load(source, data_path, n_features=n_features, sample_frac=sample_frac, seed=seed)
    split = make_split(ds.n_days, warmup=warmup)
    dev_days = sorted(int(d) for d in split.dev_days)
    test_days = sorted(int(d) for d in split.test_days)
    print(f"  seed {seed}: {len(ds.y):,} rows | dev {dev_days} | test {test_days}", flush=True)

    bank = build_bank(ds, split.eval_days, seed=seed, n_jobs=n_jobs)
    eval_days = sorted(bank)

    mix = selected["shared_mixture"]
    q_by_day, _ = adaptive_q_by_day(bank, eval_days, eta=mix["eta"], halflife=mix["halflife"])

    block_sec, delay_sec = selected["block_sec"], selected["delay_sec"]
    m = selected["dualtime"]["m"]

    methods = {}
    methods["long_only"] = [{"day": d, "y": bank[d].y, "p": q_by_day[d],
                             "sec_in_day": bank[d].sec_in_day} for d in eval_days]

    ops_cfg = selected["ops"]
    cfg = CalibConfig(B=ops_cfg["B"], eta0=ops_cfg["eta0"], eta_schedule=ops_cfg["schedule"],
                      update="block", block_sec=block_sec, delay_sec=delay_sec, platt=True)
    methods["ops"], _ = ops_method(bank, eval_days, q_by_day, cfg)

    dt = selected["dualtime"]
    dt_cfg = DualTimeConfig(block_sec=block_sec, delay_sec=delay_sec, m=dt["m"],
                            cross_dim=dt["cross_dim"], B_w=dt["B_w"])
    methods["dualtime"] = dualtime_method(ds, bank, eval_days, q_by_day, dt_cfg,
                                          sketch_seed=seed, hash_seed=seed)

    v5_cfg = v5_cfg_for_seed(source, seed)
    methods["frozen_v5"], v5_info = frozen_v5_records(
        ds, bank, q_by_day, eval_days, dev_days, test_days, block_sec, delay_sec, m, seed, v5_cfg)

    test_set = set(test_days)
    per_day_rows = []
    for name, recs in methods.items():
        for r in per_day_frame([x for x in recs if x["day"] in test_set]):
            per_day_rows.append({"method": name, **r})

    seed_out = out_dir / f"seed{seed}"
    seed_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_day_rows).to_csv(seed_out / "per_day_metrics.csv", index=False)

    etas = {"selected": (mix["eta"], mix["halflife"]), "equal": (0.0, mix["halflife"]),
            "ftl_1e6": (1e6, mix["halflife"])}
    pd.DataFrame(weight_trace(bank, eval_days, etas)).to_csv(seed_out / "mixture_weights.csv", index=False)

    (seed_out / "summary.json").write_text(json.dumps({
        "seed": seed, "n_rows": int(len(ds.y)), "test_days": test_days,
        "frozen_v5": v5_info, "mixture": mix, "runtime_s": time.time() - t,
    }, indent=2, default=float))

    test_records = {name: [x for x in recs if x["day"] in test_set] for name, recs in methods.items()}
    del ds, bank
    return test_records


def paired_day_deltas(m_recs, base_recs):
    from twoscale.metrics import paired_day_diffs
    _, d = paired_day_diffs(m_recs, base_recs)
    return d.tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["criteo", "avazu"], required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--n-features", type=int, default=2 ** 18)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--config", required=True, help="selected_configs.json (frozen)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    selected = json.loads(Path(args.config).read_text())
    assert selected["source"] == args.source
    warmup = args.warmup if args.warmup is not None else (4 if args.source == "criteo" else 3)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    method_order = ["long_only", "ops", "dualtime", "frozen_v5"]
    all_recs = {m: [] for m in method_order}
    for seed in SEEDS:
        tr = run_seed(args.source, args.data or DATA_PATHS[args.source], args.sample_frac,
                      args.n_features, warmup, args.n_jobs, seed, selected, out)
        for m in method_order:
            all_recs[m].append((seed, tr[m]))

    # ---- day-level aggregate: average seeds first, then inference over days
    rows = []
    for base in ("long_only", "ops"):
        for m in method_order:
            if m == base:
                continue
            # seed-averaged per-day loss delta
            per_day = {}
            for (s_m, recs_m), (s_b, recs_b) in zip(all_recs[m], all_recs[base]):
                assert s_m == s_b
                from twoscale.metrics import paired_day_diffs
                days, d = paired_day_diffs(recs_m, recs_b)
                for dd, val in zip(days, d):
                    per_day.setdefault(int(dd), []).append(val)
            deltas = [float(np.mean(v)) for _, v in sorted(per_day.items())]
            s = day_summary(deltas, seed=0)
            rows.append({"baseline": base, "method": m, "n_days": s["n_days"],
                         "mean_delta": s["mean_delta"], "median_delta": s["median_delta"],
                         "ci95_lo": s["ci95_lo"], "ci95_hi": s["ci95_hi"],
                         "ci_excludes_0": bool(s["ci95_hi"] < 0 or s["ci95_lo"] > 0),
                         "n_days_won": s["n_days_won"], "sign_test_p": s["sign_test_p"]})
    agg = pd.DataFrame(rows)
    agg.to_csv(out / "diagnostic_day_level.csv", index=False)

    # mean imp-weighted log loss per method (per seed then averaged)
    from twoscale.metrics import impression_weighted_logloss
    head = []
    for m in method_order:
        vals = [impression_weighted_logloss(recs) for _, recs in all_recs[m]]
        head.append({"method": m, "mean_imp_wt_ll": float(np.mean(vals)),
                     "std_across_seeds": float(np.std(vals))})
    pd.DataFrame(head).sort_values("mean_imp_wt_ll").to_csv(out / "diagnostic_headline.csv", index=False)

    (out / "summary.json").write_text(json.dumps({
        "source": args.source, "config_used": str(args.config), "seeds": list(SEEDS),
        "note": "DEVELOPMENT EVIDENCE ONLY - test days already inspected; not a fresh locked test",
        "runtime_s": time.time() - t0,
    }, indent=2, default=float))

    print("\n=== diagnostic headline (mean imp-wt log loss, 3 seeds) ===", flush=True)
    print(pd.DataFrame(head).sort_values("mean_imp_wt_ll").to_string(index=False), flush=True)
    print("\n=== day-level deltas (seeds averaged first) ===", flush=True)
    print(agg.to_string(index=False), flush=True)
    print(f"\nruntime {time.time() - t0:.1f}s -> {out}/", flush=True)


if __name__ == "__main__":
    main()
