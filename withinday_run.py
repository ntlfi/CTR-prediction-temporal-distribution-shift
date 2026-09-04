"""Within-day capacity ladder -- one (dataset, seed) evaluation cell.

Runs the protocol of ``CTR_Within_Day_Capacity_Ladder_Experiment_Plan.pdf``:

  1. load data + fit the twoscale long-term bank (identical backbone/base
     predictor for every candidate, plan section 2.3)
  2. Stage A: build the causal replay cache (block tokens, summaries,
     current-impression input) for the twoscale dev+test days
  3. Stage B: train each ladder candidate (V1-V5) on the early dev days,
     early-stopping on the later dev days, together with its own
     no-history / shuffled-chronology / no-context-interaction /
     no-residual-sketch / label-free-history ablations (plan section 7)
  4. Decision rules (section 8): gate each candidate, then apply the
     parsimony rule among the ones that pass
  5. Stage C: if a candidate was selected, replay it (and the required
     baselines) on the locked test days -- opened once.

Defaults are the plan's fixed default sizes (section 3), not a tuned grid;
``--config`` overrides them. A follow-up ``withinday_hpo.py`` doing the
plan's small validation-grid search (section 5.2) is not implemented here.

Writes everything to ``--out`` as CSV + summary.json.
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
from twoscale.longterm import adaptive_weights, build_bank, long_term_predictions
from twoscale.methods import build_suite
from twoscale.metrics import (bootstrap_paired_ci, days_won, impression_weighted_logloss,
                              paired_day_diffs, per_day_frame)
from twoscale.splits import make_split

from withinday.adapters import VARIANTS
from withinday.blocks import summary_dim, token_dim
from withinday.cache import build_cache
from withinday.train import DEFAULT_CFG, predict_records, train_variant

DATA_PATHS = {
    "criteo": "/insomnia001/home/tn2447/data/criteo/criteo_attribution_dataset.tsv.gz",
    "avazu": "/insomnia001/home/tn2447/data/avazu/Avazu_x4.zip",
}
CALIB_CONFIG = dict(B=1.0, eta0=0.1, eta_schedule="inv_sqrt", update="block",
                    block_sec=1800, delay_sec=1800, eps=1e-5, init_b=0.0, carryover_rho=0.0)

# plan section 7: the 5 required ablations, and which kwarg to train_variant() realizes them
ABLATION_KWARGS = {
    "no_history": dict(ablation="no_history"),
    "shuffled_chronology": dict(ablation="shuffled_chronology"),
    "no_context_interaction": dict(zero_query=True),
    "no_residual_sketch": dict(ablation="no_residual_sketch"),
    "label_free_history": dict(ablation="label_free_history"),
}
# ladder order, most to least complex (used for the parsimony rule, which
# prefers the *last* entry -- simplest -- within 1 SE of the best)
LADDER_SIMPLE_TO_COMPLEX = tuple(reversed(VARIANTS))


def paired(recs_a, recs_b):
    _, deltas = paired_day_diffs(recs_a, recs_b)
    m, lo, hi = bootstrap_paired_ci(deltas)
    frac, won, tot = days_won(recs_a, recs_b)
    return {"mean_delta": m, "ci95": [lo, hi], "days_won_frac": frac,
           "days_won": won, "days_total": tot, "significant_below_zero": hi < 0}


def train_one(name, caches_train, caches_dev, a_dim, tok_dim, summ_dim, cfg, m,
             ablation=None, zero_query=False, seed=0, verbose=False):
    """Train variant ``name`` under one condition (normal, or one of the 5
    ablations) and return ``(model, dev_records)``."""
    kw = dict(ablation=ablation, zero_query=zero_query)
    # ablated training must see the *same* ablated inputs every forward pass,
    # so we monkey-patch to_tensors's ablation args through train_variant by
    # pre-ablating the caches once (cheap: it's a per-day numpy transform).
    from withinday.ablations import apply_token_ablation
    if ablation is not None:
        tr = []
        for c in caches_train:
            tok2, sum2 = apply_token_ablation(ablation, c.block_tokens, m, seed)
            tr.append(_replace_history(c, tok2, sum2))
        dv = []
        for c in caches_dev:
            tok2, sum2 = apply_token_ablation(ablation, c.block_tokens, m, seed)
            dv.append(_replace_history(c, tok2, sum2))
    else:
        tr, dv = caches_train, caches_dev

    model, dev_ll = train_variant(name, tr, dv, a_dim, tok_dim, summ_dim, cfg=cfg,
                                  zero_query=zero_query, verbose=verbose)
    recs = predict_records(name, model, dv, K=cfg.get("K", DEFAULT_CFG["K"]), zero_query=zero_query)
    return model, recs, dev_ll


def _replace_history(cache, tokens, summary):
    from dataclasses import replace
    return replace(cache, block_tokens=tokens, block_summary=summary)


def evaluate_candidate(name, caches_adtr, caches_addev, a_dim, tok_dim, summ_dim, cfg, m,
                       long_only_dev, online_platt_dev, margin, seed, verbose):
    """Trains the normal model plus its 5 ablations; returns a dict with the
    per-day records, the decision-rule gate booleans and the section 6.2
    deltas."""
    model, recs_normal, ll_normal = train_one(name, caches_adtr, caches_addev, a_dim, tok_dim,
                                              summ_dim, cfg, m, seed=seed, verbose=verbose)
    ablation_recs = {}
    for ab_name, kw in ABLATION_KWARGS.items():
        _, recs_ab, ll_ab = train_one(name, caches_adtr, caches_addev, a_dim, tok_dim, summ_dim,
                                      cfg, m, seed=seed, verbose=verbose, **kw)
        ablation_recs[ab_name] = (recs_ab, ll_ab)

    ll_normal_iw = impression_weighted_logloss(recs_normal)
    delta_long = paired(recs_normal, long_only_dev)
    delta_ops = paired(recs_normal, online_platt_dev)
    ll_no_hist = impression_weighted_logloss(ablation_recs["no_history"][0])
    ll_shuf = impression_weighted_logloss(ablation_recs["shuffled_chronology"][0])
    ll_noctx = impression_weighted_logloss(ablation_recs["no_context_interaction"][0])
    delta_chron = ll_shuf - ll_normal_iw          # eq 20, >0 => chronology matters
    delta_ctx = ll_noctx - ll_normal_iw           # eq 21, >0 => context interaction matters
    delta_nohist = ll_no_hist - ll_normal_iw      # >0 => real history helps beyond capacity alone

    gates = {
        "beats_long_only": delta_long["mean_delta"] < 0,
        "beats_online_platt": delta_ops["mean_delta"] < 0,
        "beats_no_history_control": delta_nohist > margin,
        "chron_or_ctx_signal": (delta_chron > margin) or (delta_ctx > margin),
    }
    return {
        "name": name, "model": model, "dev_records": recs_normal, "dev_ll": ll_normal_iw,
        "ablation_records": {k: v[0] for k, v in ablation_recs.items()},
        "ablation_ll": {k: v[1] for k, v in ablation_recs.items()},
        "delta_long_only": delta_long, "delta_online_platt": delta_ops,
        "delta_chron": delta_chron, "delta_ctx": delta_ctx, "delta_no_history": delta_nohist,
        "gates": gates, "eligible": all(gates.values()),
    }


def select_winner(results, dev_days):
    """Parsimony rule (plan section 8): among eligible candidates, the
    simplest whose dev log loss is within 1 SE of the best."""
    eligible = [r for r in results if r["eligible"]]
    if not eligible:
        return None
    best = min(r["dev_ll"] for r in eligible)
    se_of = {}
    for r in eligible:
        pdf = pd.DataFrame(per_day_frame(r["dev_records"]))
        se_of[r["name"]] = float(pdf["log_loss"].std(ddof=1) / max(len(pdf), 1) ** 0.5) if len(pdf) > 1 else 0.0
    best_se = se_of[min(eligible, key=lambda r: r["dev_ll"])["name"]]
    within = [r for r in eligible if r["dev_ll"] <= best + best_se]
    for name in LADDER_SIMPLE_TO_COMPLEX:
        for r in within:
            if r["name"] == name:
                return r
    return min(eligible, key=lambda r: r["dev_ll"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["criteo", "avazu"], default="criteo")
    ap.add_argument("--data", default=None)
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-features", type=int, default=2 ** 18)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--block-sec", type=int, default=900)
    ap.add_argument("--delay-sec", type=int, default=1800)
    ap.add_argument("--sketch-dim", type=int, default=32)
    ap.add_argument("--adapter-train-frac", type=float, default=0.7,
                    help="fraction of twoscale's dev days used to fit adapters; the rest early-stops them")
    ap.add_argument("--margin", type=float, default=1e-4, help="decision-rule 'practically visible margin'")
    ap.add_argument("--materiality-floor", type=float, default=2e-4)
    ap.add_argument("--config", default=None, help="JSON overriding withinday.train.DEFAULT_CFG")
    ap.add_argument("--verbose-train", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cfg = dict(DEFAULT_CFG)
    if args.config:
        cfg.update(json.loads(Path(args.config).read_text()))
    cfg["seed"] = args.seed
    m = args.sketch_dim

    path = args.data or DATA_PATHS[args.source]
    print(f"loading {args.source} from {path} (sample_frac={args.sample_frac}) ...", flush=True)
    ds = load(args.source, path, n_features=args.n_features, sample_frac=args.sample_frac, seed=args.seed)
    print(f"  {len(ds.y):,} rows, {ds.n_days} days, click rate {ds.y.mean():.4f}", flush=True)

    split = make_split(ds.n_days, warmup=args.warmup)
    print(f"  split: train {list(map(int, split.train_days))}  dev {list(map(int, split.dev_days))}  "
          f"test {list(map(int, split.test_days))}", flush=True)

    print("fitting long-term candidate bank ...", flush=True)
    bank = build_bank(ds, split.eval_days, seed=args.seed, n_jobs=args.n_jobs)
    eval_days = sorted(bank)
    w = adaptive_weights(bank, eval_days)
    q_by_day = long_term_predictions(bank, eval_days, "adaptive", weights=w)

    calib_cfg = CalibConfig(**CALIB_CONFIG)
    methods, _, _ = build_suite(bank, eval_days, calib_cfg, include_platt=True)

    print(f"building causal cache (block_sec={args.block_sec}, delay_sec={args.delay_sec}, m={m}) ...", flush=True)
    cache = build_cache(ds, bank, q_by_day, eval_days, block_sec=args.block_sec,
                        delay_sec=args.delay_sec, m=m, seed=args.seed)

    dev_days = sorted(d for d in split.dev_days if d in cache)
    test_days = sorted(d for d in split.test_days if d in cache)
    n_adtr = max(1, int(round(len(dev_days) * args.adapter_train_frac)))
    n_adtr = min(n_adtr, len(dev_days) - 1) if len(dev_days) > 1 else len(dev_days)
    adapter_train_days = dev_days[:n_adtr]
    adapter_dev_days = dev_days[n_adtr:] or dev_days[-1:]
    print(f"  adapter-train days {adapter_train_days}  adapter-dev days {adapter_dev_days}  "
          f"locked test days {test_days}", flush=True)

    a_dim, tok_dim, summ_dim = m + 2, token_dim(m), summary_dim(m)
    caches_adtr = [cache[d] for d in adapter_train_days]
    caches_addev = [cache[d] for d in adapter_dev_days]
    caches_test = [cache[d] for d in test_days]

    dev_set, test_set = set(adapter_dev_days), set(test_days)
    long_only_dev = [r for r in methods["long_only"] if r["day"] in dev_set]
    online_platt_dev = [r for r in methods["online_platt"] if r["day"] in dev_set]

    print("\n=== Stage B: training capacity-ladder candidates + ablations ===", flush=True)
    results = []
    for name in VARIANTS:
        t1 = time.time()
        r = evaluate_candidate(name, caches_adtr, caches_addev, a_dim, tok_dim, summ_dim,
                               cfg, m, long_only_dev, online_platt_dev, args.margin,
                               args.seed, args.verbose_train)
        results.append(r)
        print(f"  {name:16s} dev_ll={r['dev_ll']:.6f}  "
              f"Dlong={r['delta_long_only']['mean_delta']:+.6f}  "
              f"Dops={r['delta_online_platt']['mean_delta']:+.6f}  "
              f"Dchron={r['delta_chron']:+.6f}  Dctx={r['delta_ctx']:+.6f}  "
              f"Dnohist={r['delta_no_history']:+.6f}  gates={r['gates']}  "
              f"({time.time() - t1:.1f}s)", flush=True)

    winner = select_winner(results, adapter_dev_days)

    # ---- per-candidate report (dev) -------------------------------------
    cand_rows = []
    for r in results:
        cand_rows.append({
            "variant": r["name"], "dev_imp_wt_ll": r["dev_ll"],
            "delta_long_only": r["delta_long_only"]["mean_delta"],
            "delta_online_platt": r["delta_online_platt"]["mean_delta"],
            "delta_chron": r["delta_chron"], "delta_ctx": r["delta_ctx"],
            "delta_no_history": r["delta_no_history"],
            **{f"gate_{k}": v for k, v in r["gates"].items()},
            "eligible": r["eligible"], "selected": bool(winner and r["name"] == winner["name"]),
        })
    pd.DataFrame(cand_rows).to_csv(out / "candidates_dev.csv", index=False)

    ablation_rows = []
    for r in results:
        for ab_name, recs in r["ablation_records"].items():
            ablation_rows.append({"variant": r["name"], "ablation": ab_name,
                                  "dev_imp_wt_ll": impression_weighted_logloss(recs)})
    pd.DataFrame(ablation_rows).to_csv(out / "ablations_dev.csv", index=False)

    summary = {
        "config": {"source": args.source, "seed": args.seed, "sample_frac": args.sample_frac,
                   "n_rows": int(len(ds.y)), "n_days": ds.n_days,
                   "block_sec": args.block_sec, "delay_sec": args.delay_sec, "sketch_dim": m,
                   "adapter_train_days": list(map(int, adapter_train_days)),
                   "adapter_dev_days": list(map(int, adapter_dev_days)),
                   "test_days": list(map(int, test_days)), "adapter_cfg": cfg,
                   "margin": args.margin, "materiality_floor": args.materiality_floor},
        "candidates": {r["name"]: {"dev_ll": r["dev_ll"], "gates": r["gates"], "eligible": r["eligible"]}
                      for r in results},
        "winner": winner["name"] if winner else None,
    }

    # ---- Stage C: locked test, only if the plan's gate was actually cleared
    if winner is None:
        print("\nNo candidate cleared the decision-rule gates on development days.", flush=True)
        print("Plan section 8: retain Online Platt as baseline; do not open the locked test.", flush=True)
        summary["locked_test"] = None
        summary["interpretation"] = ("no candidate beats OPS and controls on dev -- within-day history "
                                     "is not exploitable by this capacity ladder; stop the branch")
    else:
        print(f"\n=== Stage C: locked test for selected model '{winner['name']}' "
              f"({len(test_days)} days, opened once) ===", flush=True)
        test_recs = predict_records(winner["name"], winner["model"], caches_test,
                                    K=cfg.get("K", DEFAULT_CFG["K"]))
        long_only_test = [r for r in methods["long_only"] if r["day"] in test_set]
        online_platt_test = [r for r in methods["online_platt"] if r["day"] in test_set]

        pd.DataFrame(per_day_frame(test_recs)).to_csv(out / "test_per_day.csv", index=False)
        test_ll = impression_weighted_logloss(test_recs)
        test_delta_long = paired(test_recs, long_only_test)
        test_delta_ops = paired(test_recs, online_platt_test)
        print(f"  {winner['name']:16s} test_imp_wt_ll={test_ll:.6f}  "
              f"vs long_only {test_delta_long}  vs online_platt {test_delta_ops}", flush=True)

        beats_both = test_delta_long["mean_delta"] < 0 and test_delta_ops["mean_delta"] < 0
        material = abs(test_delta_long["mean_delta"]) >= args.materiality_floor
        summary["locked_test"] = {
            "winner": winner["name"], "test_imp_wt_logloss": test_ll,
            "vs_long_only": test_delta_long, "vs_online_platt": test_delta_ops,
            "beats_both_on_locked_test": beats_both, "meets_materiality_floor": material,
        }
        summary["interpretation"] = (
            "proceed to downstream extensions: selected model beats long-only and Online Platt "
            "on the locked test" if beats_both and material else
            "selected model cleared dev gates but did not beat both baselines (or gain is below "
            "the materiality floor) on the locked test -- do not proceed to downstream extensions")

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nruntime {time.time() - t0:.1f}s -> {out}/", flush=True)


if __name__ == "__main__":
    main()
