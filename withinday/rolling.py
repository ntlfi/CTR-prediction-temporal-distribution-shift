"""Day-level rolling-origin evaluation for V5 (rolling-protocol sections 2-3).

``long_only`` and ``online_platt`` need no new machinery: ``twoscale``'s
per-day causal fits are already rolling-origin correct over whatever
``eval_days`` range they are given (see
``withinday_experiments/ROLLING_PROTOCOL_FREEZE.md``, "necessary new
choices" #1) -- callers just build them once over the full eligible range
and read off each day. Only V5 needs a genuine walk-forward loop: at each
outer day ``d``, select its hyperparameters by inner rolling-origin
validation on the (capped) most recent eligible days before ``d``, refit
on all of ``days < d``, and evaluate once on ``d``.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from twoscale.metrics import impression_weighted_logloss

from .train import DEFAULT_CFG, predict_records, train_variant

INNER_K = 3   # frozen: most-recent eligible prior days used per inner-validation fold
KNOB_GRID_V5 = dict(cross_dim=[16, 32, 64], lr=[3e-4, 1e-3], weight_decay=[1e-5, 1e-4])


def full_grid(knob_dict: dict) -> list[dict]:
    keys = list(knob_dict)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(knob_dict[k] for k in keys))]


def _simplicity_key(overlay: dict) -> int:
    return overlay.get("cross_dim", 32)


def _split_train_dev(days: list[int]):
    """Most recent day held out for early stopping, rest is adapter-train."""
    if len(days) >= 2:
        return days[:-1], days[-1:]
    return days, days


def select_config_inner_cv(cache_by_day: dict, candidate_days: list[int], grid: list[dict],
                           a_dim: int, tok_dim: int, summ_dim: int, cfg_base: dict,
                           inner_k: int = INNER_K, seed: int = 0, verbose: bool = False):
    """Nested selection (rolling-protocol section 2): inner rolling-origin
    validation over ``grid`` using the ``inner_k`` most recent days in
    ``candidate_days`` that themselves have >=1 day of history within
    ``candidate_days``. Returns ``(best_overlay, inner_days_used, rows)``."""
    eligible_inner = [v for v in candidate_days if candidate_days.index(v) >= 1]
    inner_days = eligible_inner[-inner_k:]

    rows, fold_losses_by_key = [], {}
    for overlay in grid:
        cfg = {**cfg_base, **overlay, "seed": seed}
        losses = []
        for v in inner_days:
            train_pool = [e for e in candidate_days if e < v]
            if not train_pool:
                continue
            adtr, addev = _split_train_dev(train_pool)
            model, _ = train_variant("v5_linear", [cache_by_day[e] for e in adtr],
                                     [cache_by_day[e] for e in addev],
                                     a_dim, tok_dim, summ_dim, cfg=cfg, verbose=verbose)
            recs = predict_records("v5_linear", model, [cache_by_day[v]], K=cfg["K"])
            losses.append(impression_weighted_logloss(recs))
        score = float(np.mean(losses)) if losses else float("inf")
        key = tuple(sorted(overlay.items()))
        fold_losses_by_key[key] = losses
        rows.append({**overlay, "inner_val_days": list(inner_days), "score": score, "n_folds": len(losses)})

    valid = [r for r in rows if np.isfinite(r["score"])]
    if not valid:
        return {}, inner_days, rows

    best = min(valid, key=lambda r: r["score"])
    best_key = tuple(sorted({k: best[k] for k in KNOB_GRID_V5}.items()))
    best_folds = fold_losses_by_key[best_key]
    se = float(np.std(best_folds, ddof=1) / max(len(best_folds), 1) ** 0.5) if len(best_folds) > 1 else 0.0
    within_1se = [r for r in valid if r["score"] <= best["score"] + se]
    winner = min(within_1se, key=_simplicity_key)
    best_overlay = {k: winner[k] for k in KNOB_GRID_V5}
    return best_overlay, inner_days, rows


@dataclass
class RollingDayResult:
    day: int
    n_impressions: int
    empirical_ctr: float
    y: np.ndarray
    p_long_only: np.ndarray
    p_online_platt: np.ndarray
    p_v5: np.ndarray
    p_v5_no_history: np.ndarray
    p_v5_shuffled_history: np.ndarray
    chosen_overlay: dict
    train_days: list = field(default_factory=list)
    inner_val_days: list = field(default_factory=list)

    @property
    def ll_long_only(self):
        from twoscale.metrics import day_logloss
        return day_logloss(self.y, self.p_long_only)

    @property
    def ll_online_platt(self):
        from twoscale.metrics import day_logloss
        return day_logloss(self.y, self.p_online_platt)

    @property
    def ll_v5(self):
        from twoscale.metrics import day_logloss
        return day_logloss(self.y, self.p_v5)

    @property
    def ll_v5_no_history(self):
        from twoscale.metrics import day_logloss
        return day_logloss(self.y, self.p_v5_no_history)

    @property
    def ll_v5_shuffled_history(self):
        from twoscale.metrics import day_logloss
        return day_logloss(self.y, self.p_v5_shuffled_history)


def rolling_origin_v5(cache_by_day: dict, outer_days: list[int], long_only_records: list,
                      online_platt_records: list, a_dim: int, tok_dim: int, summ_dim: int, m: int,
                      seed: int = 0, inner_k: int = INNER_K, verbose: bool = False):
    """Runs the full walk-forward loop over ``outer_days`` (ascending).
    Returns ``(list[RollingDayResult], list[inner-grid-search rows for the
    manifest])``."""
    grid = full_grid(KNOB_GRID_V5)
    cfg_base = dict(DEFAULT_CFG)
    long_by_day = {r["day"]: r for r in long_only_records}
    ops_by_day = {r["day"]: r for r in online_platt_records}

    results, all_inner_rows = [], []
    for d in sorted(outer_days):
        candidate_days = sorted(e for e in cache_by_day if e < d)
        if not candidate_days:
            raise ValueError(f"outer day {d} has no eligible prior days in the cache")

        best_overlay, inner_days, inner_rows = select_config_inner_cv(
            cache_by_day, candidate_days, grid, a_dim, tok_dim, summ_dim, cfg_base,
            inner_k=inner_k, seed=seed, verbose=verbose)
        for r in inner_rows:
            all_inner_rows.append({"outer_day": d, **r})

        chosen_cfg = {**cfg_base, **best_overlay, "seed": seed}
        adtr, addev = _split_train_dev(candidate_days)
        model, _ = train_variant("v5_linear", [cache_by_day[e] for e in adtr],
                                 [cache_by_day[e] for e in addev],
                                 a_dim, tok_dim, summ_dim, cfg=chosen_cfg, verbose=verbose)

        recs = predict_records("v5_linear", model, [cache_by_day[d]], K=chosen_cfg["K"])
        recs_nohist = predict_records("v5_linear", model, [cache_by_day[d]], K=chosen_cfg["K"],
                                      ablation="no_history", m=m)
        recs_shuf = predict_records("v5_linear", model, [cache_by_day[d]], K=chosen_cfg["K"],
                                    ablation="shuffled_chronology", m=m, ablation_seed=seed)

        y = cache_by_day[d].y
        nan_p = np.full(len(y), np.nan)
        results.append(RollingDayResult(
            day=d, n_impressions=len(y), empirical_ctr=float(y.mean()), y=y,
            p_long_only=np.asarray(long_by_day[d]["p"]) if d in long_by_day else nan_p,
            p_online_platt=np.asarray(ops_by_day[d]["p"]) if d in ops_by_day else nan_p,
            p_v5=np.asarray(recs[0]["p"]),
            p_v5_no_history=np.asarray(recs_nohist[0]["p"]),
            p_v5_shuffled_history=np.asarray(recs_shuf[0]["p"]),
            chosen_overlay=best_overlay, train_days=candidate_days, inner_val_days=inner_days,
        ))
    return results, all_inner_rows
