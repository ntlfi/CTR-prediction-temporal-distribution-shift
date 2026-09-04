"""Assemble the baseline / ablation suite of plan section 4 into per-day
prediction records.

Each method is a list of ``{"day", "y", "p", "sec_in_day"}`` dicts over the
evaluation days. Long-term ``q`` is always frozen per day and comes from
:mod:`twoscale.longterm`; the calibrated methods add the causal within-day
replay of :mod:`twoscale.calib` on top.
"""
from __future__ import annotations

import numpy as np

from .calib import CalibConfig, oracle_intercept, replay_day, time_of_day_intercepts, _sigmoid, _logit
from .longterm import adaptive_weights, long_term_predictions

SLOT_SEC = 3600
N_SLOTS = 24


def _uncalibrated(q_by_day, bank, days):
    return [{"day": d, "y": bank[d].y, "p": q_by_day[d], "sec_in_day": bank[d].sec_in_day}
            for d in days if d in q_by_day]


def _calibrated(q_by_day, bank, days, cfg: CalibConfig, shuffle_seed=None):
    """Sequential day-by-day replay with optional cross-day carry-over of b."""
    recs, traces = [], []
    prev_b_end = 0.0
    for d in sorted(days):
        if d not in q_by_day:
            continue
        init_b = cfg.carryover_rho * prev_b_end if cfg.carryover_rho else cfg.init_b
        out = replay_day(q_by_day[d], bank[d].y, bank[d].sec_in_day, cfg,
                         init_b=init_b, shuffle_seed=(shuffle_seed + d if shuffle_seed is not None else None))
        prev_b_end = out["b_end"]
        recs.append({"day": d, "y": bank[d].y, "p": out["p_hat"], "sec_in_day": bank[d].sec_in_day})
        traces.append({"day": d, "b_end": out["b_end"], "a_end": out["a_end"], "trace": out["trace"]})
    return recs, traces


def _time_of_day(q_by_day, bank, days, B, eps):
    recs = []
    ev = sorted(q_by_day)
    for d in sorted(days):
        if d not in q_by_day:
            continue
        hist = [e for e in ev if e < d]
        if hist:
            hq = np.concatenate([q_by_day[e] for e in hist])
            hy = np.concatenate([bank[e].y for e in hist])
            hs = np.concatenate([(bank[e].sec_in_day // SLOT_SEC).astype(int) for e in hist])
            c = time_of_day_intercepts(hq, hy, hs, N_SLOTS, B=B, eps=eps)
        else:
            c = np.zeros(N_SLOTS)
        slot = np.minimum((bank[d].sec_in_day // SLOT_SEC).astype(int), N_SLOTS - 1)
        p = _sigmoid(_logit(q_by_day[d], eps) + c[slot])
        recs.append({"day": d, "y": bank[d].y, "p": p, "sec_in_day": bank[d].sec_in_day})
    return recs


def _oracle(q_by_day, bank, days, B, eps):
    recs = []
    for d in sorted(days):
        if d not in q_by_day:
            continue
        b_star, _ = oracle_intercept(q_by_day[d], bank[d].y, B=B, eps=eps)
        p = _sigmoid(_logit(q_by_day[d], eps) + b_star)
        recs.append({"day": d, "y": bank[d].y, "p": p, "sec_in_day": bank[d].sec_in_day})
    return recs


def build_suite(bank: dict, eval_days, cfg: CalibConfig,
                mixture_eta: float = 60.0, mixture_halflife: float = 5.0,
                include_platt: bool = True):
    """Returns ``(methods, q_sources, traces)``:

    * ``methods``: name -> per-day records
    * ``q_sources``: name -> {day: frozen long-term q}  (for regret / oracle diagnostics)
    * ``traces``: name -> per-day (b_end, a_end, trace) for the calibrated methods
    """
    eval_days = sorted(int(d) for d in eval_days if d in bank)
    w = adaptive_weights(bank, eval_days, eta=mixture_eta, halflife=mixture_halflife)

    q = {
        "roll3": long_term_predictions(bank, eval_days, "roll3"),
        "roll7": long_term_predictions(bank, eval_days, "roll7"),
        "expanding": long_term_predictions(bank, eval_days, "expanding"),
        "equal": long_term_predictions(bank, eval_days, "equal"),
        "adaptive": long_term_predictions(bank, eval_days, "adaptive", weights=w),
    }

    methods, q_sources, traces = {}, {}, {}

    def add_uncal(name, qsrc):
        methods[name] = _uncalibrated(q[qsrc], bank, eval_days)
        q_sources[name] = q[qsrc]

    add_uncal("expanding", "expanding")
    add_uncal("rolling_3", "roll3")
    add_uncal("rolling_7", "roll7")
    add_uncal("equal_ensemble", "equal")
    add_uncal("long_only", "adaptive")

    methods["short_only"], traces["short_only"] = _calibrated(q["expanding"], bank, eval_days, cfg)
    q_sources["short_only"] = q["expanding"]
    methods["combined"], traces["combined"] = _calibrated(q["adaptive"], bank, eval_days, cfg)
    q_sources["combined"] = q["adaptive"]

    methods["time_of_day"] = _time_of_day(q["adaptive"], bank, eval_days, cfg.B, cfg.eps)
    q_sources["time_of_day"] = q["adaptive"]
    methods["oracle_intercept"] = _oracle(q["adaptive"], bank, eval_days, cfg.B, cfg.eps)
    q_sources["oracle_intercept"] = q["adaptive"]

    if include_platt:
        from dataclasses import replace
        pcfg = replace(cfg, platt=True)
        methods["online_platt"], traces["online_platt"] = _calibrated(q["adaptive"], bank, eval_days, pcfg)
        q_sources["online_platt"] = q["adaptive"]

    return methods, q_sources, traces
