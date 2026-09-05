"""Builds the six headline methods from one shared three-expert bank.
Every non-Expanding method reads the *same* bank (``twoscale.longterm``'s
``roll3``/``roll7``/``expanding``); OPS and DualTime-CTR read
byte-identical ``q_{d,i}`` (the adaptive cross-day mixture) -- see
``final_experiments/PROGRESS.md`` for what in this module is newly
written versus reused unchanged from ``twoscale``/``withinday``.

Run scripts in this directory with the repo root on PYTHONPATH, e.g.
``PYTHONPATH=. .venv/bin/python final_experiments/run_hpo.py ...`` from
the repo root -- Python then also puts this directory on sys.path
automatically, so sibling imports (``from methods import ...``) resolve.
"""
from __future__ import annotations

from twoscale.calib import CalibConfig
from twoscale.longterm import HORIZONS, adaptive_weights, long_term_predictions
from twoscale.methods import _calibrated
from twoscale.metrics import day_logloss, impression_weighted_logloss

from dualtime.adamoe import initial_weights as adamoe_initial_weights
from dualtime.adamoe import mixture_prediction, next_weights as adamoe_next_weights
from dualtime.arw import select_expert as arw_select_expert
from dualtime.online import DualTimeConfig, build_hash_projection
from dualtime.online import replay_day as dualtime_replay_day
from withinday.blocks import summary_dim
from withinday.contextsketch import build_projection


def expanding_method(bank, days):
    days = sorted(d for d in days if d in bank)
    return [{"day": d, "y": bank[d].y, "p": bank[d].preds["expanding"], "sec_in_day": bank[d].sec_in_day}
           for d in days]


def best_fixed_window(bank, dev_days, all_days):
    """Choose h* once via dev-day loss (section 7.2); frozen for every
    later day (dev included, for uniformity of the returned records)."""
    dev_days = [d for d in dev_days if d in bank]
    losses = {h: impression_weighted_logloss(
        [{"day": d, "y": bank[d].y, "p": bank[d].preds[h]} for d in dev_days]) for h in HORIZONS}
    h_star = min(losses, key=losses.get)
    days = sorted(d for d in all_days if d in bank)
    records = [{"day": d, "y": bank[d].y, "p": bank[d].preds[h_star], "sec_in_day": bank[d].sec_in_day}
              for d in days]
    return h_star, losses, records


def arw_method(bank, days, delta, min_history=3):
    """Day-by-day Adaptive Rolling Window selection (section 7.3):
    causal -- day d's choice only ever sees days < d's realized losses."""
    days = sorted(d for d in days if d in bank)
    loss_history = {h: [] for h in HORIZONS}
    records, choices = [], []
    for d in days:
        expert = arw_select_expert(loss_history, delta=delta, min_history=min_history)
        p = bank[d].preds[expert]
        records.append({"day": d, "y": bank[d].y, "p": p, "sec_in_day": bank[d].sec_in_day})
        choices.append({"day": d, "chosen_expert": expert})
        for h in HORIZONS:
            loss_history[h].append(day_logloss(bank[d].y, bank[d].preds[h]))
    return records, choices


def adamoe_method(bank, days, lam):
    """Section 7.4: weights for day d were produced by day d-1's update;
    day d's own labels update the weights only after its own predictions
    are recorded."""
    days = sorted(d for d in days if d in bank)
    w = adamoe_initial_weights()
    records, trace = [], []
    for d in days:
        preds = {h: bank[d].preds[h] for h in HORIZONS}
        q = mixture_prediction(w, preds)
        records.append({"day": d, "y": bank[d].y, "p": q, "sec_in_day": bank[d].sec_in_day})
        trace.append({"day": d, **w})
        day_losses = {h: day_logloss(bank[d].y, preds[h]) for h in HORIZONS}
        w = adamoe_next_weights(w, day_losses, lam)
    return records, trace


def adaptive_q_by_day(bank, days, eta, halflife):
    """Section 8: the shared adaptive cross-day mixture -- the identical
    q_{d,i} OPS and DualTime-CTR both receive."""
    days = sorted(d for d in days if d in bank)
    w = adaptive_weights(bank, days, eta=eta, halflife=halflife)
    return long_term_predictions(bank, days, "adaptive", weights=w), w


def ops_method(bank, days, q_by_day, cfg: CalibConfig):
    days = sorted(d for d in days if d in bank)
    records, traces = _calibrated(q_by_day, bank, days, cfg)
    return records, traces


def dualtime_method(ds, bank, days, q_by_day, dt_cfg: DualTimeConfig, sketch_seed=0, hash_seed=0):
    """Section 11: DualTime-CTR's within-day correction on top of the
    shared adaptive q_by_day. ``ds`` is the twoscale ``Dataset`` (for the
    raw hashed feature rows, to build each day's context sketch)."""
    days = sorted(d for d in days if d in bank)
    R_sketch = build_projection(ds.X.shape[1], dt_cfg.m, seed=sketch_seed)
    a_dim, s_dim = dt_cfg.m + 2, summary_dim(dt_cfg.m)
    Ra, Rs = build_hash_projection(a_dim, s_dim, cross_dim=dt_cfg.cross_dim, seed=hash_seed)
    records = []
    for d in days:
        sl = ds.day_slice(d)
        out = dualtime_replay_day(q_by_day[d], bank[d].y, bank[d].sec_in_day, ds.X[sl],
                                  R_sketch, Ra, Rs, dt_cfg)
        records.append({"day": d, "y": bank[d].y, "p": out["p_hat"], "sec_in_day": bank[d].sec_in_day})
    return records
