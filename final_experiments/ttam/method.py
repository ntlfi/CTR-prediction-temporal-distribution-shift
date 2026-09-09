"""The method variants, all on identical five-horizon expert predictions,
identical evaluation impressions, identical origin dates and seed IDs.

Main table (6):

    variant              historical prediction                 calibration
    -------------------  ------------------------------------   --------------
    expanding            expanding-history expert               none
    best_fixed_window    horizon h* chosen on inner validation  none
    arw                  causal adaptive window selection       none
    adamoe               causal mature-loss mixture             none
    ops                  validation-fitted fixed mixture        daily-reset OPS (free a, b)
    ttam                 AMG-TP                                  AP-OPS

Ablation -- a 2x2 crossing historical predictor {AMG-TP, expanding-history}
with calibration {AP-OPS, none} (TTAM_FROZEN.md sec. 4):

    cell                 historical prediction   calibration    also in
    -------------------  ---------------------   -----------    --------------
    ttam                 AMG-TP                  AP-OPS         main table
    amgtp_only           AMG-TP                  none           --
    expanding_apops      expanding-history       AP-OPS         --
    expanding            expanding-history       none           main table

Identical historical predictions are shared within each row of the 2x2:
``amgtp_only`` / ``ttam`` both feed the AMG-TP q-stream, ``expanding`` /
``expanding_apops`` both feed ``bank[d].preds["expanding"]``.

Every function returns a ``records`` list ``[{"day", "y", "p",
"sec_in_day"}, ...]`` over the days it is given; the nested runner slices
that into an inner-validation segment (for selection) and the single
scored origin day.

``arw`` / ``adamoe`` are the five-horizon generalisations of
``dualtime.arw`` / ``dualtime.adamoe`` (whose algorithm building blocks
are horizon-agnostic and reused directly). ``ops`` / the AP-OPS arms
reuse ``final_experiments/apops`` unchanged -- ``lambda_AP = 0`` makes
the AP-OPS calibration reproduce ``ops`` prediction by prediction, which
is the plan's identity check.
"""
from __future__ import annotations

import numpy as np

from dualtime.arw import pairwise_prefers_first

from twoscale.metrics import day_logloss
from twoscale.calib import CalibConfig

from methods import ops_method
from apops.method import APOPSConfig, build_ap_experts
from apops.aggregate import MetaConfig, aggregate

from .bank import HORIZONS5

# --------------------------------------------------------------------------- #
#  helpers                                                                     #
# --------------------------------------------------------------------------- #
def _rec(bank, d, p):
    return {"day": int(d), "y": bank[d].y, "p": np.asarray(p, float),
            "sec_in_day": bank[d].sec_in_day}


def _days(bank, days):
    return sorted(int(d) for d in days if d in bank)


def q_records(bank, days, q_by_day):
    return [_rec(bank, d, q_by_day[d]) for d in _days(bank, days) if d in q_by_day]


# --------------------------------------------------------------------------- #
#  1. Expanding                                                                #
# --------------------------------------------------------------------------- #
def expanding(bank, days):
    return [_rec(bank, d, bank[d].preds["expanding"]) for d in _days(bank, days)]


# --------------------------------------------------------------------------- #
#  2. Best Fixed Window -- h* on inner validation, frozen for the origin day   #
# --------------------------------------------------------------------------- #
def select_best_fixed_window(bank, inner_days):
    """h* minimising pooled inner-validation impression-weighted log loss.
    Ties are broken toward the shorter nominal window (HORIZONS5 order)."""
    best_h, best_l = None, np.inf
    for h in HORIZONS5:
        num = den = 0.0
        for d in inner_days:
            if d not in bank:
                continue
            y = np.asarray(bank[d].y, float)
            p = np.clip(np.asarray(bank[d].preds[h], float), 1e-12, 1 - 1e-12)
            num += -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
            den += len(y)
        l = num / den if den else np.inf
        if l < best_l - 1e-15:
            best_h, best_l = h, l
    return best_h, best_l


def best_fixed_window(bank, days, h_star):
    return [_rec(bank, d, bank[d].preds[h_star]) for d in _days(bank, days)]


# --------------------------------------------------------------------------- #
#  3. ARW -- causal single-elimination tournament over the five horizons       #
# --------------------------------------------------------------------------- #
def arw(bank, days, delta, min_history: int = 3, fallback: str = "expanding"):
    days = _days(bank, days)
    loss_hist = {h: [] for h in HORIZONS5}
    records, choices = [], []
    for d in days:
        if min(len(loss_hist[h]) for h in HORIZONS5) < min_history:
            chosen = fallback
        else:
            chosen = HORIZONS5[0]
            for challenger in HORIZONS5[1:]:
                prefers_current = pairwise_prefers_first(
                    np.asarray(loss_hist[chosen]), np.asarray(loss_hist[challenger]),
                    delta, n_candidates=len(HORIZONS5))
                chosen = chosen if prefers_current else challenger
        records.append(_rec(bank, d, bank[d].preds[chosen]))
        choices.append({"day": d, "chosen": chosen})
        for h in HORIZONS5:
            loss_hist[h].append(day_logloss(bank[d].y, bank[d].preds[h]))
    return records, choices


# --------------------------------------------------------------------------- #
#  4. AdaMoE -- EMA of the instantaneous inverse-loss softmax, mature losses    #
# --------------------------------------------------------------------------- #
def _softmax(x):
    x = np.asarray(x, float)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def adamoe(bank, days, lam):
    days = _days(bank, days)
    w = np.full(len(HORIZONS5), 1.0 / len(HORIZONS5))
    records, trace = [], []
    for d in days:
        preds = np.stack([np.asarray(bank[d].preds[h], float) for h in HORIZONS5], axis=1)
        q = preds @ w
        records.append(_rec(bank, d, q))
        trace.append({"day": d, **dict(zip(HORIZONS5, w.tolist()))})
        losses = np.array([day_logloss(bank[d].y, bank[d].preds[h]) for h in HORIZONS5])
        target = _softmax(-losses)
        w = lam * w + (1.0 - lam) * target
        w = w / w.sum()
    return records, trace


# --------------------------------------------------------------------------- #
#  5. OPS -- validation-fitted fixed mixture q, daily-reset OPS (free a, b)     #
# --------------------------------------------------------------------------- #
def ops(bank, days, q_by_day, ops_hp: dict, block_sec: int, delay_sec: int):
    cfg = CalibConfig(B=ops_hp["B"], eta0=ops_hp["eta0"], eta_schedule=ops_hp["schedule"],
                      update="block", block_sec=block_sec, delay_sec=delay_sec, platt=True,
                      a_bounds=tuple(ops_hp.get("a_bounds", (0.2, 5.0))))
    recs, _ = ops_method(bank, _days(bank, days), q_by_day, cfg)
    return recs


# --------------------------------------------------------------------------- #
#  6 / 9. AP-OPS on a given q stream (ttam uses q_amgtp; apops_only q_fixed)    #
# --------------------------------------------------------------------------- #
def apops_config(sel: dict, block_sec: int, delay_sec: int, ons_lam: float, ons_eta: float,
                 eta_m: float, lambda_ap: float, tau_h: float, persist_hl_h: float) -> APOPSConfig:
    return APOPSConfig(block_sec=block_sec, delay_sec=delay_sec,
                       ons_lam=ons_lam, ons_eta=ons_eta, eta_m=eta_m,
                       switch_half_life_h=tau_h, lambda_ap=lambda_ap,
                       a_bounds=tuple(sel["a_bounds"]), b_bounds=tuple(sel["b_bounds"]),
                       persistent_half_life_h=persist_hl_h)


def apops_on(bank, days, q_by_day, cfg: APOPSConfig, ops_hp: dict):
    """AP-OPS: reset anchor R (= daily-reset OPS on this q) + persistent
    short/long Platt experts, delayed fixed-share aggregation. Returns
    ``(records, info)``."""
    days = _days(bank, days)
    experts, traces = build_ap_experts(bank, days, q_by_day, cfg, ops_hp, learn_slope=True)
    mcfg = MetaConfig(eta_m=cfg.eta_m, switch_half_life_h=cfg.switch_half_life_h,
                      block_sec=cfg.block_sec, delay_sec=cfg.delay_sec, lambda_ap=cfg.lambda_ap)
    recs, info = aggregate(experts, bank, days, mcfg)
    info["expert_end_state"] = {k: (v[-1] if v else None) for k, v in traces.items()}
    return recs, info
