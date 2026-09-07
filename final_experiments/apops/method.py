"""Assemble the AP-OPS evaluation rows from one shared frozen cross-day
``q_by_day``.

Old fixed-test rows (``build_rows``, kept as preliminary supporting
evidence): ``current_ops`` / ``ap_ops`` / ``single_memory`` / ``no_slope``.

Nested rolling-origin rows (``build_nested_rows``, the 2026-09-06
additional-experiments spec) -- four headline methods, all on identical
``q_{d,i}``:

  ops             projected gradient, reset daily      (existing reference)
  reset_ons       discounted ONS, reset daily          (isolate the optimizer change)
  persistent_ons  discounted ONS, carried across days  (isolate the value of persistence)
  ap_ops          reset R + short/long persistent S/L, adaptively aggregated (lambda_AP)

plus ``no_slope`` (a fixed at 1) as a supporting ablation.

Expert R / ``ops`` is ``twoscale.calib.replay_day`` verbatim, so
``lambda_AP = 0`` makes ``ap_ops`` reproduce ``ops`` prediction by
prediction.  ``reset_ons`` and ``persistent_ons`` share one half-life
(selected on validation) so their only difference is reset-vs-carry.
"""
from __future__ import annotations

from dataclasses import dataclass

from twoscale.calib import CalibConfig

from methods import ops_method
from apops.aggregate import MetaConfig, aggregate
from apops.experts import OnsConfig, replay_ons_stream

SHORT_HL_H = 4.0
LONG_HL_H = 16.0


@dataclass
class APOPSConfig:
    block_sec: int
    delay_sec: int
    ons_lam: float = 1e-2
    ons_eta: float = 1.0
    eta_m: float = 10.0
    switch_half_life_h: float = 16.0
    lambda_ap: float = 0.5                 # adaptive mass; 0 => ap_ops == ops
    a_bounds: tuple = (0.2, 5.0)
    b_bounds: tuple = (-0.25, 0.25)
    single_memory_pick: str = "L"          # "S" or "L" (old build_rows only)
    persistent_half_life_h: float = 4.0    # shared by reset_ons + persistent_ons

    def ons_cfg(self, half_life_h: float, learn_slope: bool = True,
                reset_daily: bool = False) -> OnsConfig:
        return OnsConfig(half_life_h=half_life_h, lam=self.ons_lam, eta_ons=self.ons_eta,
                         block_sec=self.block_sec, delay_sec=self.delay_sec,
                         a_bounds=self.a_bounds, b_bounds=self.b_bounds,
                         learn_slope=learn_slope, reset_daily=reset_daily)

    def meta_cfg(self) -> MetaConfig:
        return MetaConfig(eta_m=self.eta_m, switch_half_life_h=self.switch_half_life_h,
                          block_sec=self.block_sec, delay_sec=self.delay_sec,
                          lambda_ap=self.lambda_ap)


def _ops_calib_cfg(cfg: APOPSConfig, ops_hp: dict, platt: bool) -> CalibConfig:
    return CalibConfig(B=ops_hp["B"], eta0=ops_hp["eta0"], eta_schedule=ops_hp["schedule"],
                       update="block", block_sec=cfg.block_sec, delay_sec=cfg.delay_sec,
                       platt=platt, a_bounds=tuple(ops_hp.get("a_bounds", (0.2, 5.0))))


def ops_row(bank, days, q_by_day, ops_hp: dict, block_sec: int, delay_sec: int, platt: bool = True):
    cfg = CalibConfig(B=ops_hp["B"], eta0=ops_hp["eta0"], eta_schedule=ops_hp["schedule"],
                      update="block", block_sec=block_sec, delay_sec=delay_sec, platt=platt,
                      a_bounds=tuple(ops_hp.get("a_bounds", (0.2, 5.0))))
    return ops_method(bank, days, q_by_day, cfg)


def current_ops(bank, days, q_by_day, ops_hp: dict, block_sec: int, delay_sec: int):
    return ops_row(bank, days, q_by_day, ops_hp, block_sec, delay_sec, platt=True)


def reset_ons_row(bank, days, q_by_day, cfg: APOPSConfig, half_life_h: float, learn_slope: bool = True):
    return replay_ons_stream(q_by_day, bank, days,
                             cfg.ons_cfg(half_life_h, learn_slope, reset_daily=True))


def persistent_ons_row(bank, days, q_by_day, cfg: APOPSConfig, half_life_h: float, learn_slope: bool = True):
    return replay_ons_stream(q_by_day, bank, days,
                             cfg.ons_cfg(half_life_h, learn_slope, reset_daily=False))


def build_ap_experts(bank, days, q_by_day, cfg: APOPSConfig, ops_hp: dict, learn_slope: bool = True):
    """R (day-local OPS) + S / L persistent-ONS.  Independent of the meta
    knobs (lambda_AP, tau) -- reuse across a meta sweep."""
    r_recs, r_tr = ops_row(bank, days, q_by_day, ops_hp, cfg.block_sec, cfg.delay_sec, platt=learn_slope)
    s_recs, s_tr = persistent_ons_row(bank, days, q_by_day, cfg, SHORT_HL_H, learn_slope)
    l_recs, l_tr = persistent_ons_row(bank, days, q_by_day, cfg, LONG_HL_H, learn_slope)
    return {"R": r_recs, "S": s_recs, "L": l_recs}, {"R": r_tr, "S": s_tr, "L": l_tr}


def build_nested_rows(bank, days, q_by_day, cfg: APOPSConfig, ops_hp: dict):
    """The four headline rows + the no-slope ablation, one frozen ``cfg``."""
    rows, info = {}, {}
    rows["ops"], _ = ops_row(bank, days, q_by_day, ops_hp, cfg.block_sec, cfg.delay_sec)

    ro_recs, ro_tr = reset_ons_row(bank, days, q_by_day, cfg, cfg.persistent_half_life_h)
    rows["reset_ons"] = ro_recs
    po_recs, po_tr = persistent_ons_row(bank, days, q_by_day, cfg, cfg.persistent_half_life_h)
    rows["persistent_ons"] = po_recs
    info["reset_ons_trace"] = ro_tr[-3:]
    info["persistent_ons_trace"] = po_tr[-3:]

    experts, traces = build_ap_experts(bank, days, q_by_day, cfg, ops_hp, learn_slope=True)
    ap_recs, ap_info = aggregate(experts, bank, days, cfg.meta_cfg())
    rows["ap_ops"] = ap_recs
    info["ap_ops"] = ap_info
    info["expert_traces"] = {k: v[-3:] for k, v in traces.items()}

    ns_experts, _ = build_ap_experts(bank, days, q_by_day, cfg, ops_hp, learn_slope=False)
    ns_recs, ns_info = aggregate(ns_experts, bank, days, cfg.meta_cfg())
    rows["no_slope"] = ns_recs
    info["no_slope"] = ns_info
    return rows, info


# --------------------------------------------------------------------------- #
#  old fixed-test rows (kept; now use the corrected aggregator + abs-time queue) #
# --------------------------------------------------------------------------- #
def build_experts(bank, days, q_by_day, cfg: APOPSConfig, ops_hp: dict, learn_slope: bool = True):
    return build_ap_experts(bank, days, q_by_day, cfg, ops_hp, learn_slope)


def build_rows(bank, days, q_by_day, cfg: APOPSConfig, ops_hp: dict):
    rows, info = {}, {}
    rows["current_ops"], _ = ops_row(bank, days, q_by_day, ops_hp, cfg.block_sec, cfg.delay_sec)
    experts, traces = build_ap_experts(bank, days, q_by_day, cfg, ops_hp, learn_slope=True)
    ap_recs, ap_info = aggregate(experts, bank, days, cfg.meta_cfg())
    rows["ap_ops"] = ap_recs
    info["ap_ops"] = ap_info
    info["expert_traces"] = traces
    rows["single_memory"] = experts[cfg.single_memory_pick]
    info["single_memory_pick"] = cfg.single_memory_pick
    ns_experts, _ = build_ap_experts(bank, days, q_by_day, cfg, ops_hp, learn_slope=False)
    ns_recs, ns_info = aggregate(ns_experts, bank, days, cfg.meta_cfg())
    rows["no_slope"] = ns_recs
    info["no_slope"] = ns_info
    return rows, info


def anchor_only_ap_ops(bank, days, q_by_day, cfg: APOPSConfig, ops_hp: dict):
    """lambda_AP = 0 path: AP-OPS must equal OPS prediction by prediction."""
    experts, _ = build_ap_experts(bank, days, q_by_day, cfg, ops_hp, learn_slope=True)
    zero_cfg = MetaConfig(eta_m=cfg.eta_m, switch_half_life_h=cfg.switch_half_life_h,
                          block_sec=cfg.block_sec, delay_sec=cfg.delay_sec, lambda_ap=0.0)
    recs, _ = aggregate(experts, bank, days, zero_cfg)
    return recs
