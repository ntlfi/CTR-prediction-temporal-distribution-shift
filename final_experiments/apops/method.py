"""Assemble the plan's four evaluation rows (section 3 table) from one
shared frozen cross-day ``q_by_day``:

  current_ops     exact locked OPS -- golden reference (== headline OPS row)
  ap_ops          three-expert R / S / L delayed fixed-share aggregate
  single_memory   the better of S / L on dev, no meta mixture
  no_slope        AP-OPS with slope a fixed at 1 everywhere (update only b)

Expert R is reused verbatim from ``methods.ops_method`` (which is
``twoscale.calib.replay_day``, daily reset, projected gradient) so
``current_ops`` and the R component of ``ap_ops`` are the same computation
the headline table already ran.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from twoscale.calib import CalibConfig

from methods import ops_method
from apops.aggregate import MetaConfig, aggregate
from apops.experts import OnsConfig, replay_ons_stream

SHORT_HL_H = 4.0
LONG_HL_H = 16.0


@dataclass
class APOPSConfig:
    """Everything tuned on dev (plan: ONS scale/regularization + the two
    meta parameters) plus the frozen block/delay context."""
    block_sec: int
    delay_sec: int
    ons_lam: float = 1e-2
    ons_eta: float = 1.0
    eta_m: float = 10.0
    switch_half_life_h: float = 16.0
    a_bounds: tuple = (0.2, 5.0)
    b_bounds: tuple = (-0.25, 0.25)
    single_memory_pick: str = "L"     # "S" or "L" -- chosen on dev

    def ons_cfg(self, half_life_h: float, learn_slope: bool = True) -> OnsConfig:
        return OnsConfig(half_life_h=half_life_h, lam=self.ons_lam, eta_ons=self.ons_eta,
                         block_sec=self.block_sec, delay_sec=self.delay_sec,
                         a_bounds=self.a_bounds, b_bounds=self.b_bounds,
                         learn_slope=learn_slope)

    def meta_cfg(self) -> MetaConfig:
        return MetaConfig(eta_m=self.eta_m, switch_half_life_h=self.switch_half_life_h,
                          block_sec=self.block_sec, delay_sec=self.delay_sec)


def _ops_calib_cfg(cfg: APOPSConfig, ops_hp: dict, platt: bool) -> CalibConfig:
    return CalibConfig(B=ops_hp["B"], eta0=ops_hp["eta0"], eta_schedule=ops_hp["schedule"],
                       update="block", block_sec=cfg.block_sec, delay_sec=cfg.delay_sec,
                       platt=platt, a_bounds=tuple(ops_hp.get("a_bounds", (0.2, 5.0))))


def build_experts(bank, days, q_by_day, cfg: APOPSConfig, ops_hp: dict, learn_slope: bool = True):
    """Returns ``(expert_records, traces)`` with R / S / L in that order."""
    r_cfg = _ops_calib_cfg(cfg, ops_hp, platt=learn_slope)
    r_recs, r_tr = ops_method(bank, days, q_by_day, r_cfg)
    s_recs, s_tr = replay_ons_stream(q_by_day, bank, days, cfg.ons_cfg(SHORT_HL_H, learn_slope))
    l_recs, l_tr = replay_ons_stream(q_by_day, bank, days, cfg.ons_cfg(LONG_HL_H, learn_slope))
    expert_records = {"R": r_recs, "S": s_recs, "L": l_recs}
    traces = {"R": r_tr, "S": s_tr, "L": l_tr}
    return expert_records, traces


def current_ops(bank, days, q_by_day, ops_hp: dict, block_sec: int, delay_sec: int):
    cfg = CalibConfig(B=ops_hp["B"], eta0=ops_hp["eta0"], eta_schedule=ops_hp["schedule"],
                      update="block", block_sec=block_sec, delay_sec=delay_sec, platt=True,
                      a_bounds=tuple(ops_hp.get("a_bounds", (0.2, 5.0))))
    recs, traces = ops_method(bank, days, q_by_day, cfg)
    return recs, traces


def build_rows(bank, days, q_by_day, cfg: APOPSConfig, ops_hp: dict):
    """All four rows.  ``info`` carries the AP-OPS mechanism diagnostics."""
    rows, info = {}, {}

    rows["current_ops"], _ = current_ops(bank, days, q_by_day, ops_hp, cfg.block_sec, cfg.delay_sec)

    experts, traces = build_experts(bank, days, q_by_day, cfg, ops_hp, learn_slope=True)
    ap_recs, ap_info = aggregate(experts, bank, days, cfg.meta_cfg())
    rows["ap_ops"] = ap_recs
    info["ap_ops"] = ap_info
    info["expert_traces"] = traces

    pick = cfg.single_memory_pick
    rows["single_memory"] = experts[pick]
    info["single_memory_pick"] = pick

    ns_experts, _ = build_experts(bank, days, q_by_day, cfg, ops_hp, learn_slope=False)
    ns_recs, ns_info = aggregate(ns_experts, bank, days, cfg.meta_cfg())
    rows["no_slope"] = ns_recs
    info["no_slope"] = ns_info

    return rows, info


def anchor_only_ap_ops(bank, days, q_by_day, cfg: APOPSConfig, ops_hp: dict):
    """AP-OPS with weights pinned to (1, 0, 0): must equal ``current_ops``."""
    experts, _ = build_experts(bank, days, q_by_day, cfg, ops_hp, learn_slope=True)
    recs, _ = aggregate(experts, bank, days, cfg.meta_cfg(), fixed_weights=(1.0, 0.0, 0.0))
    return recs
