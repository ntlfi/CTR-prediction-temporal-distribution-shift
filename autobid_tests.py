"""Invariant tests for the autobidding simulator (autobid.py).

Run: .venv/bin/python autobid_tests.py
"""
import numpy as np
import pandas as pd

from autobid import (linear_frontier, noskill_pctr, oracle_pctr, paced_auction,
                     shuffled_pctr, value_at_matched_spend)


def _toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    day = np.repeat(np.arange(20), n // 20)
    # click depends on a latent score; cost correlates with click (clicked
    # impressions are more expensive, as in Criteo)
    score = rng.normal(size=n)
    click = (rng.random(n) < 1 / (1 + np.exp(-score))).astype(int)
    cost = np.abs(rng.normal(scale=1e-4, size=n)) + 3e-4 * click + 1e-5
    conv = (click & (rng.random(n) < 0.1)).astype(int)
    attrib = conv.copy()
    cpo = np.full(n, 0.2)
    true_pctr = 1 / (1 + np.exp(-score))            # a good model
    return dict(day=day, click=click, cost=cost, conv=conv, attrib=attrib,
                cpo=cpo, good=true_pctr)


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def test_frontier_monotone():
    t = _toy()
    fr = linear_frontier(t["good"], t["click"], t["cost"], t["day"],
                         conv=t["conv"], attrib=t["attrib"], cpo=t["cpo"], n_scales=60)
    check("spend non-decreasing in scale", (fr["spend"].diff().dropna() >= -1e-12).all())
    check("clicks non-decreasing in scale", (fr["clicks"].diff().dropna() >= -1e-9).all())
    check("win_rate in [0,1]", fr["win_rate"].between(0, 1).all())
    check("frontier spans low..high win rate",
          fr["win_rate"].min() < 0.05 and fr["win_rate"].max() > 0.95)


def test_oracle_dominates():
    t = _toy()
    grid = np.linspace(0.05, 0.4, 30) * t["cost"].sum()
    fr = {
        "oracle": linear_frontier(oracle_pctr(t["click"]), t["click"], t["cost"], t["day"], n_scales=80),
        "good":   linear_frontier(t["good"], t["click"], t["cost"], t["day"], n_scales=80),
        "noskill": linear_frontier(noskill_pctr(t["click"]), t["click"], t["cost"], t["day"], n_scales=80),
        "shuffled": linear_frontier(shuffled_pctr(t["good"], 1), t["click"], t["cost"], t["day"], n_scales=80),
    }
    ms = value_at_matched_spend(fr, grid, value="clicks")
    check("oracle >= good at matched spend", (ms["oracle"] >= ms["good"] - 1e-6).all())
    check("good >= noskill at matched spend (mostly)",
          (ms["good"] >= ms["noskill"] - 1e-6).mean() > 0.9)
    check("good > shuffled at matched spend (mean)", ms["good"].mean() > ms["shuffled"].mean())


def test_paced_respects_budget():
    t = _toy()
    for frac in (0.1, 0.3, 0.6, 1.0):
        budget = frac * t["cost"].sum()
        s, blocks = paced_auction(t["good"], t["click"], t["cost"], t["day"], budget,
                                  conv=t["conv"], attrib=t["attrib"], cpo=t["cpo"])
        check(f"spend <= budget (frac={frac})", s["spend"] <= budget + 1e-12)
        check(f"budget_violation == 0 (frac={frac})", s["budget_violation"] == 0.0)
        check(f"per-block allowance stays finite (frac={frac})",
              np.isfinite(blocks["allowance_after"]).all())


def test_paced_more_budget_more_value():
    t = _toy()
    prev = -1
    for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
        s, _ = paced_auction(t["good"], t["click"], t["cost"], t["day"], frac * t["cost"].sum())
        check(f"clicks non-decreasing in budget (frac={frac})", s["clicks"] >= prev)
        prev = s["clicks"]


def test_paced_beats_noskill_at_equal_budget():
    t = _toy()
    b = 0.3 * t["cost"].sum()
    g, _ = paced_auction(t["good"], t["click"], t["cost"], t["day"], b)
    ns, _ = paced_auction(noskill_pctr(t["click"]), t["click"], t["cost"], t["day"], b)
    check("skilled bidder wins more clicks than no-skill at equal budget",
          g["clicks"] > ns["clicks"])
    check("both spend within budget", g["spend"] <= b + 1e-12 and ns["spend"] <= b + 1e-12)


def test_determinism():
    t = _toy()
    a = linear_frontier(t["good"], t["click"], t["cost"], t["day"], n_scales=25)
    b = linear_frontier(t["good"], t["click"], t["cost"], t["day"], n_scales=25)
    check("linear_frontier deterministic", a.equals(b))
    s1, _ = paced_auction(t["good"], t["click"], t["cost"], t["day"], 0.4 * t["cost"].sum())
    s2, _ = paced_auction(t["good"], t["click"], t["cost"], t["day"], 0.4 * t["cost"].sum())
    check("paced_auction deterministic", s1 == s2)


def test_matched_spend_endpoints():
    t = _toy()
    fr = {"good": linear_frontier(t["good"], t["click"], t["cost"], t["day"], n_scales=50)}
    lo, hi = fr["good"]["spend"].min(), fr["good"]["spend"].max()
    ms = value_at_matched_spend(fr, [lo, hi], value="clicks")
    check("matched-spend interp hits frontier endpoints",
          abs(ms["good"].iloc[0] - fr["good"].sort_values("spend")["clicks"].iloc[0]) < 1e-6
          and abs(ms["good"].iloc[-1] - fr["good"]["clicks"].max()) < 1e-6)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(fn.__name__)
        fn()
    print(f"\nall {len(tests)} autobid test groups passed")
