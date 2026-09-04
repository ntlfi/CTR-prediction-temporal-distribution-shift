"""Leakage / causality / identity / reproducibility checks for the
two-timescale package. Runs on tiny synthetic-ish arrays -- no real data
needed. ``python twoscale_tests.py``.
"""
from __future__ import annotations

import numpy as np

from twoscale.calib import CalibConfig, oracle_intercept, replay_day, _sigmoid, _logit
from twoscale.longterm import HORIZONS, adaptive_weights, DayBank, long_term_predictions
from twoscale.splits import make_split

PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}")


def _fake_day(n, seed):
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.05, 0.6, n)
    y = (rng.uniform(size=n) < q).astype(float)
    sec = np.sort(rng.uniform(0, 86400, n))
    return q, y, sec


# 1. no future leakage: b used to predict block k depends only on labels
#    matured before block k ends. Verify by zeroing all labels after a cutoff
#    time and checking predictions before the cutoff are unchanged.
def test_causal_replay():
    q, y, sec = _fake_day(4000, 1)
    cfg = CalibConfig(update="block", block_sec=1800, delay_sec=1800)
    full = replay_day(q, y, sec, cfg)["p_hat"]
    cut = 43200.0
    y2 = y.copy()
    y2[sec + cfg.delay_sec > cut] = 0.0        # scramble everything not yet matured by `cut`
    part = replay_day(q, y2, sec, cfg)["p_hat"]
    before = sec < cut - cfg.delay_sec - cfg.block_sec
    check("block replay uses only matured labels", np.allclose(full[before], part[before]))


def test_impression_causal():
    q, y, sec = _fake_day(3000, 2)
    cfg = CalibConfig(update="impression", delay_sec=1800)
    full = replay_day(q, y, sec, cfg)["p_hat"]
    cut = 50000.0
    y2 = y.copy(); y2[sec + cfg.delay_sec > cut] = 1 - y2[sec + cfg.delay_sec > cut]
    part = replay_day(q, y2, sec, cfg)["p_hat"]
    before = sec < cut - cfg.delay_sec
    check("impression replay uses only matured labels", np.allclose(full[before], part[before]))


# 2. identity: eta0 = 0  =>  p_hat == q exactly (no calibration movement)
def test_zero_lr_identity():
    q, y, sec = _fake_day(2000, 3)
    out = replay_day(q, y, sec, CalibConfig(eta0=0.0, init_b=0.0))
    check("eta0=0 is identity", np.allclose(out["p_hat"], np.clip(q, 1e-5, 1 - 1e-5), atol=1e-9)
          or np.allclose(_logit(out["p_hat"], 1e-5), _logit(q, 1e-5), atol=1e-9))


# 3. projection respected
def test_projection():
    q = np.full(500, 0.001); y = np.ones(500); sec = np.linspace(0, 86000, 500)
    out = replay_day(q, y, sec, CalibConfig(B=0.5, eta0=5.0, delay_sec=0))
    check("b stays in [-B, B]", abs(out["b_end"]) <= 0.5 + 1e-9)


# 4. oracle intercept actually minimises
def test_oracle_intercept():
    rng = np.random.default_rng(4)
    q = rng.uniform(0.1, 0.5, 5000)
    y = (rng.uniform(size=5000) < np.clip(q + 0.15, 0, 1)).astype(float)   # q is biased low
    b, l = oracle_intercept(q, y, B=3.0)
    grid = np.linspace(-3, 3, 601)
    losses = []
    for bb in grid:
        p = np.clip(_sigmoid(_logit(q, 1e-5) + bb), 1e-12, 1 - 1e-12)
        losses.append(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    check("oracle intercept ~ grid argmin", abs(b - grid[int(np.argmin(losses))]) < 0.05)
    check("oracle intercept positive when q biased low", b > 0)


# 4b. chronology placebo: when the calibration need is correlated with
#     within-day time, shuffling the order destroys most of the gain.
def test_chronology_placebo():
    rng = np.random.default_rng(41)
    n = 40000
    sec = np.sort(rng.uniform(0, 86400, n))
    z_true = rng.uniform(-2.5, -0.5, n)
    drift = np.linspace(-1.5, 1.5, n)                 # net-zero daily bias, strong intraday ramp
    q = _sigmoid(z_true - drift)                       # q mis-set by exactly -drift
    y = (rng.uniform(size=n) < _sigmoid(z_true)).astype(float)
    cfg = CalibConfig(update="block", block_sec=1800, delay_sec=900, eta0=0.3,
                      eta_schedule="const", B=2.0)
    real = replay_day(q, y, sec, cfg)["p_hat"]
    shuf = replay_day(q, y, sec, cfg, shuffle_seed=7)["p_hat"]

    def ll(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()

    g_real = ll(q) - ll(real)
    g_shuf = ll(q) - ll(shuf)
    check("real replay beats raw q under time-correlated drift", g_real > 0.005)
    check("shuffling destroys most of the within-day gain", g_shuf < 0.5 * g_real)


# 5. reproducibility
def test_reproducible():
    q, y, sec = _fake_day(2500, 5)
    a = replay_day(q, y, sec, CalibConfig())["p_hat"]
    b = replay_day(q, y, sec, CalibConfig())["p_hat"]
    check("replay deterministic", np.array_equal(a, b))


# 6. adaptive mixture is causal + a valid simplex
def test_mixture():
    rng = np.random.default_rng(6)
    bank = {}
    for d in range(10):
        n = 300
        preds = {h: rng.uniform(0.1, 0.4, n) for h in HORIZONS}
        bank[d] = DayBank(d=d, y=(rng.uniform(size=n) < 0.3).astype(int),
                          sec_in_day=np.sort(rng.uniform(0, 86400, n)), preds=preds)
    w = adaptive_weights(bank, range(10))
    check("mixture weights sum to 1", all(abs(sum(w[d].values()) - 1) < 1e-9 for d in w))
    check("mixture day 0 is uniform (no history)", abs(w[0][HORIZONS[0]] - 1 / 3) < 1e-9)
    q = long_term_predictions(bank, range(10), "adaptive", weights=w)
    check("adaptive q within expert hull",
          all(np.all(q[d] >= min(bank[d].preds[h].min() for h in HORIZONS) - 1e-9) for d in q))


# 7. split proportions + ordering
def test_split():
    s = make_split(116, warmup=4)
    check("split ~ 60/21/35 for n=116", len(s.train_days) == 60 and len(s.dev_days) == 21 and len(s.test_days) == 35)
    check("split contiguous + ordered",
          s.train_days.max() < s.dev_days.min() < s.dev_days.max() < s.test_days.min())
    s31 = make_split(31, warmup=4)
    check("split covers all days for n=31",
          len(s31.train_days) + len(s31.dev_days) + len(s31.test_days) == 31)


if __name__ == "__main__":
    for fn in [test_causal_replay, test_impression_causal, test_chronology_placebo, test_zero_lr_identity,
               test_projection, test_oracle_intercept, test_reproducible,
               test_mixture, test_split]:
        print(fn.__name__)
        fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
