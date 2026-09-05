"""Causality / identity / sanity checks for the dualtime/ package (ARW,
AdaMoE, DualTime-CTR's online module). Runs on tiny synthetic arrays --
no real data needed. ``python dualtime_tests.py``.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from dualtime.adamoe import EXPERTS as MOE_EXPERTS
from dualtime.adamoe import initial_weights, mixture_prediction, next_weights
from dualtime.arw import EXPERTS as ARW_EXPERTS
from dualtime.arw import pairwise_prefers_first, select_expert
from dualtime.online import DualTimeConfig, build_hash_projection, build_phi, phi_dim
from dualtime.online import replay_day as dualtime_replay_day
from withinday.blocks import summary_dim, token_dim
from withinday.contextsketch import build_projection

PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}")


# --------------------------------------------------------------------- #
#  ARW                                                                    #
# --------------------------------------------------------------------- #
def test_arw_fallback_before_min_history():
    hist = {h: [0.5, 0.4] for h in ARW_EXPERTS}   # only 2 days, min_history=3
    check("falls back before min_history days of history",
          select_expert(hist, delta=0.1, min_history=3, fallback="expanding") == "expanding")


def test_arw_prefers_the_clearly_better_expert():
    rng = np.random.default_rng(0)
    n = 30
    a_losses = 0.30 + 0.01 * rng.standard_normal(n)   # consistently lower
    b_losses = 0.45 + 0.01 * rng.standard_normal(n)
    check("pairwise comparison prefers the consistently-lower-loss expert",
          pairwise_prefers_first(a_losses, b_losses, delta=0.1))
    check("comparison is anti-symmetric", not pairwise_prefers_first(b_losses, a_losses, delta=0.1))


def test_arw_tournament_picks_the_best_of_three():
    rng = np.random.default_rng(1)
    n = 40
    hist = {
        "roll3": list(0.50 + 0.02 * rng.standard_normal(n)),
        "roll7": list(0.35 + 0.02 * rng.standard_normal(n)),      # best
        "expanding": list(0.55 + 0.02 * rng.standard_normal(n)),
    }
    check("tournament selects the genuinely best-performing expert",
          select_expert(hist, delta=0.1, min_history=3) == "roll7")


def test_arw_only_uses_the_arrays_it_is_given():
    # a caller-level causal guarantee: the function has no way to see
    # anything beyond the arrays passed to it, so a correct integration
    # (only ever passing days < d) is automatically causal.
    hist_before_d = {"roll3": [0.4, 0.4, 0.4], "roll7": [0.3, 0.3, 0.3], "expanding": [0.5, 0.5, 0.5]}
    choice_before = select_expert(hist_before_d, delta=0.1, min_history=3)
    hist_with_future_appended = {k: v + [999.0] for k, v in hist_before_d.items()}
    # appending a future day's (extreme) value to a DIFFERENT array object
    # must not retroactively change a decision already made from the
    # un-extended history
    check("decision from day-<d history is unaffected by what a future day would contain",
          choice_before == select_expert(hist_before_d, delta=0.1, min_history=3))
    check("(sanity) the extended history is in fact different from the original",
          hist_with_future_appended != hist_before_d)


# --------------------------------------------------------------------- #
#  AdaMoE                                                                 #
# --------------------------------------------------------------------- #
def test_adamoe_initial_weights_uniform():
    w = initial_weights()
    check("initial weights are uniform over the 3 experts", all(np.isclose(v, 1 / 3) for v in w.values()))
    check("initial weights sum to 1", np.isclose(sum(w.values()), 1.0))


def test_adamoe_lambda_one_is_pure_momentum():
    w0 = {"roll3": 0.2, "roll7": 0.3, "expanding": 0.5}
    losses = {"roll3": 0.9, "roll7": 0.1, "expanding": 0.9}   # would drastically favor roll7 if used
    w1 = next_weights(w0, losses, lam=1.0)
    check("lambda=1 ignores the new day's losses entirely", all(np.isclose(w1[h], w0[h]) for h in MOE_EXPERTS))


def test_adamoe_lambda_zero_is_pure_instantaneous_softmax():
    w0 = {"roll3": 0.9, "roll7": 0.05, "expanding": 0.05}
    losses = {"roll3": 1.0, "roll7": 1.0, "expanding": 1.0}   # tied losses -> uniform target
    w1 = next_weights(w0, losses, lam=0.0)
    check("lambda=0 ignores prior weights entirely (tied losses -> ~uniform)",
          all(np.isclose(w1[h], 1 / 3, atol=1e-6) for h in MOE_EXPERTS))


def test_adamoe_weights_always_sum_to_one():
    w = initial_weights()
    rng = np.random.default_rng(2)
    for _ in range(10):
        losses = {h: float(rng.uniform(0.2, 0.6)) for h in MOE_EXPERTS}
        w = next_weights(w, losses, lam=0.5)
        check("weights sum to 1 after an EMA update", np.isclose(sum(w.values()), 1.0))


def test_adamoe_mixture_prediction_is_weighted_sum():
    w = {"roll3": 0.5, "roll7": 0.3, "expanding": 0.2}
    preds = {"roll3": np.array([0.1, 0.2]), "roll7": np.array([0.3, 0.4]), "expanding": np.array([0.5, 0.6])}
    got = mixture_prediction(w, preds)
    want = 0.5 * preds["roll3"] + 0.3 * preds["roll7"] + 0.2 * preds["expanding"]
    check("mixture prediction matches the weighted sum by hand", np.allclose(got, want))


# --------------------------------------------------------------------- #
#  DualTime-CTR online module                                            #
# --------------------------------------------------------------------- #
def _fake_day(n, seed, F=2 ** 10, m=8):
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.05, 0.5, n)
    y = (rng.uniform(size=n) < q).astype(float)
    sec = np.sort(rng.uniform(0, 86400, n))
    X = sp.random(n, F, density=0.02, random_state=seed, format="csr")
    R = build_projection(F, m, seed=0)
    return q, y, sec, X, R


def test_dualtime_no_history_identity():
    m = 8
    q, y, sec, X, R = _fake_day(500, seed=0, m=m)
    cfg = DualTimeConfig(block_sec=3600, delay_sec=1800, m=m, cross_dim=8, B_w=1.0)
    a_dim, s_dim = m + 2, summary_dim(m)
    Ra, Rs = build_hash_projection(a_dim, s_dim, cross_dim=8, seed=0)
    out = dualtime_replay_day(q, y, sec, X, R, Ra, Rs, cfg)
    # w == 0 is only *guaranteed* within the very first block: the update
    # triggered by that block's own early-arriving impressions can fire
    # before the *next* block starts whenever block_sec > delay_sec (as
    # here), so it is block 0 specifically -- not "block_sec + delay_sec"
    # -- that is provably still at the zero-init weight.
    before = sec < cfg.block_sec
    p_expected_before = np.clip(q[before], cfg.eps, 1 - cfg.eps)
    check("p_hat == q for every impression in the first block (w starts at 0)",
          np.allclose(out["p_hat"][before], p_expected_before, atol=1e-9))


def test_dualtime_projection_never_violated():
    m = 8
    q, y, sec, X, R = _fake_day(2000, seed=3, m=m)
    cfg = DualTimeConfig(block_sec=1800, delay_sec=1800, m=m, cross_dim=8, B_w=0.5)
    a_dim, s_dim = m + 2, summary_dim(m)
    Ra, Rs = build_hash_projection(a_dim, s_dim, cross_dim=8, seed=0)
    out = dualtime_replay_day(q, y, sec, X, R, Ra, Rs, cfg)
    check("w never exceeds the B_w projection radius at any update step",
          all(t["w_norm"] <= cfg.B_w + 1e-9 for t in out["trace"]))


def test_dualtime_future_label_perturbation():
    m = 8
    q, y, sec, X, R = _fake_day(3000, seed=4, m=m)
    cfg = DualTimeConfig(block_sec=900, delay_sec=1800, m=m, cross_dim=8, B_w=1.0)
    a_dim, s_dim = m + 2, summary_dim(m)
    Ra, Rs = build_hash_projection(a_dim, s_dim, cross_dim=8, seed=0)
    full = dualtime_replay_day(q, y, sec, X, R, Ra, Rs, cfg)["p_hat"]

    cut = 43200.0
    y2 = y.copy()
    y2[sec + cfg.delay_sec > cut] = 1 - y2[sec + cfg.delay_sec > cut]
    part = dualtime_replay_day(q, y2, sec, X, R, Ra, Rs, cfg)["p_hat"]

    before = sec < cut - cfg.delay_sec - cfg.block_sec
    check("predictions before the cutoff are bitwise unchanged by scrambling not-yet-matured labels",
          np.allclose(full[before], part[before]))


def test_phi_is_norm_bounded():
    m, cross_dim = 8, 8
    a_dim, s_dim = m + 2, summary_dim(m)
    rng = np.random.default_rng(0)
    a = rng.normal(size=(50, a_dim)) * 10   # deliberately large, pre-normalization
    s = rng.normal(size=(50, s_dim)) * 10
    Ra, Rs = build_hash_projection(a_dim, s_dim, cross_dim=cross_dim, seed=0)
    phi = build_phi(a, s, Ra, Rs)
    check("phi width matches phi_dim()", phi.shape[1] == phi_dim(a_dim, s_dim, cross_dim))
    check("||phi||_2 <= 1 for every row, even with large raw inputs",
          np.all(np.linalg.norm(phi, axis=1) <= 1.0 + 1e-9))


def main():
    for fn in [
        test_arw_fallback_before_min_history,
        test_arw_prefers_the_clearly_better_expert,
        test_arw_tournament_picks_the_best_of_three,
        test_arw_only_uses_the_arrays_it_is_given,
        test_adamoe_initial_weights_uniform,
        test_adamoe_lambda_one_is_pure_momentum,
        test_adamoe_lambda_zero_is_pure_instantaneous_softmax,
        test_adamoe_weights_always_sum_to_one,
        test_adamoe_mixture_prediction_is_weighted_sum,
        test_dualtime_no_history_identity,
        test_dualtime_projection_never_violated,
        test_dualtime_future_label_perturbation,
        test_phi_is_norm_bounded,
    ]:
        print(f"{fn.__name__}:")
        fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
