"""Leakage / causality / identity / reproducibility checks for the
within-day capacity-ladder package. Runs on tiny synthetic arrays -- no
real data needed. ``python withinday_tests.py``.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import torch

from withinday.adapters import VARIANTS, build_variant
from withinday.blocks import (build_block_tokens, deterministic_summary,
                              last_available_block, n_blocks_per_day,
                              shuffle_block_order, summary_dim, token_dim)
from withinday.contextsketch import build_projection, context_sketch
from withinday.train import DayTensors, forward_variant

PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}")


def _fake_day(n, seed, block_sec=900):
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.05, 0.6, n)
    y = (rng.uniform(size=n) < q).astype(float)
    sec = np.sort(rng.uniform(0, 86400, n))
    return q, y, sec


# --------------------------------------------------------------------- #
#  context sketch                                                        #
# --------------------------------------------------------------------- #
def test_sketch_deterministic_and_bounded():
    rng = np.random.default_rng(0)
    n, F, m = 500, 2 ** 12, 32
    X = sp.random(n, F, density=0.01, random_state=0, format="csr")
    X.data[:] = 1.0
    R = build_projection(F, m, seed=7)
    c1 = context_sketch(X, m=m, R=R)
    c2 = context_sketch(X, m=m, R=R)
    check("sketch is deterministic given R", np.array_equal(c1, c2))
    norms = np.linalg.norm(c1, axis=1)
    nz = X.getnnz(axis=1) > 0
    check("sketch has bounded (unit) norm on nonzero rows",
          np.allclose(norms[nz], 1.0, atol=1e-8))
    R2 = build_projection(F, m, seed=8)
    c3 = context_sketch(X, m=m, R=R2)
    check("different seed gives a different projection", not np.allclose(c1, c3))


def test_sketch_zero_row_is_safe():
    m = 16
    X = sp.csr_matrix((3, 2 ** 10))  # all-zero rows
    c = context_sketch(X, m=m, seed=0)
    check("all-zero input row -> all-zero sketch row, no NaN/inf",
          np.all(c == 0) and np.isfinite(c).all())


# --------------------------------------------------------------------- #
#  block availability (causal clock all variants share)                  #
# --------------------------------------------------------------------- #
def test_last_available_block_boundaries():
    block_sec, delay_sec = 900, 1800
    # block 0 spans [0, 900); it matures at 900 + 1800 = 2700
    sec = np.array([0.0, 899.0, 900.0, 2699.0, 2700.0, 2701.0, 3599.0, 3600.0])
    k = last_available_block(sec, block_sec, delay_sec)
    expect = np.array([-1, -1, -1, -1, 0, 0, 0, 1])
    check("block maturation boundary is exact (block_end + delay)", np.array_equal(k, expect))


def test_last_available_block_matches_bruteforce():
    rng = np.random.default_rng(1)
    block_sec, delay_sec = 300, 1200
    sec = rng.uniform(0, 86400, 5000)
    got = last_available_block(sec, block_sec, delay_sec)
    nb = n_blocks_per_day(block_sec)
    want = np.full(len(sec), -1, dtype=int)
    for k in range(nb):
        matured_at = (k + 1) * block_sec + delay_sec
        want[sec >= matured_at] = k
    check("last_available_block matches brute-force reference", np.array_equal(got, want))


def test_never_sees_own_or_future_block():
    """An impression arriving inside block k (or any later block within the
    delay window) must not see block k itself as available history."""
    block_sec, delay_sec = 900, 1800
    q, y, sec = _fake_day(4000, 3)
    k_avail = last_available_block(sec, block_sec, delay_sec)
    own_block = np.minimum((sec // block_sec).astype(int), n_blocks_per_day(block_sec) - 1)
    check("k_avail is always strictly before the impression's own block",
          np.all(k_avail < own_block))


# --------------------------------------------------------------------- #
#  block tokens / deterministic summary (eq 5, 11)                       #
# --------------------------------------------------------------------- #
def test_token_shape_and_manual_values():
    block_sec = 3600
    nb = n_blocks_per_day(block_sec)
    m = 4
    q = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    sec = np.array([10.0, 20.0, 3610.0, 3620.0])  # first two in block 0, last two in block 1
    csketch = np.tile(np.arange(m, dtype=float), (4, 1))
    tokens = build_block_tokens(q, y, sec, csketch, block_sec)
    check("token array has one row per block of the day", tokens.shape[0] == nb)
    check("token width matches token_dim(m)", tokens.shape[1] == token_dim(m))
    check("mean_y for block 0 matches manual mean", np.isclose(tokens[0, 1], 0.5))
    check("mean_q for block 1 matches manual mean", np.isclose(tokens[1, 2], 0.35))
    check("untouched blocks stay exactly zero (no NaN)", np.all(tokens[2:] == 0))


def test_summary_shape_and_causality():
    rng = np.random.default_rng(4)
    nb, m = 20, 8
    tokens = rng.normal(size=(nb, token_dim(m)))
    summ = deterministic_summary(tokens)
    check("summary width matches summary_dim(m)", summ.shape[1] == summary_dim(m))
    cutoff = 10
    tokens2 = tokens.copy()
    tokens2[cutoff:] = rng.normal(size=tokens2[cutoff:].shape) * 100  # scramble the "future"
    summ2 = deterministic_summary(tokens2)
    check("s_k for k < cutoff is unaffected by scrambling blocks >= cutoff",
          np.allclose(summ[:cutoff], summ2[:cutoff]))
    check("s_k for k >= cutoff DOES change (sanity: the scramble is actually visible)",
          not np.allclose(summ[cutoff:], summ2[cutoff:]))


def test_ewma_matches_manual_recursion():
    tokens = np.array([[1.0], [2.0], [3.0], [4.0]])
    halflife = 2.0
    from withinday.blocks import _ewma_causal
    got = _ewma_causal(tokens, halflife)
    alpha = 1 - 0.5 ** (1 / halflife)
    want = np.empty(4)
    want[0] = 1.0
    for k in range(1, 4):
        want[k] = alpha * tokens[k, 0] + (1 - alpha) * want[k - 1]
    check("causal EWMA matches manual recursion", np.allclose(got[:, 0], want))


def test_shuffle_block_order_is_a_permutation():
    rng = np.random.default_rng(5)
    tokens = rng.normal(size=(30, 6))
    shuffled = shuffle_block_order(tokens, seed=42)
    check("shuffle preserves the block count", shuffled.shape == tokens.shape)
    orig_rows = {tuple(r) for r in tokens}
    shuf_rows = {tuple(r) for r in shuffled}
    check("shuffle is a permutation of the same rows (no row invented/dropped)",
          orig_rows == shuf_rows)
    check("shuffle actually reorders (not the identity, with overwhelming probability)",
          not np.allclose(tokens, shuffled))
    shuffled2 = shuffle_block_order(tokens, seed=42)
    check("shuffle is reproducible given the same seed", np.allclose(shuffled, shuffled2))


# --------------------------------------------------------------------- #
#  capacity-ladder adapters (identity property, context-interaction)     #
# --------------------------------------------------------------------- #
def _fake_daytensors(seed=0, n=50, nb=10, m=4):
    rng = np.random.default_rng(seed)
    a_dim, tok_dim, summ_dim = m + 2, token_dim(m), summary_dim(m)
    return DayTensors(
        d=0,
        a=torch.tensor(rng.normal(size=(n, a_dim)), dtype=torch.float32),
        y=torch.tensor(rng.integers(0, 2, n), dtype=torch.float32),
        q=torch.tensor(rng.uniform(0.05, 0.5, n), dtype=torch.float32),
        tokens=torch.tensor(rng.normal(size=(nb, tok_dim)), dtype=torch.float32),
        summary=torch.tensor(rng.normal(size=(nb, summ_dim)), dtype=torch.float32),
        k_avail=torch.tensor(rng.integers(-1, nb, n), dtype=torch.long),
        sec_in_day=np.zeros(n),
    ), a_dim, tok_dim, summ_dim


def test_adapters_zero_init_identity():
    day, a_dim, tok_dim, summ_dim = _fake_daytensors(seed=10)
    for name in VARIANTS:
        model = build_variant(name, a_dim, tok_dim, summ_dim, {})
        model.eval()
        with torch.no_grad():
            delta = forward_variant(name, model, day, K=6)
        check(f"{name}: zero-initialized -> delta == 0 (p_hat == q identity, plan eq 2)",
              torch.allclose(delta, torch.zeros_like(delta), atol=1e-6))


def test_no_context_interaction_removes_a_dependence():
    day, a_dim, tok_dim, summ_dim = _fake_daytensors(seed=11)
    # give two impressions the same k_avail (same available history) but
    # different current-impression input `a`
    day.k_avail[0] = 3
    day.k_avail[1] = 3
    for name in VARIANTS:
        torch.manual_seed(0)
        model = build_variant(name, a_dim, tok_dim, summ_dim, {})
        # nudge weights off zero so the model isn't trivially constant
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.01 * torch.randn_like(p))
        model.eval()
        with torch.no_grad():
            delta_normal = forward_variant(name, model, day, K=6, zero_query=False)
            delta_ablated = forward_variant(name, model, day, K=6, zero_query=True)
        check(f"{name}: normally two impressions sharing k_avail can still differ (context matters)",
              not torch.allclose(delta_normal[0], delta_normal[1], atol=1e-9))
        check(f"{name}: with zero_query, impressions sharing k_avail get the identical correction",
              torch.allclose(delta_ablated[0], delta_ablated[1], atol=1e-6))


def main():
    for fn in [
        test_sketch_deterministic_and_bounded,
        test_sketch_zero_row_is_safe,
        test_last_available_block_boundaries,
        test_last_available_block_matches_bruteforce,
        test_never_sees_own_or_future_block,
        test_token_shape_and_manual_values,
        test_summary_shape_and_causality,
        test_ewma_matches_manual_recursion,
        test_shuffle_block_order_is_a_permutation,
        test_adapters_zero_init_identity,
        test_no_context_interaction_removes_a_dependence,
    ]:
        print(f"{fn.__name__}:")
        fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
