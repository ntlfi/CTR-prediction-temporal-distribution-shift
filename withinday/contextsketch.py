"""Fixed signed-hash context sketch ``c(x)`` (plan eq 6), dimension ``m``.

The plan asks for "a fixed signed-hash sketch of the ten categorical fields,
normalized to have bounded norm." ``twoscale.data`` only keeps the *already
hashed* pre-bid feature matrix (``Dataset.X``, width ``2**18``) -- it never
retains the raw categorical strings. Re-parsing the raw files just to hash
them a second time at a different width would duplicate a lot of I/O for no
benefit, since a second feature-hashing pass is exactly what a fixed random
sign/bucket projection of the *existing* hashed columns already gives: each
raw categorical token maps to exactly one column of ``X``, so hashing those
column indices down to ``m`` signed buckets is algebraically the same
"hash the categorical tokens" operation, just composed with the first hash.

``build_projection`` fixes that random sign/bucket map once (seeded, so it
is reproducible and identical across every day and every capacity-ladder
variant, per plan section 2.3's "identical across all variants" rule);
``context_sketch`` applies it and L2-normalizes each row to bounded norm.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def build_projection(n_features: int, m: int, seed: int = 0) -> sp.csr_matrix:
    """Fixed sparse signed projection ``R`` (``n_features x m``): each input
    column is hashed to exactly one of ``m`` output buckets with a random
    sign, so ``X @ R`` is a cheap single sparse matmul."""
    rng = np.random.default_rng(seed)
    buckets = rng.integers(0, m, size=n_features)
    signs = (rng.integers(0, 2, size=n_features) * 2 - 1).astype(np.float64)
    return sp.csr_matrix((signs, (np.arange(n_features), buckets)),
                         shape=(n_features, m), dtype=np.float64)


def context_sketch(X, m: int = 32, seed: int = 0, R: sp.csr_matrix | None = None) -> np.ndarray:
    """``c(x)`` in ``R^m`` for every row of ``X``, L2-normalized to bounded
    (unit) norm. Pass a shared ``R`` (from :func:`build_projection`) across
    days/datasets so the sketch means the same thing everywhere; otherwise
    one is built fresh from ``(X.shape[1], m, seed)``."""
    if R is None:
        R = build_projection(X.shape[1], m, seed=seed)
    C = X @ R
    C = np.asarray(C.todense()) if sp.issparse(C) else np.asarray(C)
    norm = np.linalg.norm(C, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return C / norm
