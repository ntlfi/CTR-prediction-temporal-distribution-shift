"""The five required ablations (plan section 7, items 1-2, 4-5, plus the
'context-only control' baseline of section 4, which is the same object as
ablation 1). Each token-level ablation is a pure transform of a day's block
tokens; the caller recomputes ``deterministic_summary`` on the result so V1's
window, V2's GRU state and V3-V5's summary all see a *consistent* ablated
history. The remaining required ablation, "no context interaction" (H2 /
eq 21's ``context-free`` arm), is not a token transform -- it zeros the
*current-impression* input ``a`` at the model-call site instead, so history
still varies but the correction no longer depends on which impression is
asking. See ``withinday/train.py``'s ``zero_query`` argument.
"""
from __future__ import annotations

import numpy as np

from .blocks import TOKEN_SCALAR_FIELDS, deterministic_summary, shuffle_block_order

_MEAN_Y = TOKEN_SCALAR_FIELDS.index("mean_y")
_MEAN_R = TOKEN_SCALAR_FIELDS.index("mean_r")
_MEAN_ABSR = TOKEN_SCALAR_FIELDS.index("mean_absr")
_MEAN_LOGLOSS = TOKEN_SCALAR_FIELDS.index("mean_logloss")
_N_SCALAR = len(TOKEN_SCALAR_FIELDS)


def zero_history(block_tokens: np.ndarray) -> np.ndarray:
    """Ablation 1 ('no history') == the 'context-only control' baseline."""
    return np.zeros_like(block_tokens)


def no_residual_sketch(block_tokens: np.ndarray, m: int) -> np.ndarray:
    """Ablation 4: drop the residual-weighted context sketch r*c(x), the
    plan's flagged "key heterogeneous-drift feature"."""
    out = block_tokens.copy()
    out[:, _N_SCALAR + m:_N_SCALAR + 2 * m] = 0.0
    return out


def label_free(block_tokens: np.ndarray, m: int) -> np.ndarray:
    """Ablation 5: keep traffic + context-composition, drop every feature
    that touches the label (y, residual, log loss -- including r*c(x),
    itself a residual feature)."""
    out = block_tokens.copy()
    out[:, [_MEAN_Y, _MEAN_R, _MEAN_ABSR, _MEAN_LOGLOSS]] = 0.0
    out[:, _N_SCALAR + m:_N_SCALAR + 2 * m] = 0.0
    return out


def shuffled_chronology(block_tokens: np.ndarray, seed: int) -> np.ndarray:
    """Ablation 2 / H3 chronology placebo: same blocks, scrambled order."""
    return shuffle_block_order(block_tokens, seed)


TOKEN_ABLATIONS = {
    "no_history": lambda tokens, m, seed: zero_history(tokens),
    "shuffled_chronology": lambda tokens, m, seed: shuffled_chronology(tokens, seed),
    "no_residual_sketch": lambda tokens, m, seed: no_residual_sketch(tokens, m),
    "label_free_history": lambda tokens, m, seed: label_free(tokens, m),
}

# ablations resolved at the model-call site (see withinday/train.py)
QUERY_ABLATIONS = {"no_context_interaction"}

ALL_ABLATIONS = tuple(TOKEN_ABLATIONS) + tuple(QUERY_ABLATIONS)


def apply_token_ablation(name: str, block_tokens: np.ndarray, m: int, seed: int = 0):
    """``(block_tokens', block_summary')`` for one of ``TOKEN_ABLATIONS``."""
    tokens2 = TOKEN_ABLATIONS[name](block_tokens, m, seed)
    return tokens2, deterministic_summary(tokens2)
