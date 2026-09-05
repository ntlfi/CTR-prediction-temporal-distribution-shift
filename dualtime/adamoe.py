"""EMA-weighted mixture of the three shared experts, in the spirit of Liu
et al. (2022), *"On the Adaptation to Concept Drift for CTR Prediction"*
(arXiv:2204.05101), whose ``AdaMoE`` framework mixes CTR experts with a
statistical weighting policy to track concept drift.

**Provenance note**: the paper's own weight-update formula was not
independently verified (only its high-level framing was found via
search); this module implements the exact recipe given by this
experiment's own specification -- weights initialized uniformly, blended
via an EMA of the instantaneous inverse-loss softmax with momentum
``lambda`` -- which is precise and self-contained regardless of whether
it reproduces the authors' exact update rule.
"""
from __future__ import annotations

import numpy as np

EXPERTS = ("roll3", "roll7", "expanding")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def next_weights(weights: dict, day_losses: dict, lam: float) -> dict:
    """One EMA update (protocol section 7.4): after day ``d``'s labels are
    known, blend the current weights with day ``d``'s instantaneous
    inverse-loss softmax target. ``weights``/``day_losses`` are keyed by
    expert name; returns the weights to use for day ``d + 1``."""
    losses = np.array([day_losses[h] for h in EXPERTS])
    target = _softmax(-losses)
    w = np.array([weights[h] for h in EXPERTS])
    w_next = lam * w + (1.0 - lam) * target
    w_next = w_next / w_next.sum()
    return {h: float(w_next[i]) for i, h in enumerate(EXPERTS)}


def initial_weights() -> dict:
    return {h: 1.0 / len(EXPERTS) for h in EXPERTS}


def mixture_prediction(weights: dict, preds: dict) -> np.ndarray:
    """``q_{d,i} = sum_h w_h * p_{d,i}^{(h)}`` -- the weights used to
    predict day ``d`` must be the ones produced by the *previous* day's
    update, never day ``d``'s own losses (enforced by the caller passing
    day ``d``'s ``day_losses`` into :func:`next_weights` only after this
    is called)."""
    return sum(weights[h] * np.asarray(preds[h]) for h in EXPERTS)
