"""Stage B (plan section 5.2): train one capacity-ladder adapter on the
adapter-training days, early-stopping on separate development days. Also
Stage C's replay (:func:`predict_records`), which just runs the frozen model
forward with no gradient step -- causality for evaluation was already baked
into ``k_avail`` at cache-build time (Stage A), so "replay causally" here
just means "look up the block that was actually available," which is what
every forward pass already does.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .ablations import apply_token_ablation
from .adapters import build_variant, gather_summary, logit, sigmoid, v1_window
from .cache import DayCache

DEFAULT_CFG = dict(hidden=32, n_heads=2, mlp_hidden=(64, 32), rank=4, cross_dim=32,
                   delta_max=1.0, dropout=0.0, lr=1e-3, weight_decay=1e-5,
                   lam_delta=0.0, K=16, seed=0)


@dataclass
class DayTensors:
    d: int
    a: torch.Tensor
    y: torch.Tensor
    q: torch.Tensor
    tokens: torch.Tensor
    summary: torch.Tensor
    k_avail: torch.Tensor
    sec_in_day: np.ndarray


def to_tensors(cache: DayCache, device="cpu", ablation: str | None = None,
               m: int | None = None, seed: int = 0) -> DayTensors:
    tokens, summary = cache.block_tokens, cache.block_summary
    if ablation is not None:
        tokens, summary = apply_token_ablation(ablation, cache.block_tokens, m, seed)
    return DayTensors(
        d=cache.d,
        a=torch.tensor(cache.a, dtype=torch.float32, device=device),
        y=torch.tensor(cache.y, dtype=torch.float32, device=device),
        q=torch.tensor(cache.q, dtype=torch.float32, device=device),
        tokens=torch.tensor(tokens, dtype=torch.float32, device=device),
        summary=torch.tensor(summary, dtype=torch.float32, device=device),
        k_avail=torch.tensor(cache.k_avail, dtype=torch.long, device=device),
        sec_in_day=cache.sec_in_day,
    )


def forward_variant(name: str, model, day: DayTensors, K: int = 16, zero_query: bool = False):
    a = torch.zeros_like(day.a) if zero_query else day.a
    if name == "v1_transformer":
        seq, mask = v1_window(day.tokens, day.k_avail, K=K)
        return model(a, seq, mask)
    if name == "v2_gru":
        h_states = model.run_day(day.tokens)
        h_sel = h_states[day.k_avail + 1]
        return model(a, h_sel)
    s = gather_summary(day.summary, day.k_avail)
    return model(a, s)


def predicted_prob(name, model, day: DayTensors, K: int = 16, zero_query: bool = False, eps: float = 1e-5):
    delta = forward_variant(name, model, day, K=K, zero_query=zero_query)
    return sigmoid(logit(day.q, eps) + delta), delta


def compute_loss(p, delta, y, lam_delta: float = 0.0):
    p = p.clamp(1e-7, 1 - 1e-7)
    bce = -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
    return bce + lam_delta * (delta ** 2).mean()


def _mean_logloss(days_tensors, name, model, K, zero_query):
    tot, n = 0.0, 0
    with torch.no_grad():
        for day in days_tensors:
            p, _ = predicted_prob(name, model, day, K=K, zero_query=zero_query)
            p = p.clamp(1e-7, 1 - 1e-7)
            ll = -(day.y * torch.log(p) + (1 - day.y) * torch.log(1 - p))
            tot += float(ll.sum())
            n += ll.numel()
    return tot / max(n, 1)


def train_variant(name: str, caches_train, caches_dev, a_dim: int, token_dim: int,
                  summary_dim: int, cfg: dict | None = None, device: str = "cpu",
                  max_epochs: int = 30, patience: int = 4, zero_query: bool = False,
                  verbose: bool = False):
    """Trains one variant (plan eq 18, penalty ``lam_delta``; output layer
    zero-init so it starts exactly at the long-only predictor). Early-stops
    on ``caches_dev`` mean log loss. ``zero_query=True`` implements the
    'no context interaction' ablation at every forward pass, train and eval
    alike. Returns ``(model, best_dev_logloss)``."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    torch.manual_seed(cfg["seed"])
    model = build_variant(name, a_dim, token_dim, summary_dim, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    train_days = [to_tensors(c, device) for c in caches_train]
    dev_days = [to_tensors(c, device) for c in caches_dev]
    if not train_days or not dev_days:
        raise ValueError("need at least one adapter-training day and one dev day")

    K = cfg["K"]
    best_loss = _mean_logloss(dev_days, name, model, K, zero_query)
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    bad = 0
    rng = np.random.default_rng(cfg["seed"])

    for epoch in range(max_epochs):
        model.train()
        for i in rng.permutation(len(train_days)):
            day = train_days[i]
            opt.zero_grad()
            delta = forward_variant(name, model, day, K=K, zero_query=zero_query)
            p = sigmoid(logit(day.q) + delta)
            loss = compute_loss(p, delta, day.y, lam_delta=cfg["lam_delta"])
            loss.backward()
            opt.step()
        model.eval()
        dev_loss = _mean_logloss(dev_days, name, model, K, zero_query)
        if verbose:
            print(f"    [{name}] epoch {epoch}: dev logloss {dev_loss:.6f}", flush=True)
        if dev_loss < best_loss - 1e-6:
            best_loss, bad = dev_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_loss


def predict_records(name: str, model, caches, K: int = 16, device: str = "cpu",
                    zero_query: bool = False, ablation: str | None = None,
                    m: int | None = None, ablation_seed: int = 0):
    """Frozen-model causal replay (Stage C): one ``{"day","y","p",
    "sec_in_day"}`` record per day, compatible with ``twoscale.metrics``."""
    model.eval()
    recs = []
    with torch.no_grad():
        for c in caches:
            day = to_tensors(c, device, ablation=ablation, m=m, seed=ablation_seed)
            p, _ = predicted_prob(name, model, day, K=K, zero_query=zero_query)
            recs.append({"day": c.d, "y": c.y, "p": p.cpu().numpy(), "sec_in_day": c.sec_in_day})
    return recs
