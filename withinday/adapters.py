"""Capacity ladder V1-V5 (plan section 3): five function classes for the
residual correction ``delta_{d,i}`` in ``p_hat = sigmoid(logit(q) + delta)``,
from most to least expressive. Every variant is zero-initialized so
``delta == 0`` before any training (plan eq 2's identity property), and every
variant reads the *same* current-impression input ``a`` and the *same*
per-block history (``withinday.cache.DayCache``) -- only the function class
applied to that history differs, per plan section 2.3's fairness rule.

The three history *lookups* used by the ladder (all causal by construction,
since they only ever index blocks ``<= k_avail[i]``):

* V1 attends over the last ``K`` block tokens ending at ``k_avail[i]``
  (:func:`v1_window`);
* V2's GRU state after processing block ``k_avail[i]`` (:meth:`V2GRU.run_day`
  + gather by ``k_avail[i] + 1``);
* V3/V4/V5 read the deterministic summary ``s_k`` at ``k_avail[i]``
  (:func:`gather_summary`).

All three treat ``k_avail[i] == -1`` (no block has matured yet) as "no
history": an all-padding/all-zero input, not an error.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def zero_init_(linear: nn.Linear):
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def sigmoid(z):
    return torch.sigmoid(z.clamp(-30, 30))


def logit(p, eps=1e-5):
    p = p.clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))


# --------------------------------------------------------------------- #
#  causal history lookups shared by every variant                        #
# --------------------------------------------------------------------- #
def gather_summary(block_summary: torch.Tensor, k_avail: torch.Tensor) -> torch.Tensor:
    """``s_{d,i}`` = the deterministic summary at block ``k_avail[i]``, or an
    all-zero vector if ``k_avail[i] == -1`` (no history yet)."""
    nb, dim = block_summary.shape
    pad = torch.cat([block_summary, block_summary.new_zeros(1, dim)], dim=0)
    idx = torch.where(k_avail < 0, nb, k_avail)
    return pad[idx]


def v1_window(block_tokens: torch.Tensor, k_avail: torch.Tensor, K: int = 16):
    """Last ``K`` available block tokens ending at ``k_avail[i]`` (eq 7),
    zero-padded/masked where fewer than ``K`` blocks have matured yet.
    Returns ``(seq [n, K, token_dim], mask [n, K] bool, True = real block)``.
    """
    nb, dim = block_tokens.shape
    pad = torch.cat([block_tokens, block_tokens.new_zeros(1, dim)], dim=0)
    offsets = torch.arange(K, device=block_tokens.device)
    idx = k_avail[:, None] - (K - 1) + offsets[None, :]           # (n, K)
    valid = (idx >= 0) & (idx <= k_avail[:, None])
    idx_safe = torch.where(valid, idx, nb)
    seq = pad[idx_safe]
    return seq, valid


# --------------------------------------------------------------------- #
#  shared MLP head: delta = delta_max * tanh(mlp(x)), zero-init last layer #
# --------------------------------------------------------------------- #
class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden, delta_max: float = 1.0, dropout: float = 0.0):
        super().__init__()
        dims = [in_dim] + list(hidden)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(dims[-1], 1)
        zero_init_(self.out)
        self.delta_max = delta_max

    def forward(self, x):
        h = self.body(x)
        return self.delta_max * torch.tanh(self.out(h)).squeeze(-1)


# --------------------------------------------------------------------- #
#  V1: context-query Transformer (eq 7-8)                                #
# --------------------------------------------------------------------- #
class V1Transformer(nn.Module):
    def __init__(self, a_dim: int, token_dim: int, hidden: int = 32, n_heads: int = 2,
                delta_max: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.wq = nn.Linear(a_dim, hidden)
        self.wk = nn.Linear(token_dim, hidden)
        self.wv = nn.Linear(token_dim, hidden)
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True, dropout=dropout)
        self.head = MLPHead(a_dim + hidden, [32], delta_max, dropout)

    def forward(self, a, hist_seq, hist_mask):
        q = self.wq(a).unsqueeze(1)                      # (B, 1, hidden)
        k = self.wk(hist_seq)
        v = self.wv(hist_seq)
        all_masked = hist_mask.sum(dim=1) == 0
        key_padding_mask = ~hist_mask
        key_padding_mask = key_padding_mask.clone()
        key_padding_mask[all_masked, 0] = False           # avoid all-True row (would NaN)
        z, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask)
        z = z.squeeze(1)
        z = torch.where(all_masked.unsqueeze(-1), torch.zeros_like(z), z)
        return self.head(torch.cat([a, z], dim=-1))


# --------------------------------------------------------------------- #
#  V2: contextual GRU adapter (eq 9-10)                                  #
# --------------------------------------------------------------------- #
class V2GRU(nn.Module):
    def __init__(self, a_dim: int, token_dim: int, hidden: int = 32,
                delta_max: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.hidden = hidden
        self.cell = nn.GRUCell(token_dim, hidden)
        self.Wh = nn.Linear(hidden, a_dim, bias=False)    # projects h_k for the a' (x) (W h_k) term
        self.head = MLPHead(2 * a_dim + hidden, [32], delta_max, dropout)

    def run_day(self, block_tokens: torch.Tensor) -> torch.Tensor:
        """``h_states[0]`` = initial (no-history) state; ``h_states[k + 1]``
        = state after processing block ``k``. Returns ``(nb + 1, hidden)``,
        differentiable so V2 trains end-to-end through the recurrence."""
        h = block_tokens.new_zeros(1, self.hidden)
        states = [h]
        for k in range(block_tokens.shape[0]):
            h = self.cell(block_tokens[k:k + 1], h)
            states.append(h)
        return torch.cat(states, dim=0)

    def forward(self, a, h_sel):
        inter = a * self.Wh(h_sel)
        return self.head(torch.cat([a, h_sel, inter], dim=-1))


# --------------------------------------------------------------------- #
#  V3: fixed-window contextual MLP (eq 11-12)                            #
# --------------------------------------------------------------------- #
class V3MLP(nn.Module):
    def __init__(self, a_dim: int, summary_dim: int, hidden=(64, 32),
                delta_max: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.head = MLPHead(a_dim + summary_dim, hidden, delta_max, dropout)

    def forward(self, a, s):
        return self.head(torch.cat([a, s], dim=-1))


# --------------------------------------------------------------------- #
#  V4: low-rank bilinear adapter (eq 13)                                 #
# --------------------------------------------------------------------- #
class V4Bilinear(nn.Module):
    def __init__(self, a_dim: int, summary_dim: int, rank: int = 4):
        super().__init__()
        self.alpha = nn.Linear(summary_dim, 1, bias=False)
        self.beta = nn.Linear(a_dim, 1, bias=True)
        self.U = nn.Linear(a_dim, rank, bias=False)
        self.V = nn.Linear(summary_dim, rank, bias=False)
        for lin in (self.alpha, self.beta, self.U, self.V):
            zero_init_(lin)

    def forward(self, a, s):
        bilinear = (self.U(a) * self.V(s)).sum(-1)
        return self.alpha(s).squeeze(-1) + self.beta(a).squeeze(-1) + bilinear


# --------------------------------------------------------------------- #
#  V5: linear interaction adapter (eq 14-15)                             #
# --------------------------------------------------------------------- #
class V5Linear(nn.Module):
    """``phi = [1, a, s, hash(a (x) s)]``, ``delta = w^T phi``. The hashed
    cross term uses two fixed (non-trainable) random sign projections of
    ``a`` and ``s`` into a shared ``cross_dim``-wide space, multiplied
    elementwise -- a compact-bilinear-pooling style hash of the outer
    product that needs no ``a_dim * summary_dim``-sized table. Required per
    plan section 3.5: "without it, this model is little more than global
    calibration."""

    def __init__(self, a_dim: int, summary_dim: int, cross_dim: int = 32, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        Ra = (torch.randint(0, 2, (a_dim, cross_dim), generator=g).float() * 2 - 1)
        Rs = (torch.randint(0, 2, (summary_dim, cross_dim), generator=g).float() * 2 - 1)
        self.register_buffer("Ra", Ra)
        self.register_buffer("Rs", Rs)
        self.w = nn.Linear(1 + a_dim + summary_dim + cross_dim, 1)
        zero_init_(self.w)

    def forward(self, a, s):
        cross = (a @ self.Ra) * (s @ self.Rs)
        ones = a.new_ones(a.shape[0], 1)
        phi = torch.cat([ones, a, s, cross], dim=-1)
        return self.w(phi).squeeze(-1)


VARIANTS = ("v1_transformer", "v2_gru", "v3_mlp", "v4_bilinear", "v5_linear")


def build_variant(name: str, a_dim: int, token_dim: int, summary_dim: int, cfg: dict):
    """Construct one ladder variant by name from a flat hyperparameter dict
    (see ``withinday/train.py`` for the fields each variant reads)."""
    dm = cfg.get("delta_max", 1.0)
    dp = cfg.get("dropout", 0.0)
    if name == "v1_transformer":
        return V1Transformer(a_dim, token_dim, hidden=cfg.get("hidden", 32),
                             n_heads=cfg.get("n_heads", 2), delta_max=dm, dropout=dp)
    if name == "v2_gru":
        return V2GRU(a_dim, token_dim, hidden=cfg.get("hidden", 32), delta_max=dm, dropout=dp)
    if name == "v3_mlp":
        return V3MLP(a_dim, summary_dim, hidden=cfg.get("mlp_hidden", (64, 32)), delta_max=dm, dropout=dp)
    if name == "v4_bilinear":
        return V4Bilinear(a_dim, summary_dim, rank=cfg.get("rank", 4))
    if name == "v5_linear":
        return V5Linear(a_dim, summary_dim, cross_dim=cfg.get("cross_dim", 32), seed=cfg.get("seed", 0))
    raise ValueError(f"unknown variant {name!r}")
