"""Zhu et al., "Generalize for Future: Slow and Fast Trajectory Learning
for CTR Prediction" (SFTL), AAAI 2024 -- P2 CTR-specific baseline (PDF 3.7,
the alternative to AdaMoE).

Confirmed from the AAAI paper (ojs.aaai.org/index.php/AAAI/article/view/27797):
three copies of one architecture-agnostic model, trained over a stream of
temporal domains (here, one domain = one day):

  - Working learner (theta_w): trained normally, one gradient step per
    minibatch.
  - Slow learner (theta_s): hard-copied from theta_w once per domain, at
    the domain boundary (eq. 7) -- a snapshot of "where the model stood
    at the end of the last stable period."
  - Fast learner (theta_f): EMA of theta_w, updated every minibatch
    (eq. 8): theta_f <- alpha*theta_f + (1-alpha)*theta_w.

Once past a warmup number of domains, a trajectory loss (eq. 9) is added
to the working learner's BCE loss: a soft bipartite-ranking loss pushing
theta_w's positive-vs-negative margin to exceed whatever margin theta_s /
theta_f currently achieve on the same batch. Both targets are
stop-gradient (only theta_w receives gradients from this term); the two
terms are weighted separately (eq. 11): L = L_ce + lambda_s*L_tra(vs
slow) + lambda_f*L_tra(vs fast). At inference, the FAST learner is served
("it captures more short-term temporal information" -- paper's words).

Unlike every other P1/P2 method here, SFTL is not re-fit per prediction
day from scratch: it is ONE continuously-trained model walked through the
full chronological day sequence, so this module owns its own streaming
loop (run_sftl) rather than plugging into candidate_bank.py's per-day
fit_predict pattern.

Not disclosed in the accessible paper text (deferred to an appendix that
could not be retrieved): the EMA coefficient alpha, the loss weights
lambda_s/lambda_f, and the warmup length. The defaults below are
reasonable choices, not reproductions of the authors' own tuning --
documented in README.md.

Stability note (found empirically, not in the paper): lambda_s/lambda_f
need to be small (default 0.05). The trajectory loss has no equilibrium on
its own -- since the slow learner is a hard copy of the working learner at
each domain boundary, "beat the slow learner's margin" becomes "beat your
own slightly-more-confident-than-last-time past self" every domain, a
positive feedback loop. At lambda=1.0 this makes predicted probabilities
escalate toward 0/1 within a handful of domains (log loss > 4 by the time
locked-test evaluation starts); at lambda=0.05 the BCE term's calibration
pull dominates and log loss stays stable and comparable to a no-trajectory-
loss ablation over a 60-domain run. This may just reflect that the paper's
own lambda_s/lambda_f (undisclosed) are similarly small.
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CTRNet(nn.Module):
    """Embedding + MLP CTR model. The paper's framework is explicitly
    architecture-agnostic (their own experiments use a DCN-Mix backbone,
    embedding rank 16); this is a simpler embedding-concat + MLP model,
    sized down from their 1024-512-256 hidden stack for CPU feasibility
    at this benchmark's data scale.
    """

    def __init__(self, n_columns: int, vocab_size: int, embed_dim: int = 16, hidden=(128, 64)):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size, embed_dim) for _ in range(n_columns)])
        layers, prev = [], n_columns * embed_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_idx: torch.Tensor) -> torch.Tensor:
        """x_idx: (batch, n_columns) long. Returns raw logits (batch,)."""
        embs = [emb(x_idx[:, j]) for j, emb in enumerate(self.embeddings)]
        h = torch.cat(embs, dim=1)
        return self.mlp(h).squeeze(-1)


class SFTL:
    def __init__(self, n_columns: int, vocab_size: int = 2**16, embed_dim: int = 16, hidden=(128, 64),
                 lr: float = 1e-3, weight_decay: float = 1e-5, ema_alpha: float = 0.9,
                 lambda_slow: float = 0.05, lambda_fast: float = 0.05, warmup_domains: int = 3,
                 batch_size: int = 512, seed: int = 0, device: str = "cpu"):
        torch.manual_seed(seed)
        self.device = device
        self.working = CTRNet(n_columns, vocab_size, embed_dim, hidden).to(device)
        self.slow = copy.deepcopy(self.working)
        self.fast = copy.deepcopy(self.working)
        for net in (self.slow, self.fast):
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.working.parameters(), lr=lr, weight_decay=weight_decay)
        self.bce = nn.BCEWithLogitsLoss()
        self.ema_alpha = ema_alpha
        self.lambda_slow = lambda_slow
        self.lambda_fast = lambda_fast
        self.warmup_domains = warmup_domains
        self.batch_size = batch_size
        self.domain_idx = 0
        self.n_seen = 0

    def predict_fast(self, x_idx: np.ndarray) -> np.ndarray:
        """Fast learner's predictions -- what the paper serves at inference,
        since it "captures more short-term temporal information."""
        self.fast.eval()
        with torch.no_grad():
            x = torch.as_tensor(x_idx, dtype=torch.long, device=self.device)
            return torch.sigmoid(self.fast(x)).cpu().numpy()

    def _trajectory_loss(self, working_logits: torch.Tensor, target_net: nn.Module,
                          x_idx: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Eq. 9: soft bipartite-ranking loss over (positive, negative)
        pairs in the batch. Pushes the working learner's margin above
        whatever margin the (stop-gradient) target model currently gets."""
        pos, neg = y == 1, y == 0
        if pos.sum() == 0 or neg.sum() == 0:
            return working_logits.sum() * 0.0
        with torch.no_grad():
            target_logits = target_net(x_idx)
        u = working_logits[pos].unsqueeze(1) - working_logits[neg].unsqueeze(0)
        v = target_logits[pos].unsqueeze(1) - target_logits[neg].unsqueeze(0)
        # log(1+exp(-(u-v))) = softplus(-(u-v)); softplus's stable
        # formulation avoids the exp() overflow -> NaN that a literal
        # log1p(exp(...)) hits once margins grow beyond ~large values,
        # which happens easily with randomly-initialized embeddings.
        return F.softplus(-(u - v)).mean()

    def train_domain(self, x_idx: np.ndarray, y: np.ndarray, rng: np.random.Generator, epochs: int = 1):
        """`epochs` passes over one domain's (day's) data. The paper reports
        results for both a one-pass streaming setting (epochs=1) and a
        standard multi-epoch setting -- at this benchmark's smaller
        per-domain row counts (thousands, not the paper's much larger
        industrial daily volumes), one pass gives too few gradient steps
        for the embeddings to leave their random initialization, so the
        default here uses their multi-epoch variant instead (see
        --epochs-per-domain in run_sftl.py)."""
        self.working.train()
        n = len(y)
        x_all = torch.as_tensor(x_idx, dtype=torch.long, device=self.device)
        y_all = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        use_trajectory = self.domain_idx >= self.warmup_domains

        for _ in range(epochs):
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb, yb = x_all[idx], y_all[idx]
                logits = self.working(xb)
                loss = self.bce(logits, yb)
                if use_trajectory:
                    loss = (loss + self.lambda_slow * self._trajectory_loss(logits, self.slow, xb, yb)
                                 + self.lambda_fast * self._trajectory_loss(logits, self.fast, xb, yb))
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.working.parameters(), max_norm=5.0)
                self.opt.step()
                with torch.no_grad():
                    for pf, pw in zip(self.fast.parameters(), self.working.parameters()):
                        pf.mul_(self.ema_alpha).add_(pw, alpha=1 - self.ema_alpha)

        self.slow.load_state_dict(self.working.state_dict())  # domain-boundary hard copy (eq. 7)
        self.domain_idx += 1
        self.n_seen += n


def run_sftl(x_idx: np.ndarray, y: np.ndarray, day: np.ndarray, eligible_days,
             n_columns: int, vocab_size: int = 2**16, warmup_domains: int = 3,
             epochs_per_domain: int = 1, seed: int = 0, **sftl_kwargs):
    """Walks the full chronological day range once. Predicts each day in
    `eligible_days` with the fast learner BEFORE training on that day (so
    only strictly-earlier days inform the prediction), then trains on that
    day's now-revealed labels. Domains not in `eligible_days` (the warmup
    days) are still trained on, so the model has real history by the time
    predictions start -- matching how every other method here treats
    warmup days as history-only, never scored.
    """
    eligible = set(int(d) for d in eligible_days)
    model = SFTL(n_columns=n_columns, vocab_size=vocab_size, warmup_domains=warmup_domains,
                 seed=seed, **sftl_kwargs)
    rng = np.random.default_rng(seed)

    rows = []
    for t in range(int(day.min()), int(day.max()) + 1):
        mask = day == t
        if mask.sum() == 0:
            continue
        x_t, y_t = x_idx[mask], y[mask]

        if t in eligible:
            start = time.time()
            y_pred = model.predict_fast(x_t)
            rows.append({
                "day": t, "y_true": y_t, "y_pred": y_pred,
                "n_train": model.n_seen, "fit_time": time.time() - start,
            })

        model.train_domain(x_t, y_t, rng, epochs=epochs_per_domain)

    return rows
