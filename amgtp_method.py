"""AMG-TP -- Adaptive Multi-Timescale Gating with Temporal Persistence
(AMG-TP_Academic_LaTeX.pdf sections 2-3). Extends M5b (m5_multiscale_gate)
by making the gate's day-to-day inertia *adaptive* instead of a fixed
`smooth_reg`:

    q_t(x)   = softmax(g_phi(x, p^(1:K)_t(x), s_{t-1}))         raw context gate (= M5b's gate)
    m_t      = (1 - rho) m_{t-1} + rho * E_x[pi_t(x)]           persistent temporal state (EMA of deployed weights)
    beta_t   = sigmoid(r_psi(s_{t-1})) in [0, 1]                adaptive persistence (one global value per day)
    pi_t(x)  = (1 - beta_t) q_t(x) + beta_t * m_{t-1}           deployed temporal mixture
    p_hat_t(x) = sum_k pi_{t,k}(x) p^(k)_t(x)

Recommended initial implementation (PDF section 3): global beta_t with a
context-dependent q_t(x). Example-specific beta_t(x) is deliberately not
attempted here.

Everything is causal: day t's prediction uses q, beta and m carried from
days < t only; phi, psi and m are updated after day t's labels mature.
`s_{t-1}` (the persistence-model input) is built from strictly-earlier days:
recent per-expert loss, recent short/long disagreement, recent CTR, recent
deployed gate movement, and normalized time.
"""
import numpy as np
import torch
import torch.nn as nn

from baselines import WINDOW_FAMILY
from han_arw import per_sample_log_loss
from m5_multiscale_gate import MultiExpertGate, _entropy, gate_feature_matrix, multi_days


def _bcast(beta):
    """A beta that is scalar (global) or (n,) (per-example) -> shape that
    broadcasts against pi (n, K)."""
    return beta.unsqueeze(-1) if beta.ndim == 1 else beta

PERSIST_STATE_NAMES = (["recent_loss_" + n for n in WINDOW_FAMILY]
                       + ["recent_disagreement", "recent_ctr", "recent_gate_move", "norm_time",
                          "loss_jump", "q_vs_m_div"])


class PersistenceNet(nn.Module):
    """r_psi: day-level state features -> scalar logit -> beta_t in [0,1].

    hidden=0 (default, Stage 2 architecture): a single linear layer. Weights
    zero-initialised and bias = init_bias, so beta_t starts at
    sigmoid(init_bias) (no imposed inertia) and AMG-TP has to *learn* to
    raise persistence where it helps.

    hidden>0 (Extension A): one tanh hidden layer of that width, for a
    nonlinear state -> persistence map (e.g. "drop beta after a loss jump
    *regardless* of the other features"). The *output* layer is still
    zero-initialised (weight 0, bias init_bias), so beta_t again starts at
    exactly sigmoid(init_bias): hidden=0 stays bit-identical to Stage 2 and
    hidden>0 is a strict superset that must learn any curvature it uses.
    """
    def __init__(self, n_features: int, init_bias: float = -1.0, hidden: int = 0):
        super().__init__()
        self.hidden = hidden
        if hidden > 0:
            self.h = nn.Linear(n_features, hidden)
            self.out = nn.Linear(hidden, 1)
        else:
            self.out = nn.Linear(n_features, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, init_bias)

    def logit(self, s: torch.Tensor) -> torch.Tensor:
        if self.hidden > 0:
            s = torch.tanh(self.h(s))
        return self.out(s).squeeze(-1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logit(s))


class BetaContextHead(nn.Module):
    """g_xi: per-example gate features -> a scalar logit *offset* added to the
    global persistence logit before the sigmoid, giving beta_t(x) (Extension
    B, PDF section 3's deferred example-specific persistence).

    Zero-initialised, so at the start g_xi == 0 and beta_t(x) collapses to the
    global beta_t exactly -- the per-example term only moves off zero if it
    lowers loss, and a Var_x[beta_t(x)] penalty pulls it back toward the
    global value. hidden>0 adds one tanh layer (still zero-init output)."""
    def __init__(self, n_features: int, hidden: int = 0):
        super().__init__()
        self.hidden = hidden
        if hidden > 0:
            self.h = nn.Linear(n_features, hidden)
            self.out = nn.Linear(hidden, 1)
        else:
            self.out = nn.Linear(n_features, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        if self.hidden > 0:
            c = torch.tanh(self.h(c))
        return self.out(c).squeeze(-1)


def _expert_mean_loss(bank: dict, d: int) -> np.ndarray:
    return np.array([per_sample_log_loss(bank[n][d]["y_true"], bank[n][d]["y_pred"]).mean()
                     for n in WINDOW_FAMILY])


def _persist_state(bank: dict, day_list: list, t: int, T: int, prev_gate_move: float,
                   prev_mean_q: np.ndarray, m_state: np.ndarray, mode: str = "full") -> np.ndarray:
    """r_psi input, all from days < t: per-expert recent loss, short/long
    disagreement, recent CTR, recent deployed gate movement, normalized time,
    a loss-jump shift detector (max |Δ per-expert loss| between the two most
    recent matured days), and the L1 divergence between the last raw gate
    mean and the persistent state m_{t-1} (large -> m is stale, beta should
    drop)."""
    norm_t = t / max(T, 1)
    if mode == "time_only":
        return np.array([norm_t], dtype=np.float32)
    past = [d for d in day_list if d < t]
    if not past:
        return np.zeros(len(PERSIST_STATE_NAMES), dtype=np.float32)
    prev = past[-1]
    recent_loss = _expert_mean_loss(bank, prev)
    loss_jump = float(np.abs(recent_loss - _expert_mean_loss(bank, past[-2])).max()) if len(past) >= 2 else 0.0
    p_short = bank["rolling_3"][prev]["y_pred"]
    p_long = bank["expanding"][prev]["y_pred"]
    disagreement = float(np.abs(p_short - p_long).mean())
    recent_ctr = float(bank[WINDOW_FAMILY[0]][prev]["y_true"].mean())
    q_vs_m = float(np.abs(np.asarray(prev_mean_q) - np.asarray(m_state)).sum()) if prev_mean_q is not None else 0.0
    return np.concatenate([recent_loss,
                           [disagreement, recent_ctr, prev_gate_move, norm_t, loss_jump, q_vs_m]]).astype(np.float32)


def run_amgtp(bank: dict, eligible_days, T: int, lr: float = 0.05, l2: float = 1e-3,
              entropy_reg: float = 1e-3, rho: float = 0.3, beta_entropy_reg: float = 0.0,
              epochs_per_day: int = 3, seed: int = 0,
              context: np.ndarray = None, day: np.ndarray = None,
              adaptive_beta: bool = True, fixed_beta: float = 0.0, init_bias: float = -1.0,
              context_gate: bool = True, uniform_q: bool = False,
              state_features: str = "full", persist_hidden: int = 0,
              beta_per_example: bool = False, beta_var_reg: float = 0.0,
              beta_hidden: int = 0, group: np.ndarray = None):
    """Returns rows: {day, y_true, y_pred, n_train, fit_time, weights,
    mean_weights (dict expert -> mean deployed pi), beta (mean deployed),
    beta_std, mean_q (dict), m_state (dict)}; if `group` is given also
    beta_A / beta_B (mean deployed beta per subgroup). `y_pred` is the
    deployed AMG-TP mixture for day t (q, beta, m all carried from days < t).

    Ablation switches (PDF Table 3):
      adaptive_beta=False, fixed_beta=b   -> A2/A3: fix persistence at b
      context_gate=False                  -> A4: q uses no per-example context
      uniform_q=True                      -> A5: q is fixed uniform; only beta/m adapt
      state_features='time_only'          -> A7: strip recent loss/shift features from r_psi
      persist_hidden=H                     -> A10: nonlinear (H-wide tanh) persistence net
                                             in place of the linear one (H=0, default)

    Extension B (PDF section 3 -- example-specific persistence):
      beta_per_example=True   -> beta_t(x) = sigmoid(r_psi(s_{t-1}) + g_xi(feats_t(x)));
                                 g_xi zero-init so it starts == the global beta_t
      beta_var_reg=lambda     -> penalty lambda * Var_x[beta_t(x)] (A12: lambda huge
                                 -> collapses back to the global beta_t, an identity check)
      beta_hidden=H           -> g_xi gets an H-wide tanh hidden layer
    """
    torch.manual_seed(seed)
    days = multi_days(bank, eligible_days)
    if not days:
        return []

    gate_context = context if context_gate else None

    K = len(WINDOW_FAMILY)
    n_context = gate_context.shape[1] if gate_context is not None else 0
    n_gate_features = K + 2 + K + n_context  # preds, [spread, norm_time], recent per-expert loss, context
    gate = MultiExpertGate(n_gate_features, K)
    n_state = len(PERSIST_STATE_NAMES) if state_features == "full" else 1
    persist = PersistenceNet(n_state, init_bias=init_bias, hidden=persist_hidden)
    gxi = BetaContextHead(n_gate_features, hidden=beta_hidden) if (adaptive_beta and beta_per_example) else None
    trainable = list(gate.parameters())
    if adaptive_beta:
        trainable += list(persist.parameters())
        if gxi is not None:
            trainable += list(gxi.parameters())
    opt = torch.optim.Adam(trainable, lr=lr) if trainable else None

    def beta_of(feats_tensor):
        """Deployed beta for day t: scalar (global) or (n,) (per-example),
        all from state/context carried from days < t."""
        if not adaptive_beta:
            return torch.tensor(float(fixed_beta))
        g = persist.logit(s_prev)
        if gxi is not None:
            g = g + gxi(feats_tensor)
        return torch.sigmoid(g)

    m_state = torch.full((K,), 1.0 / K)          # m_{t-1}
    prev_gate_move = 0.0
    prev_deployed_mean = None
    prev_mean_q = None

    rows = []
    for t in days:
        feats, preds = gate_feature_matrix(bank, days, t, T, context=gate_context, day=day)
        y_true = bank[WINDOW_FAMILY[0]][t]["y_true"]
        feats_t = torch.tensor(feats, dtype=torch.float32)
        preds_t = torch.tensor(preds, dtype=torch.float32)
        y_t = torch.tensor(y_true, dtype=torch.float32)
        s_prev = torch.tensor(_persist_state(bank, days, t, T, prev_gate_move, prev_mean_q,
                                             m_state.numpy(), state_features),
                              dtype=torch.float32)

        with torch.no_grad():
            q = torch.full((len(y_true), K), 1.0 / K) if uniform_q else gate(feats_t)  # (n, K)
            beta = beta_of(feats_t)
            pi = (1 - _bcast(beta)) * q + _bcast(beta) * m_state.unsqueeze(0)   # (n, K)
            pi_np = pi.numpy()
            q_np = q.numpy()
            beta_arr = np.full(len(y_true), float(beta)) if beta.ndim == 0 else beta.numpy()
            beta_val = float(beta_arr.mean())
            beta_std = float(beta_arr.std())
        y_pred = (preds * pi_np).sum(axis=1)
        mean_pi = pi_np.mean(axis=0)
        mean_q = q_np.mean(axis=0)

        gate_move = float(np.abs(mean_pi - prev_deployed_mean).sum()) if prev_deployed_mean is not None else np.nan
        row = {
            "day": t, "y_true": y_true, "y_pred": y_pred,
            "n_train": int(np.mean([bank[n][t]["n_train"] for n in WINDOW_FAMILY])),
            "fit_time": float(sum(bank[n][t]["fit_time"] for n in WINDOW_FAMILY)),
            "weights": pi_np,
            "mean_weights": dict(zip(WINDOW_FAMILY, mean_pi.tolist())),
            "mean_q": dict(zip(WINDOW_FAMILY, mean_q.tolist())),
            "beta": beta_val,
            "beta_std": beta_std,
            "m_state": dict(zip(WINDOW_FAMILY, m_state.tolist())),
        }
        if group is not None and day is not None:
            g_t = np.asarray(group)[np.asarray(day) == t]
            if len(g_t) == len(beta_arr):
                row["beta_A"] = float(beta_arr[g_t].mean()) if g_t.any() else np.nan
                row["beta_B"] = float(beta_arr[~g_t].mean()) if (~g_t).any() else np.nan
        rows.append(row)

        # --- updates, only now that day t's labels are observed ----------
        if opt is not None:
            for _ in range(epochs_per_day):
                opt.zero_grad()
                q_tr = torch.full((len(y_true), K), 1.0 / K) if uniform_q else gate(feats_t)
                beta_tr = beta_of(feats_t)
                pi_tr = (1 - _bcast(beta_tr)) * q_tr + _bcast(beta_tr) * m_state.unsqueeze(0).detach()
                p_mix = (preds_t * pi_tr).sum(dim=-1).clamp(1e-7, 1 - 1e-7)
                bce = -(y_t * p_mix.log() + (1 - y_t) * (1 - p_mix).log()).mean()
                l2_term = sum((p ** 2).sum() for p in trainable)
                loss = bce + l2 * l2_term
                if not uniform_q:
                    loss = loss + entropy_reg * (-_entropy(q_tr).mean())
                if adaptive_beta and beta_entropy_reg > 0:
                    b = beta_tr.mean().clamp(1e-6, 1 - 1e-6)
                    loss = loss - beta_entropy_reg * (-(b * b.log() + (1 - b) * (1 - b).log()))
                if gxi is not None and beta_var_reg > 0 and beta_tr.ndim == 1:
                    loss = loss + beta_var_reg * beta_tr.var()
                loss.backward()
                opt.step()

        with torch.no_grad():
            q_new = torch.full((len(y_true), K), 1.0 / K) if uniform_q else gate(feats_t)
            beta_new = beta_of(feats_t)
            pi_new = ((1 - _bcast(beta_new)) * q_new + _bcast(beta_new) * m_state.unsqueeze(0)).mean(dim=0)
            m_state = (1 - rho) * m_state + rho * pi_new.detach()
            prev_gate_move = 0.0 if np.isnan(gate_move) else gate_move
            prev_mean_q = mean_q
            prev_deployed_mean = mean_pi

    return rows
