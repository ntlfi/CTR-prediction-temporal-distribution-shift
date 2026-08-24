"""Liu et al., "On the Adaptation to Concept Drift for CTR Prediction"
(AdaMoE) -- P2 CTR-specific drift-adaptation baseline (PDF 3.7).

AdaMoE's architectural novelty is a closed-form, gradient-free
aggregation-weight update across a mixture of experts: each expert's
per-sample "correctness" y~ = y*yhat + (1-y)*(1-yhat) is normalized across
experts and averaged over a batch, then EMA'd into the running weight
vector. The paper's own backbone (which expert network computes yhat) is
explicitly swappable; the mechanism being tested here is the weight
update, so the experts are the existing P0 window-family candidates
(expanding + rolling 1/3/7/14) rather than a from-scratch neural backbone.

Causality: the prediction for day t uses w_{t-1} (the EMA as of the day
before, computed with a stop-gradient / no-lookahead in the original
paper's Algorithm 1) -- weights are only updated *after* day t's true
labels are observed, so nothing about day t leaks into its own prediction.

Reference implementation: https://github.com/Yuejiang-li/FuxiCTR/tree/liyuejiang/develope
"""
import numpy as np

from baselines import WINDOW_FAMILY

LAMBDA = 0.5  # EMA momentum (paper's best tradeoff among {0, .25, .5, .75, .99})


def run_adamoe(bank: dict, eligible_days, lam: float = LAMBDA):
    names = list(WINDOW_FAMILY)
    m = len(names)
    w = np.full(m, 1.0 / m)

    rows = []
    for t in sorted(eligible_days):
        if not all(t in bank[n] for n in names):
            continue
        preds = np.stack([bank[n][t]["y_pred"] for n in names], axis=1)  # (n, m)
        y_true = bank[names[0]][t]["y_true"]

        agg_pred = preds @ w
        rows.append({
            "day": t,
            "y_true": y_true,
            "y_pred": agg_pred,
            "n_train": int(np.mean([bank[n][t]["n_train"] for n in names])),
            "fit_time": float(sum(bank[n][t]["fit_time"] for n in names)),
            "weights": dict(zip(names, w.tolist())),
        })

        # Update the EMA weights only now that day t's true labels are
        # observed -- used for t+1 onward, never for day t itself.
        y_col = y_true.reshape(-1, 1)
        correctness = y_col * preds + (1 - y_col) * (1 - preds)  # (n, m), in [0,1]
        row_sums = correctness.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-12
        w_batch = (correctness / row_sums).mean(axis=0)
        w = lam * w + (1 - lam) * w_batch

    return rows
