"""Oracle diagnostics and temporal-adaptation metrics for the AMG-TP
experimental plan (AMG-TP_Academic_LaTeX.pdf sections 5.5 and 5.6, and
adaptive-training-methods-implementation-plan.md sections 13 and 15).

Everything here is analysis-only. The oracle quantities use the current
day's own labels and must never be fed back into a deployed prediction --
they exist to bound the remaining adaptation headroom.
"""
import numpy as np

from baselines import WINDOW_FAMILY
from han_arw import per_sample_log_loss

# Documented nominal horizons for the effective-horizon diagnostic only
# (plan section 10: "assign a clearly documented nominal horizon only for
# visualization, not for training"). `expanding` gets a large finite stand-in.
NOMINAL_HORIZON = {"rolling_1": 1, "rolling_3": 3, "rolling_7": 7, "rolling_14": 14, "expanding": 60}


def day_loss(y_true, y_pred):
    return float(per_sample_log_loss(np.asarray(y_true), np.asarray(y_pred)).mean())


# --------------------------------------------------------------------------
# Oracle diagnostics (plan section 13 / PDF 5.6)
# --------------------------------------------------------------------------

def per_day_oracle_horizon(bank: dict, t: int, mask: np.ndarray = None):
    """O2: h*_t = argmin_h L_t(h) over the fixed WINDOW_FAMILY horizons,
    using day t's own labels (diagnostic only). `mask` optionally restricts
    to a subgroup (O3). Returns (best_name, best_loss)."""
    best_name, best_loss = None, np.inf
    for name in WINDOW_FAMILY:
        if t not in bank[name]:
            continue
        yt = bank[name][t]["y_true"]
        yp = bank[name][t]["y_pred"]
        if mask is not None:
            yt, yp = yt[mask], yp[mask]
        if len(yt) == 0:
            continue
        loss = day_loss(yt, yp)
        if loss < best_loss:
            best_name, best_loss = name, loss
    return best_name, best_loss


def best_fixed_horizon(bank: dict, test_days) -> tuple:
    """O1: the single fixed horizon with the lowest mean loss over the whole
    locked-test period (chosen in hindsight)."""
    scores = {}
    for name in WINDOW_FAMILY:
        losses = [day_loss(bank[name][t]["y_true"], bank[name][t]["y_pred"])
                  for t in test_days if t in bank[name]]
        if losses:
            scores[name] = float(np.mean(losses))
    best = min(scores, key=scores.get)
    return best, scores[best], scores


def oracle_per_day_frame(bank: dict, eligible_days, group: np.ndarray = None,
                         day: np.ndarray = None, m5b_low=None, m5b_high=None) -> list:
    """One row per prediction day with the per-day oracle horizon (overall and,
    if `group` given, per subgroup) and -- if the two M5b smoothness configs
    are supplied -- the per-day oracle persistence regime (which fixed
    smoothness would have been better on that day). Diagnostic only."""
    low_by_day = {r["day"]: r for r in (m5b_low or [])}
    high_by_day = {r["day"]: r for r in (m5b_high or [])}
    rows = []
    for t in sorted(eligible_days):
        if not all(t in bank[n] for n in WINDOW_FAMILY):
            continue
        h_name, h_loss = per_day_oracle_horizon(bank, t)
        rec = {"day": t, "oracle_horizon": h_name, "oracle_horizon_loss": h_loss,
               "oracle_horizon_nominal": NOMINAL_HORIZON[h_name]}
        if group is not None and day is not None:
            g = group[day == t]
            for label, m in (("A", g), ("B", ~g)):
                if m.sum() == 0:
                    rec[f"oracle_horizon_{label}"] = None
                    rec[f"oracle_horizon_loss_{label}"] = np.nan
                    continue
                gn, gl = per_day_oracle_horizon(bank, t, mask=m)
                rec[f"oracle_horizon_{label}"] = gn
                rec[f"oracle_horizon_loss_{label}"] = gl
        if t in low_by_day and t in high_by_day:
            ll = day_loss(low_by_day[t]["y_true"], low_by_day[t]["y_pred"])
            hl = day_loss(high_by_day[t]["y_true"], high_by_day[t]["y_pred"])
            rec["m5b_low_loss"] = ll
            rec["m5b_high_loss"] = hl
            rec["oracle_persistence"] = "high" if hl < ll else "low"
        rows.append(rec)
    return rows


# --------------------------------------------------------------------------
# Adaptation metrics (plan section 15 / PDF 5.5)
# --------------------------------------------------------------------------

def _loss_series(rows: list, days: list) -> np.ndarray:
    by_day = {r["day"]: r for r in rows}
    return np.array([day_loss(by_day[t]["y_true"], by_day[t]["y_pred"])
                     if t in by_day else np.nan for t in days])


def adaptation_metrics(method_rows: list, oracle_rows: list, test_days: list,
                       shift_days: list, horizon: int = 30, tol: float = 0.02,
                       patience: int = 3) -> dict:
    """Recovery time, peak post-shift excess loss, and cumulative post-shift
    regret for one method, all measured against the per-day oracle horizon
    (plan O2). `shift_days` is the list of known change points for the regime
    (empty for stationary / recurring / gradual, where only the aggregate
    excess-over-oracle is meaningful)."""
    days = sorted(test_days)
    oracle_by_day = {r["day"]: r["oracle_horizon_loss"] for r in oracle_rows}
    L = _loss_series(method_rows, days)
    Lo = np.array([oracle_by_day.get(t, np.nan) for t in days])
    excess = L - Lo

    out = {
        "mean_excess_over_oracle": float(np.nanmean(excess)),
        "max_excess_over_oracle": float(np.nanmax(excess)) if np.isfinite(excess).any() else np.nan,
    }

    recoveries, peaks, regrets = [], [], []
    for sd in shift_days:
        win = [i for i, t in enumerate(days) if sd <= t <= sd + horizon]
        if not win:
            continue
        w_excess = excess[win]
        peaks.append(float(np.nanmax(w_excess)))
        regrets.append(float(np.nansum(w_excess)))
        # recovery: first day in-window where excess <= tol for `patience` in a row
        rec = np.nan
        run = 0
        for j, i in enumerate(win):
            if np.isfinite(excess[i]) and excess[i] <= tol:
                run += 1
                if run >= patience:
                    rec = days[win[j - patience + 1]] - sd
                    break
            else:
                run = 0
        recoveries.append(rec)

    if peaks:
        out["peak_post_shift_excess"] = float(np.nanmean(peaks))
        out["cumulative_post_shift_regret"] = float(np.nanmean(regrets))
        out["recovery_time_days"] = float(np.nanmean(recoveries)) if np.isfinite(recoveries).any() else np.nan
    return out


def gate_dynamics(weight_rows: list, expert_names: list) -> list:
    """Per-day mean gate weights, L1 gate movement ||mean_pi_t - mean_pi_{t-1}||_1,
    and effective horizon h_eff = sum_h mean_pi_h * nominal(h) for a gated method
    whose rows carry `mean_weights` {expert: weight}."""
    rows = []
    prev = None
    for r in sorted(weight_rows, key=lambda r: r["day"]):
        mw = r["mean_weights"]
        vec = np.array([mw.get(n, 0.0) for n in expert_names])
        move = float(np.abs(vec - prev).sum()) if prev is not None else np.nan
        prev = vec
        rec = {"day": r["day"], "gate_move_l1": move}
        for n in expert_names:
            rec[f"pi_{n}"] = float(mw.get(n, 0.0))
        if all(n in NOMINAL_HORIZON for n in expert_names):
            rec["h_eff"] = float(sum(mw.get(n, 0.0) * NOMINAL_HORIZON[n] for n in expert_names))
        rows.append(rec)
    return rows


# --------------------------------------------------------------------------
# Aggregation helpers -- operate on an already-computed per-day metrics frame
# (columns: method, day, is_test, <metric>..., n) so day_metrics is never
# recomputed during aggregation.
# --------------------------------------------------------------------------

def weighted_mean_df(df, key: str, weight: str = "n") -> float:
    if df is None or len(df) == 0:
        return np.nan
    v = df[key].to_numpy(dtype=float)
    w = df[weight].to_numpy(dtype=float)
    ok = np.isfinite(v) & np.isfinite(w)
    if not ok.any():
        return np.nan
    return float(np.average(v[ok], weights=w[ok]))


def block_bootstrap_ci_df(df, key: str = "log_loss", weight: str = "n",
                          n_boot: int = 2000, seed: int = 0) -> tuple:
    """Day-level (block) bootstrap 95% CI: resample whole days with
    replacement (plan section 17). `df` is one method's locked-test per-day
    rows."""
    if df is None or len(df) == 0:
        return (np.nan, np.nan, np.nan)
    v = df[key].to_numpy(dtype=float)
    w = df[weight].to_numpy(dtype=float)
    ok = np.isfinite(v) & np.isfinite(w)
    v, w = v[ok], w[ok]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = len(v)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = (v[idx] * w[idx]).sum(axis=1) / w[idx].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(np.average(v, weights=w)), float(lo), float(hi))
