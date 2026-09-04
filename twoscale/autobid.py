"""Downstream matched-budget autobidding evaluation (plan section 7.3).

Frozen CTR models -> the *same* auction + pacing policy on the *same* logged
Criteo Attribution auctions. Any difference in clicks won at matched spend is
attributed to CTR-prediction quality.

Counterfactual replay: for a logged impression with display ``cost`` (the
price that was needed to win it) and realised ``click``, an advertiser
bidding ``b = scale * pctr`` wins iff ``b >= cost``; on a win it pays
``cost`` and receives the logged ``click``. This values *down-selection among
logged impressions under a budget* -- exactly section 7.3's "frozen bidder,
matched spend" framing. Standard RTB replay methodology (Cai et al. 2017).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import CRITEO_CAT_COLUMNS, SECONDS_PER_DAY, _hash_features

_BID_COLS = ["timestamp", "click", "conversion", "attribution", "cost"] + CRITEO_CAT_COLUMNS


@dataclass
class BiddingData:
    X: object
    y: np.ndarray
    day: np.ndarray
    sec_in_day: np.ndarray
    cost: np.ndarray
    conversion: np.ndarray
    attribution: np.ndarray

    def day_slice(self, d):
        lo, hi = np.searchsorted(self.day, [d, d + 1])
        return slice(int(lo), int(hi))


def load_criteo_bidding(tsv_path, n_features: int = 2 ** 18,
                        sample_frac: float = 1.0, seed: int = 0) -> BiddingData:
    df = pd.read_csv(tsv_path, sep="\t", usecols=_BID_COLS)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)
    df["day"] = (df["timestamp"] // SECONDS_PER_DAY).astype(np.int64)
    df["sec_in_day"] = (df["timestamp"] % SECONDS_PER_DAY).astype(np.int64)
    df = df.sort_values(["day", "sec_in_day"], kind="stable").reset_index(drop=True)
    X = _hash_features(df, CRITEO_CAT_COLUMNS, n_features)
    return BiddingData(X=X, y=df["click"].to_numpy(np.int8),
                       day=df["day"].to_numpy(), sec_in_day=df["sec_in_day"].to_numpy(),
                       cost=df["cost"].to_numpy(float),
                       conversion=df["conversion"].to_numpy(np.int8),
                       attribution=df["attribution"].to_numpy(np.int8))


def linear_frontier(pctr, click, cost, conv, n_scales: int = 40):
    pctr = np.asarray(pctr, float); cost = np.asarray(cost, float)
    click = np.asarray(click); conv = np.asarray(conv)
    ratio = cost / np.maximum(pctr, 1e-12)
    scales = np.geomspace(max(np.quantile(ratio, 0.001), 1e-9),
                          max(np.quantile(ratio, 0.999), 1e-8), n_scales)
    rows = []
    for s in scales:
        win = s * pctr >= cost
        rows.append({"scale": float(s), "spend": float(cost[win].sum()),
                     "impressions": int(win.sum()), "clicks": int(click[win].sum()),
                     "conversions": int(conv[win].sum())})
    return pd.DataFrame(rows)


def paced_auction(pctr, click, cost, day, budget: float):
    """Per-day pacing: within each day take impressions in decreasing
    pctr/cost until the day's budget share (plus carry-over) is spent."""
    pctr = np.asarray(pctr, float); cost = np.asarray(cost, float)
    click = np.asarray(click); day = np.asarray(day)
    n = len(pctr)
    days = np.unique(day)
    per_day = budget / len(days)
    allowance = 0.0
    win = np.zeros(n, bool)
    for b in days:
        allowance += per_day
        idx = np.where(day == b)[0]
        order = idx[np.argsort(-(pctr[idx] / np.maximum(cost[idx], 1e-12)))]
        take = order[np.cumsum(cost[order]) <= allowance]
        win[take] = True
        allowance -= float(cost[take].sum())
    return {"spend": float(cost[win].sum()), "clicks": int(click[win].sum()),
            "impressions": int(win.sum()), "budget": float(budget),
            "budget_used_frac": float(cost[win].sum() / budget)}


def value_at_matched_spend(frontiers: dict, spend_grid, value: str = "clicks"):
    out = {"spend": np.asarray(spend_grid, float)}
    for name, fr in frontiers.items():
        fr = fr.sort_values("spend")
        out[name] = np.interp(spend_grid, fr["spend"], fr[value],
                              left=np.nan, right=fr[value].iloc[-1])
    return pd.DataFrame(out)
