"""Downstream autobidding evaluation -- AMG-TP_Academic_LaTeX.pdf section 8.

The prediction model is *frozen*, then every CTR model is fed into the *same*
bidding + pacing policy on the *same* auction episodes. Any difference in
realised value at matched spend is attributed to CTR-prediction quality, not
to the bidder (PDF section 8, equation 9).

Counterfactual replay methodology
---------------------------------
The Criteo Attribution log records, for every impression Criteo won, its
display ``cost`` and the realised ``click`` / ``conversion`` / ``attribution``.
We simulate an advertiser bidding ``b_i = scale * pctr_i`` into each logged
auction:

  * win   iff  ``b_i >= cost_i``      (``cost_i`` is the price that was needed to win)
  * on a win : pay ``cost_i``, receive the logged ``click_i`` and ``conversion_i``
  * on a loss: pay nothing, receive nothing

This evaluates *down-selection among logged impressions under a budget*: a
better pCTR ranking wins more clicks per dollar. It cannot value impressions
Criteo never won -- which is consistent with PDF section 8's "frozen bidder,
matched spend, attribute differences to the prediction method" framing. It is
the standard replay methodology of the RTB benchmark literature (iPinYou; Cai
et al. 2017; Perlich et al. 2012 linear bidding).

Two policies, both fully paired across CTR models:

``linear_frontier``
    A single global bid ``scale`` knob, swept to trace the value--spend
    frontier (PDF section 8, "the value--spend frontier"). No budget; the
    knob itself is the spend control.

``paced_auction``
    A fixed budget with per-block pacing. ``win iff scale_t * pctr >= cost``
    with ``scale_t`` chosen per block so spend tracks ``budget / n_blocks``
    (unspent budget carries over). Because ``scale * pctr >= cost`` is
    equivalent to ranking impressions by ``pctr / cost`` and taking a prefix,
    the budget-optimal pace within a block is: take impressions in
    decreasing ``pctr / cost`` until the block budget is exhausted. That is
    what a well-tuned PID pacer converges to, implemented here directly so
    the result is deterministic and controller-tuning-free.

Primary metric: **clicks won at matched spend** (the quantity the CTR models
predict). Conversions, attributed conversions and cpo-weighted conversion
value are reported as secondary (PDF section 8).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from data import CAT_COLUMNS, SECONDS_PER_DAY, hash_features, raw_numeric_features

RAW_BID_COLUMNS = ["timestamp", "campaign", "click", "conversion", "attribution",
                   "cost", "cpo"] + [f"cat{i}" for i in range(1, 10)]


# --------------------------------------------------------------------------- #
#  data                                                                       #
# --------------------------------------------------------------------------- #
class BiddingData:
    """Chronological Criteo log with the post-auction columns kept (cost,
    conversion, attribution, cpo) alongside the pre-bid CTR features."""

    __slots__ = ("X", "context", "y", "day", "cost", "conversion",
                 "attribution", "cpo", "campaign")

    def __init__(self, X, context, y, day, cost, conversion, attribution, cpo, campaign):
        self.X, self.context, self.y, self.day = X, context, y, day
        self.cost, self.conversion = cost, conversion
        self.attribution, self.cpo, self.campaign = attribution, cpo, campaign


def load_criteo_bidding(tsv_path: str | Path, sample_frac: float = 1.0, seed: int = 0,
                        n_features: int = 2 ** 18) -> BiddingData:
    """Same sampling / chronological ordering as ``data.load_raw`` (so the CTR
    features are identical to the rest of the project) but keeping the columns
    the auction needs."""
    df = pd.read_csv(tsv_path, sep="\t", usecols=RAW_BID_COLUMNS)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed).sort_values("timestamp")
    df["day"] = (df["timestamp"] // SECONDS_PER_DAY).astype(int)
    df = df.reset_index(drop=True)

    X = hash_features(df, columns=CAT_COLUMNS, n_features=n_features)
    context = raw_numeric_features(df, columns=CAT_COLUMNS)
    to_arr = lambda c, dt: df[c].to_numpy(dtype=dt)
    return BiddingData(
        X=X, context=context,
        y=to_arr("click", np.int64), day=to_arr("day", np.int64),
        cost=to_arr("cost", np.float64),
        conversion=to_arr("conversion", np.int64),
        attribution=to_arr("attribution", np.int64),
        cpo=to_arr("cpo", np.float64),
        campaign=to_arr("campaign", np.int64),
    )


def synthetic_cost(p_true, seed: int = 0, competitiveness: float = 1.0,
                   noise_sd: float = 0.6, floor: float = 1e-5) -> np.ndarray:
    """A synthetic second-price landscape for the drift-injection benchmark
    (``synthetic_data.py`` has no cost column).

    The winning price we must beat is a competing bidder's valuation:
    ``cost_i = floor + scale * p_true_i * lognormal(0, noise_sd)``. Tying the
    market price to the *true* click probability -- not to any model under
    test -- makes clicked-likely impressions genuinely more expensive, which
    is the regime where CTR-prediction skill is supposed to pay off, and
    keeps the auction episodes identical across the CTR models being
    compared. ``scale`` is normalised so mean cost ~= ``competitiveness`` x
    mean ``p_true`` (i.e. a truthful value-per-click of 1.0 roughly breaks
    even at full volume)."""
    p_true = np.asarray(p_true, float)
    rng = np.random.default_rng(seed)
    ln = rng.lognormal(mean=-0.5 * noise_sd ** 2, sigma=noise_sd, size=len(p_true))
    raw = p_true * ln
    scale = competitiveness * p_true.mean() / max(raw.mean(), 1e-12)
    return floor + scale * raw


# --------------------------------------------------------------------------- #
#  auction outcome bookkeeping                                                #
# --------------------------------------------------------------------------- #
_OUTCOME_FIELDS = ("spend", "impressions", "clicks", "conversions",
                   "attributed", "cpo_value")


def _outcome(win_mask, cost, click, conv, attrib, cpo):
    w = win_mask
    return {
        "spend": float(cost[w].sum()),
        "impressions": int(w.sum()),
        "clicks": int(click[w].sum()),
        "conversions": int(conv[w].sum()),
        "attributed": int(attrib[w].sum()),
        # conversion value: campaign cpo paid out per won conversion
        "cpo_value": float((cpo[w] * conv[w]).sum()),
    }


# --------------------------------------------------------------------------- #
#  policy 1: global-scale value--spend frontier                               #
# --------------------------------------------------------------------------- #
def linear_frontier(pctr, click, cost, day, *, conv=None, attrib=None, cpo=None,
                    scales=None, n_scales: int = 40) -> pd.DataFrame:
    """Sweep ``b = scale * pctr`` over a geometric grid of ``scale``. For each
    scale: win iff ``b >= cost``; report spend and value. ``day`` is accepted
    for interface symmetry with ``paced_auction`` (the frontier is
    day-agnostic)."""
    pctr = np.asarray(pctr, float)
    cost = np.asarray(cost, float)
    click = np.asarray(click)
    n = len(pctr)
    conv = np.zeros(n, int) if conv is None else np.asarray(conv)
    attrib = np.zeros(n, int) if attrib is None else np.asarray(attrib)
    cpo = np.zeros(n) if cpo is None else np.asarray(cpo, float)

    if scales is None:
        # scale range that spans "win almost nothing" .. "win almost everything"
        ratio = cost / np.maximum(pctr, 1e-12)
        lo, hi = np.quantile(ratio, 0.001), np.quantile(ratio, 0.999)
        scales = np.geomspace(max(lo, 1e-9), max(hi, 1e-8), n_scales)

    rows = []
    for s in scales:
        win = s * pctr >= cost
        o = _outcome(win, cost, click, conv, attrib, cpo)
        rows.append({"scale": float(s), **o,
                     "win_rate": o["impressions"] / n,
                     "clicks_per_kilo_spend": o["clicks"] / o["spend"] / 1000
                     if o["spend"] > 0 else 0.0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  policy 2: budgeted per-block pacing                                        #
# --------------------------------------------------------------------------- #
def paced_auction(pctr, click, cost, day, budget: float, *,
                  conv=None, attrib=None, cpo=None, carryover: bool = True):
    """Spend ``budget`` over the blocks present in ``day`` with per-block
    pacing. Within a block, take impressions in decreasing ``pctr / cost``
    (the budget-optimal realisation of ``scale_t * pctr >= cost``) until the
    block allowance is spent; unspent allowance carries into the next block
    when ``carryover`` is set. Returns (summary dict, per-block DataFrame)."""
    pctr = np.asarray(pctr, float)
    cost = np.asarray(cost, float)
    click = np.asarray(click)
    day = np.asarray(day)
    n = len(pctr)
    conv = np.zeros(n, int) if conv is None else np.asarray(conv)
    attrib = np.zeros(n, int) if attrib is None else np.asarray(attrib)
    cpo = np.zeros(n) if cpo is None else np.asarray(cpo, float)

    blocks = np.unique(day)
    per_block_budget = budget / len(blocks)
    allowance = 0.0
    win_all = np.zeros(n, bool)
    block_rows = []

    for b in blocks:
        allowance = (allowance + per_block_budget) if carryover else per_block_budget
        idx = np.where(day == b)[0]
        # efficiency ranking; +inf-safe (cost > 0 for every Criteo row, but guard)
        eff = pctr[idx] / np.maximum(cost[idx], 1e-12)
        order = idx[np.argsort(-eff)]
        cum = np.cumsum(cost[order])
        take = order[cum <= allowance]           # strict: never exceed the allowance
        win_all[take] = True
        spent = float(cost[take].sum())
        allowance -= spent
        o = _outcome(np.isin(np.arange(n), take), cost, click, conv, attrib, cpo)
        block_rows.append({"block": int(b), "allowance_after": allowance, **o})

    summary = _outcome(win_all, cost, click, conv, attrib, cpo)
    summary["budget"] = float(budget)
    summary["budget_used_frac"] = summary["spend"] / budget if budget > 0 else 0.0
    summary["budget_violation"] = max(0.0, summary["spend"] - budget)
    return summary, pd.DataFrame(block_rows)


# --------------------------------------------------------------------------- #
#  matched-spend comparison                                                   #
# --------------------------------------------------------------------------- #
def value_at_matched_spend(frontiers: dict, spend_grid, value: str = "clicks") -> pd.DataFrame:
    """Interpolate ``value`` vs ``spend`` for each method's frontier onto a
    common ``spend_grid`` -- the paired "value at matched spend" of PDF
    section 8. ``frontiers`` maps method name -> DataFrame from
    ``linear_frontier``."""
    out = {"spend": np.asarray(spend_grid, float)}
    for name, fr in frontiers.items():
        fr = fr.sort_values("spend")
        out[name] = np.interp(spend_grid, fr["spend"], fr[value],
                              left=np.nan, right=fr[value].iloc[-1])
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
#  reference bidders (frontier anchors, not deployable)                       #
# --------------------------------------------------------------------------- #
def oracle_pctr(click):
    """Perfect foresight: pctr == realised click. Upper bound on the frontier."""
    return np.asarray(click, float)


def noskill_pctr(click):
    """Constant pctr = base rate. Bids purely on price (cheapest first)."""
    return np.full(len(click), float(np.mean(click)))


def shuffled_pctr(pctr, seed: int = 0):
    """A model's own predictions, permuted -- destroys ranking skill, keeps
    the marginal distribution. Lower reference for the frontier."""
    rng = np.random.default_rng(seed)
    return rng.permutation(np.asarray(pctr, float))
