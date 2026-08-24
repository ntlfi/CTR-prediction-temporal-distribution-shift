# Findings: when recency helps, when history helps, where adaptive methods stand

Full run on the Criteo Attribution dataset: 16.47M rows, 31 days, click rate
0.361, 27 prediction days (19 dev, 8 locked test). Numbers below are
locked-test mean log loss unless noted. Source: `all_methods_comparison_table.csv`,
`hindsight_best_window.csv`, `han_arw_selected_window.csv`,
`diff_forgetting_eta.csv`, `adamoe_expert_weights.csv`.

## Recency helps, but only mildly, and only up to a point

The P0 ladder spans log loss 0.6072 (`rolling_7`, the best) to 0.6163
(`decay_hl1`, the worst) — a real but modest range. Recency helps: `rolling_7`
beats `expanding` (0.6080) and every decay half-life. But recency can be
overdone: `rolling_1` (0.6083) and `decay_hl1` (0.6163, a 1-day half-life)
are both worse than using more history, because a single day of impressions
is too little data — variance from small sample size outweighs any benefit
from freshness. The sweet spot in this ladder is a 7-to-14-day window, not
the shortest one available.

## Long history is a close second, not a clear loser

`expanding` (all history) and `decay_hl7` (a 7-day half-life, i.e. gentle
forgetting) sit only 0.0008–0.0016 log loss behind `rolling_7`. The gap
between "best window" and "most data" is small relative to the gap between
either of them and the aggressive-forgetting baselines. That already
suggests this 31-day dataset does not drift sharply: if it did, maximum-data
`expanding` would be markedly worse than the best recency rule, not nearly
tied with it.

## The hindsight diagnostic: real variation, but a shallow one

`hindsight_best_window.csv` shows the best fixed window changes across test
days — `rolling_7` (5/8 days), `rolling_3` (1/8), `rolling_14` (2/8) — which
is the PDF's stated trigger for "checking whether adaptive baselines track
that variation" (PDF §5, §10). But the three candidates it alternates between
are all within ~0.0002–0.0007 log loss of each other on any given day (see
the P0 comparison table); this is not a case where picking the wrong window
costs much.

## Adaptive methods (P1/P2): each rediscovers the same answer, none beats it by much

- **Han ARW** (`han_arw_selected_window.csv`): after two early days on
  `expanding` (days 7–8, too little history for its bias/variance tournament
  to prefer a shorter window yet), it selects `rolling_7` for every
  remaining day, including all 8 locked-test days. Its formal
  Goldenshluger–Lepski tournament independently re-derives exactly the same
  answer as the naive dev-period average (`validation_selected(h=rolling_7)`)
  — a good sanity check that the statistical procedure works, but it adds no
  information the simple frozen baseline didn't already have. Log loss ties
  `rolling_7` to 5 decimal places (0.607159 vs 0.607159).
- **Differentiable Forgetting** (`diff_forgetting_eta.csv`): the learned
  half-life sits at ~74 days — barely different from "don't forget" — for 23
  of 25 evaluated days, with two small dips (day 20: 48.4 days; day 26: 59.0
  days) that don't change the outcome materially. The bilevel search
  robustly converges to near-stationary weighting regardless of which day's
  local validation slice it sees, which is itself informative: it is telling
  us the outer-loss surface doesn't reward aggressive forgetting anywhere in
  this window. Result (0.6080) lands essentially on top of `expanding`
  (0.6080), which is the base rate its learned weighting reduces to.
- **AdaMoE** (`adamoe_expert_weights.csv`): the only P1/P2 method to edge
  past `rolling_7` (0.60697 vs 0.60716 — a ~0.0002 improvement), but its
  expert weights *never move far from uniform* (all five weights stay within
  ±0.0005 of 0.2 through the entire test period). The gain here is
  ensemble variance-reduction across five correlated window-length models,
  not the drift-tracking re-weighting the method exists to demonstrate — the
  EMA never finds a reason to concentrate weight on any one expert, because
  no expert is decisively and persistently better than the others.

## Where adaptive methods "fail" — and the go/no-go read (PDF §10)

None of the three P1/P2 methods fail outright, but none clearly succeed
either: all three land within 0.0011 log loss of `rolling_7`, which itself
is within 0.0002 of the naive frozen validation-selected baseline. That is
much closer to the PDF's explicit **no-go signal** — "one simple fixed
window or fixed decay rate dominates across essentially all periods" — than
to any of its **strong-motivation** cases (adaptive methods failing to track
a genuinely varying regime, slow recovery after abrupt shifts, or prediction
metrics diverging from downstream bidding value). This 31-day window shows
mild, shallow drift, and every method capable of adapting to it converges to
roughly the same answer a static 7-day window already gives.

**Caveat:** this is one dataset over one relatively short horizon. It's a
legitimate data point against inventing a new method *right now*, but it is
a weak test of what these adaptive methods are actually built for — genuine,
possibly abrupt distribution shift over a longer time horizon. The next
useful step before either committing to "no-go" or designing a new method is
to re-run this same benchmark on a dataset with a longer time span (weeks to
months, ideally spanning a known shift such as a holiday, a platform change,
or a seasonal effect), where a fixed window's disadvantage relative to
adaptive memory would have room to actually show up.
