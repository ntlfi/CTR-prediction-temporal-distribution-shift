# DualTime-CTR final experiment — findings

"Experimental Plan to Complete All TBD Results in DualTime-CTR" — 6-method
headline comparison (Expanding, Best Fixed Window, ARW, AdaMoE, OPS,
DualTime-CTR), Criteo Attribution (full data, 16.5M rows) and Avazu (full
data, 40.4M rows — a departure from every earlier line in this repo, which
subsampled Avazu to 20%), 3 seeds per dataset, hyperparameters frozen by
`run_hpo.py` on dev days only (never re-tuned per seed, never touching
test days). Full detail/resumption context: `final_experiments/PROGRESS.md`.

## DualTime-CTR is not the capacity-ladder V5

This distinction matters enough to state before any numbers: DualTime-CTR's
within-day residual weight `w` is **updated online, within the day**:

```
w_{d,i+1} = Pi_W( w_{d,i} - eta_i * grad(l_i(w_{d,i})) )
```

implemented in `dualtime/online.py::replay_day` (`w` resets to 0 every
day, projected gradient step at each matured block, `eta_k = B_w/sqrt(k)`).
This is different from the earlier capacity-ladder **V5**
(`withinday/adapters.py`, the `withinday_experiments/` line), whose `w` is
trained **offline** on historical days and **frozen** for the whole test
period — only the history features change causally there, not the
weights. V5's result motivated DualTime-CTR's `phi(x,h)` feature
architecture (the hashed context x history bilinear interaction) but is
not itself an implementation of DualTime-CTR, and the two must not be
conflated when citing results.

## Headline: locked-test impression-weighted log loss (mean over seeds)

**Criteo** (3 seeds, test days 22–30, `final_experiments/criteo/final/`):

| method | log loss (mean ± sd, 3 seeds) | delta vs Expanding (95% CI) | CI excludes 0? | seed-days won |
|---|---|---|---|---|
| **OPS** | **0.606958 ± 0.000004** | −0.001120 [−0.001243, −0.001011] | yes | 27/27 |
| DualTime-CTR | 0.607070 ± 0.000030 | −0.001010 [−0.001133, −0.000902] | yes | 27/27 |
| AdaMoE | 0.607157 ± 0.000030 | −0.000922 [−0.001022, −0.000830] | yes | 27/27 |
| Best Fixed Window | 0.607290 ± 0.000038 | −0.000790 [−0.000927, −0.000660] | yes | 27/27 |
| ARW | 0.607301 ± 0.000045 | −0.000782 [−0.000923, −0.000646] | yes | 27/27 |
| Expanding | 0.608067 ± 0.000011 | — | — | — |

**Avazu** (3 seeds, test days 7–9, `final_experiments/avazu/final/`):

| method | log loss (mean ± sd, 3 seeds) | delta vs Expanding (95% CI) | CI excludes 0? | seed-days won |
|---|---|---|---|---|
| **AdaMoE** | **0.387402 ± 0.000050** | −0.000211 [−0.000299, −0.000117] | **yes** | 8/9 |
| OPS | 0.387443 ± 0.000058 | −0.000201 [−0.000433, +0.000057] | no | 6/9 |
| ARW | 0.387530 ± 0.000043 | −0.000076 [−0.000172, +0.000013] | no | 4/9 |
| Expanding | 0.387596 ± 0.000021 | — | — | — |
| DualTime-CTR | 0.387610 ± 0.000078 | −0.000041 [−0.000300, +0.000251] | no | 6/9 |
| Best Fixed Window | 0.388147 ± 0.000214 | +0.000443 [−0.000058, +0.001006] | no | 4/9 |

`log loss (mean ± sd)` is the impression-weighted per-seed log loss's
standard deviation across the 3 seeds (`std_across_seeds` in
`headline_results.csv`) — how much each method's own score moved seed to
seed. "delta vs Expanding" is a separate quantity: it pools (seed, test
day) as the replicate unit and reports the mean day-level log-loss
difference with its 95% bootstrap CI, via `withinday.daystats.day_summary`
— see caveat below. The two shouldn't be conflated: a method can have a
small seed-sd (stable score) yet a wide delta-CI (noisy day-to-day
comparison against Expanding), as Best Fixed Window's Avazu row shows.

## Interpretation

**Criteo carries the adaptive-training story; Avazu does not.** On Criteo
every adaptive method beats plain Expanding decisively and unanimously
(27/27 seed-days each); DualTime-CTR is a clear #2, beaten only by OPS's
plain global online scalar calibration — i.e. DualTime-CTR's online
within-day residual correction adds nothing over OPS here, consistent with
this whole project's standing finding of shallow real intraday drift on
Criteo (same conclusion `twoscale`'s `combined ≈ long_only` and
`withinday`'s sub-materiality result already reached by a different route).

On Avazu the picture is much weaker: only AdaMoE clears significance, and
even that is a small (~0.05%) effect; OPS, ARW, and DualTime-CTR all trend
in the right direction but every CI crosses zero at 3 seeds. **DualTime-CTR
shows no reproducible edge on Avazu** in this final protocol — essentially
tied with Expanding. This does not contradict earlier Avazu results in this
repo (AMG-TP beating Han ARW, or capacity-ladder V5 beating Online Platt on
a 20%-subsampled Avazu) — those are different methods, different baselines,
and a different (5x smaller) data slice. It specifically means: under this
plan's exact 6-method/full-data/3-seed protocol, DualTime-CTR's online
within-day module does not reproduce a significant win on Avazu.

A metric-weighting note on the Avazu DualTime-CTR row: its impression-
weighted log loss (0.387610) is fractionally *worse* than Expanding's
(0.387596) while its day-equal-weighted delta is fractionally *better*
(−0.000041). Both are real, both are reported — the two metrics weight
days differently and can disagree in sign at a margin this small. Not a
leak (checked against the leakage test suite, see `final_experiments/
leakage_tests.txt`, 17/17 pass).

## Hyperparameters used (frozen by `run_hpo.py`, dev days only)

| | Criteo | Avazu |
|---|---|---|
| Best Fixed Window | roll7 | roll3 |
| ARW delta | 0.05 | 0.05 |
| AdaMoE lambda | 0.0 | 0.0 |
| shared mixture (eta, halflife) | 150.0, 3.0 | **1e6 (degenerate — ~equal weighting), 3.0** |
| OPS (B, eta0, schedule) | 0.25, 0.3, const | 0.25, 0.3, const |
| DualTime B_w | 2.0 | 4.0 |

Avazu's mixture eta landing on the grid's degenerate "weight everything
~equally" extreme is itself consistent with this project's standing
finding that the adaptive cross-day mixture doesn't help on Avazu.

## What's NOT covered by this document

- Rolling-origin confirmation (spec section 13, different day ranges than
  the older exploratory `withinday_experiments/rolling/criteo` run).
- Day-level statistical tables/figures beyond the inline CIs above
  (sections 14, 17–18).
- Paper text (section 20) — no paper source file has been located in this
  repo.

See `final_experiments/PROGRESS.md` for the live status of all of the
above and how to resume.
