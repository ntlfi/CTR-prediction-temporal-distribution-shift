# Adaptive Training under Temporal Distribution Shift
## AI-Agent Implementation and Empirical Screening Plan

### 1. Objective

Implement and compare a focused family of adaptive-training methods for supervised learning under temporal distribution shift. The application is click-through-rate (CTR) prediction, with autobidding reserved as a downstream evaluation after a promising training method has been identified.

The empirical goal is **not** to force every proposed method to work. The goal is to determine whether any adaptive-training mechanism produces a clear, reproducible improvement over strong recency-based baselines.

The central hypothesis is:

> Different parts of the input space may require different temporal memory lengths. Stable patterns benefit from long-term data, while drifting patterns require recent data.

The primary new methods should therefore combine **short-term** and **long-term** information in different ways.

---

## 2. Scope and decision rule

### Phase A: prediction only

Implement all methods in this document and evaluate them on:

1. controlled synthetic temporal-shift environments;
2. the chronological Criteo CTR dataset already used in the project.

Do **not** add autobidding experiments until at least one method shows convincing predictive gains.

### Phase B: downstream autobidding

Only if one or more adaptive methods pass the prediction gate, plug their CTR predictions into the **same fixed standard bidder/pacing implementation** used for all methods.

### Prediction gate

A method is considered promising if it satisfies most of the following:

- improves mean log loss over the strongest adaptive baseline;
- has no meaningful downside under stationary or weak-shift settings;
- improves recovery after abrupt shift;
- shows a clear advantage under local/subpopulation shift;
- gains are reproducible across seeds;
- results are not explained by a substantially larger model or extra future information.

---

# 3. Existing baselines to freeze

Do not modify these baselines while testing the new methods.

### B0. Expanding-history ERM
Train on all available past data.

### B1. Fixed rolling windows
At minimum:
- rolling-1;
- rolling-3;
- rolling-7;
- rolling-14.

### B2. Validation-selected fixed window
Select one horizon using development data and freeze it for test.

### B3. Exponential forgetting
Use fixed temporal decay with several half-lives.

### B4. Han Adaptive Rolling Window (ARW)
Treat this as the **primary adaptive baseline**.

Current finding: ARW reacts strongly to abrupt drift and does not show meaningful downside when no drift is present.

### B5. AdaMoE
Treat this as the primary CTR-specific adaptive baseline.

Current finding: it adapts, but more softly/slower than ARW.

### B6. Differentiable Forgetting
Keep as a secondary adaptive baseline.

Current finding: it reacts to drift but stale data retains nonzero influence, causing slower recovery.

### B7. SFTL
Keep as a secondary/negative baseline. Do not invest further unless required for a final comparison.

Current finding: it is difficult to make competitive and appears vulnerable to confidence/calibration instability.

---

# 4. Common experimental interface

Every method must implement a common interface.

```python
class TemporalModel:
    def fit_initial(self, history):
        ...

    def update(self, history, recent_block, time_index):
        ...

    def predict_proba(self, X):
        ...

    def get_diagnostics(self):
        ...
```

Each method must only use information available up to the prediction cutoff.

For day/block `t`:

1. training information is restricted to `D_1, ..., D_{t-1}`;
2. the method updates its state;
3. it predicts `D_t`;
4. labels from `D_t` become available only after prediction/evaluation;
5. move to `t+1`.

No random train/test split is allowed for temporal experiments.

---

# 5. Common short- and long-memory definitions

Start with exactly two temporal scales for Methods M1-M4 and M6.

Recommended initial definitions:

- **Short memory:** last 3 days/blocks.
- **Long memory:** expanding history or last 14 days/blocks.

Also run a sensitivity check with:
- short = 1 day;
- long = 7 days.

Keep the same short/long definitions across methods within each comparison.

Let:

\[
p_t^S(x)
\]

denote the short-memory prediction and

\[
p_t^L(x)
\]

denote the long-memory prediction.

Use the same base model family for both branches unless a method explicitly requires otherwise.

Start with logistic regression for synthetic experiments and the initial Criteo screening. After a method looks promising, repeat the strongest comparisons with one standard nonlinear CTR model.

---

# 6. Method M1 — Global Adaptive Short/Long Mixing

## Goal

Learn one global coefficient at time `t` controlling how much to trust the short-memory versus long-memory predictor.

Prediction:

\[
\hat p_t(x)
=
(1-\alpha_t)p_t^L(x)
+
\alpha_t p_t^S(x),
\qquad
\alpha_t\in[0,1].
\]

## M1a. Oracle diagnostic

Before learning `alpha_t`, calculate the hindsight value:

\[
\alpha_t^*
=
\arg\min_{\alpha\in[0,1]}
L_t\left((1-\alpha)p_t^L+\alpha p_t^S\right).
\]

This is diagnostic only and must never be used as a deployable method.

It quantifies whether adaptive mixing has headroom.

## M1b. Validation-driven adaptive alpha

Estimate `alpha_t` only from recent past blocks.

For example, choose alpha from:

```text
{0.00, 0.10, 0.20, ..., 1.00}
```

using the most recent matured validation block(s), then use it for the next block.

## M1c. Learned global gate

Construct temporal state features such as:

- recent short-model log loss;
- recent long-model log loss;
- their difference;
- short/long prediction disagreement;
- recent CTR;
- change in CTR;
- feature-distribution drift statistic;
- previous alpha.

Train a small gating model:

\[
\alpha_t=\sigma(g_\phi(s_t)).
\]

Use historical rolling episodes `(state at t -> best alpha on t+1)` to train the gate.

## Required diagnostics

Plot:
- `alpha_t`;
- short-model loss;
- long-model loss;
- short/long disagreement;
- shift points.

---

# 7. Method M2 — Context-Dependent Short/Long Gating

## Goal

Allow different examples at the same time to use different memory scales.

Prediction:

\[
\hat p_t(x)
=
[1-\alpha_t(x)]p_t^L(x)
+
\alpha_t(x)p_t^S(x).
\]

This is a primary candidate method.

## Gate inputs

Start with a compact gate:

\[
\alpha_t(x)
=
\sigma(g_\phi(u_t(x))).
\]

Candidate features:

1. short prediction `p_S`;
2. long prediction `p_L`;
3. absolute disagreement `|p_S-p_L|`;
4. signed disagreement `p_S-p_L`;
5. selected original/context features or low-dimensional embedding;
6. recent group-level short loss;
7. recent group-level long loss;
8. global shift statistic;
9. time since start / normalized time index.

Avoid giving the gate outcome information from the current prediction block.

## Training the gate

Use rolling historical meta-examples.

At historical cutoff `t`:

1. fit short and long predictors using data available through `t`;
2. predict the next block `D_{t+1}`;
3. construct gate inputs using only information available at `t`;
4. optimize gate parameters according to loss on `D_{t+1}`.

Two possible implementations:

### M2a. Direct mixture-loss training
Backpropagate next-block log loss through the gate while keeping the short/long predictors fixed for that meta-step.

### M2b. Oracle-alpha supervision
For each validation example, calculate which alpha on a fixed grid minimizes its next-block log loss, then train the gate to approximate that target.

Prefer M2a if stable.

## Regularization

Prevent pathological gating:

- entropy regularization or weak shrinkage toward `alpha=0.5`;
- optional temporal smoothness penalty:
  \[
  \lambda_\alpha |\alpha_t(x)-\alpha_{t-1}(x)|;
  \]
- cap minimum data per group if group-level features are used.

## Required diagnostics

Report distributions of `alpha_t(x)`:
- before shift;
- immediately after shift;
- during recovery;
- separately for shifted and stable subpopulations.

---

# 8. Method M3 — Long-Term Backbone + Short-Term Residual

## Goal

Explicitly decompose persistent structure from temporary adaptation.

Model:

\[
f_t(x)
=
f^L(x)
+
\Delta_t^S(x).
\]

For binary CTR prediction:

\[
\hat p_t(x)
=
\sigma(f^L(x)+\Delta_t^S(x)).
\]

## M3a. Ungated residual

1. fit the long-term backbone using long history;
2. freeze or slowly update it;
3. train the residual using recent data;
4. regularize the residual toward zero.

Objective:

\[
L_t
=
L_{\text{CTR}}
+
\lambda_r\|\theta_S\|_2^2.
\]

## M3b. Gated residual

Preferred version:

\[
f_t(x)
=
f^L(x)
+
\alpha_t(x)\Delta_t^S(x).
\]

Use a gate similar to M2.

Interpretation:
- the long backbone is the default;
- the short residual is activated only where current evidence suggests drift.

## Update schedules to test

Backbone:
- frozen after initial training;
- update once per 7 blocks;
- low learning rate continual update.

Residual:
- update every block using short window;
- optionally reset at each block versus warm-start.

## Required diagnostics

Track:
- residual norm over time;
- gate activation;
- performance of backbone-only;
- residual-only contribution;
- recovery after recurring regimes.

---

# 9. Method M4 — Adaptive Temporal Sample Weighting

## Goal

Adapt the training distribution instead of combining predictions.

Train:

\[
\theta_t
=
\arg\min_\theta
\sum_{i<t}
w_{i,t}
\ell(f_\theta(x_i),y_i).
\]

## M4a. Adaptive global decay

Use:

\[
w_{i,t}=\exp[-\gamma_t(t-t_i)].
\]

Learn or select `gamma_t` from recent past performance.

This is mainly an intermediate baseline.

## M4b. Context-dependent decay

Use:

\[
w_{i,t}
=
\exp[-\gamma_t(x_i)(t-t_i)].
\]

Parameterize:

\[
\gamma_t(x)=\text{softplus}(g_\phi(x,s_t)).
\]

This allows stable regions to retain old observations while drifting regions forget them quickly.

## M4c. Two-timescale weighting

Define fixed short and long kernels:

\[
w_{i,t}^S=e^{-\gamma_S(t-t_i)},
\]

\[
w_{i,t}^L=e^{-\gamma_L(t-t_i)},
\qquad \gamma_S>\gamma_L.
\]

Learn:

\[
w_{i,t}
=
\alpha_t(x_i)w_{i,t}^S
+
[1-\alpha_t(x_i)]w_{i,t}^L.
\]

This version most directly connects to the short/long-memory hypothesis.

## Implementation note

Use normalized or clipped weights to prevent a tiny number of observations from dominating training.

Record effective sample size:

\[
ESS_t
=
\frac{(\sum_iw_i)^2}{\sum_iw_i^2}.
\]

---

# 10. Method M5 — Multi-Timescale Mixture of Temporal Experts

## Goal

Generalize short/long memory to multiple temporal scales.

Use experts:

\[
h\in\{1,3,7,14,\text{expanding}\}.
\]

Each expert produces:

\[
p_t^{(h)}(x).
\]

Prediction:

\[
\hat p_t(x)
=
\sum_h
\pi_{t,h}(x)p_t^{(h)}(x),
\]

with:

\[
\sum_h\pi_{t,h}(x)=1.
\]

## M5a. Global gate

\[
\pi_{t,h}=\text{softmax}(g_\phi(s_t)).
\]

## M5b. Context-dependent gate

\[
\pi_{t,h}(x)=\text{softmax}(g_\phi(x,s_t,p_t^{(1:K)}(x))).
\]

This is more flexible but should be implemented only after M2 is stable.

## Regularization

Test:
- entropy regularization;
- mild sparsity encouragement;
- no regularization.

Avoid forcing hard selection initially.

## Diagnostics

Plot:
- average expert weights over time;
- expert weights by shifted versus stable group;
- effective selected horizon:
  \[
  \bar h_t(x)=\sum_h\pi_{t,h}(x)h.
  \]

For expanding history, assign a clearly documented nominal horizon only for visualization, not for training.

---

# 11. Method M6 — Adaptive Gradient Mixing

## Goal

Combine short- and long-memory learning signals at the optimization level.

Compute:

\[
g_t^S=\nabla_\theta L_t^S(\theta),
\]

\[
g_t^L=\nabla_\theta L_t^L(\theta).
\]

Update using:

\[
g_t
=
\alpha_t g_t^S
+
(1-\alpha_t)g_t^L.
\]

Then:

\[
\theta_{t+1}
=
\theta_t-\eta g_t.
\]

## Drift signal

Compute gradient cosine similarity:

\[
c_t
=
\frac{
\langle g_t^S,g_t^L\rangle
}{
\|g_t^S\|\|g_t^L\|
}.
\]

Interpretation:
- `c_t` near 1: short and long histories agree;
- `c_t` near 0: weak agreement;
- `c_t` below 0: recent and historical data push the model in conflicting directions.

## M6a. Rule-based alpha

Start with a deterministic rule such as:

```text
if cosine_similarity >= high_threshold:
    alpha = small
elif cosine_similarity <= low_threshold:
    alpha = large
else:
    interpolate
```

Tune thresholds only on development environments.

## M6b. Learned alpha

Learn:

\[
\alpha_t
=
\sigma(g_\phi(
c_t,
\|g_t^S\|,
\|g_t^L\|,
L_t^S,
L_t^L,
s_t
)).
\]

This method is easiest with a differentiable PyTorch base model.

## Diagnostics

Record:
- short/long gradient norms;
- cosine similarity;
- alpha;
- loss before and after shifts.

---

# 12. Synthetic benchmark

Synthetic experiments are the primary mechanism tests because natural drift in the current Criteo window is weak.

Use at least 120 temporal blocks/days.

Run at least 10 seeds for cheap experiments.

## S0. Stationary

No distribution shift.

Purpose:
- methods should not beat ERM through leakage;
- adaptive methods should have minimal downside.

## S1. Abrupt global drift

At known time `t_shift`:

\[
P(Y|X)
\]

changes globally.

Purpose:
- measure reaction speed;
- Han ARW is the key baseline.

## S2. Gradual drift

Change model parameters continuously over time.

Purpose:
- test whether soft adaptation has an advantage over hard switching.

## S3. Recurring drift

Regimes:

```text
A -> B -> A
```

Purpose:
- test whether retaining long-term information enables faster recovery than hard window truncation.

This is especially important for M3 and M5.

## S4. Local/subpopulation drift

Partition examples into groups A and B.

At the shift:

```text
A_old -> A_new
B remains unchanged
```

Purpose:
- this is the primary test for context-dependent methods M2, M3b, M4b/M4c, and M5b.

Expected behavior:
- short memory should dominate for A;
- long memory should remain useful for B.

## S5. Opposing local drift

Two groups drift differently or at different times.

Purpose:
- stress-test a single global memory coefficient.

---

# 13. Oracle diagnostics

Compute these only for analysis; never present them as deployable methods.

## O1. Best fixed horizon

Best single horizon over the evaluation period.

## O2. Per-block oracle horizon

\[
h_t^*
=
\arg\min_h L_t(h).
\]

## O3. Per-group oracle horizon

\[
h_{t,g}^*
=
\arg\min_h L_{t,g}(h).
\]

## O4. Per-example short/long oracle mixture

For M2 diagnostics, determine whether short or long prediction gives lower individual log loss.

The gap between deployable methods and these oracle quantities defines the remaining adaptation headroom.

---

# 14. Criteo protocol

Use the existing chronological Criteo pipeline.

Important current finding:

> Natural shift over approximately one month is shallow; rolling-7 is difficult to beat.

Therefore Criteo should test **robustness/no downside**, not be the only evidence that adaptation works.

## Procedure

For each chronological test block:

1. update models only from previous data;
2. predict current CTR;
3. record metrics;
4. move forward.

Do not random-shuffle across time.

## Primary metrics

1. log loss;
2. Brier score;
3. PR-AUC.

## Secondary

- ROC-AUC;
- calibration/reliability;
- subgroup log loss;
- daily/block-level loss.

---

# 15. Metrics for temporal adaptation

In addition to aggregate predictive metrics, compute:

## Recovery time

After a known synthetic shift, number of blocks until loss returns within a chosen tolerance of the post-shift oracle/static-best performance.

## Peak post-shift excess loss

\[
\max_{t\in \text{recovery window}}
[L_t-L_t^{oracle}].
\]

## Cumulative post-shift regret

\[
\sum_{t=t_{shift}}^{t_{shift}+H}
(L_t-L_t^{oracle}).
\]

## Stationary downside

Difference relative to strongest static baseline when no drift occurs.

## Local adaptation gap

For local-drift experiments report separately:

\[
L_A,\qquad L_B.
\]

A method should improve group A without unnecessarily damaging stable group B.

---

# 16. Hyperparameter policy

Do not conduct unbounded tuning.

Use a shared development suite and freeze all hyperparameters before confirmatory test runs.

Recommended tuning priority:

1. temporal horizon definitions;
2. gate capacity;
3. gate regularization;
4. learning rates;
5. residual regularization or weighting temperature.

For neural gates, start small:
- linear/logistic gate;
- one hidden layer;
- only add depth if clearly necessary.

A method that only works with a large search should be considered less promising.

---

# 17. Statistical protocol

For synthetic experiments:
- use paired seeds;
- at least 10 seeds for lightweight models;
- report mean, standard error or 95% interval.

For Criteo:
- report chronological block-level metrics;
- aggregate with day/block bootstrap if needed;
- do not treat individual impressions as independent experimental replicates.

Every comparison should use identical temporal episodes and random seeds whenever possible.

---

# 18. Required ablations

For any method that looks promising, run:

## A1. Short only

## A2. Long only

## A3. Fixed 50/50 mixture

## A4. Global adaptive mixture

## A5. Context-dependent adaptive version

This isolates whether the gain comes from:
- maintaining two models;
- mixing them;
- adapting the mixture;
- making adaptation local/context dependent.

For M3 additionally run:
- backbone only;
- residual without gate;
- gated residual.

For M5:
- global gate;
- local gate;
- number of experts.

---

# 19. Implementation order

The AI agent should implement in this order.

### Stage 0 — verify benchmark infrastructure

Confirm that the existing:
- rolling windows;
- Han ARW;
- AdaMoE;
- Differentiable Forgetting;
- SFTL results

are reproducible from one command/config pipeline.

### Stage 1 — M1 global adaptive mixing

This is the cheapest proof of concept.

### Stage 2 — M2 context-dependent gating

This is the primary candidate.

### Stage 3 — M3 gated residual

This is the second primary candidate.

### Stage 4 — M6 gradient mixing

Provides a conceptually different optimization-level approach.

### Stage 5 — M4 adaptive sample weighting

Implement global decay first, then local/two-timescale versions.

### Stage 6 — M5 multi-timescale mixture

Implement only after the two-expert gate is stable.

---

# 20. Repository structure

Recommended:

```text
project/
├── configs/
│   ├── synthetic/
│   ├── criteo/
│   └── methods/
├── data/
├── src/
│   ├── datasets/
│   ├── models/
│   │   ├── base_ctr.py
│   │   ├── temporal_experts.py
│   │   └── gates.py
│   ├── methods/
│   │   ├── global_mix.py
│   │   ├── context_gate.py
│   │   ├── residual_adapter.py
│   │   ├── temporal_weighting.py
│   │   ├── multiscale_moe.py
│   │   └── gradient_mix.py
│   ├── baselines/
│   ├── evaluation/
│   └── utils/
├── scripts/
│   ├── run_synthetic.py
│   ├── run_criteo.py
│   ├── run_all_methods.py
│   └── make_report.py
├── tests/
└── outputs/
    ├── metrics/
    ├── figures/
    └── checkpoints/
```

---

# 21. Required tests

Before accepting a method:

1. **No future leakage test**
   - perturb future labels;
   - predictions before those labels become available must remain unchanged.

2. **Alpha/gate range test**
   - all mixing weights stay in valid range.

3. **Mixture identity tests**
   - alpha=0 reproduces long model;
   - alpha=1 reproduces short model.

4. **Stationary sanity test**
   - adaptive method should not materially outperform an oracle through accidental leakage;
   - gate should not show unexplained shift behavior.

5. **Synthetic local-shift test**
   - verify only the designated subpopulation distribution changes.

6. **Reproducibility**
   - fixed seed gives identical metrics within expected numerical tolerance.

---

# 22. Required output tables

Generate one consolidated table per environment:

| Method | Log Loss | Brier | PR-AUC | Recovery | Stationary Downside |
|---|---:|---:|---:|---:|---:|

For local drift additionally:

| Method | Shifted Group Loss | Stable Group Loss | Overall Loss |
|---|---:|---:|---:|

---

# 23. Required figures

At minimum:

1. daily/block log-loss curves around abrupt shift;
2. recovery curves for all candidate methods;
3. short vs long baseline performance;
4. gate/alpha trajectory for M1;
5. gate distribution by shifted vs stable subgroup for M2/M3;
6. per-block oracle horizon vs Han ARW chosen horizon;
7. recurring-shift performance;
8. Criteo chronological log loss.

For any promising method, visualize how its adaptive mechanism changes at the known shift time.

---

# 24. Ranking candidate methods

After all methods run, score each candidate on:

1. synthetic abrupt drift;
2. synthetic recurring drift;
3. synthetic local drift;
4. stationary downside;
5. Criteo performance;
6. consistency across seeds;
7. implementation complexity;
8. interpretability;
9. novelty potential.

Do not choose the winner using aggregate log loss alone.

A method is especially promising if it demonstrates a **qualitatively new capability**, for example:

- matching Han ARW on global abrupt drift;
- beating Han ARW on local drift;
- beating hard windows on recurring regimes;
- doing so without hurting stationary/Criteo performance.

---

# 25. Go / no-go outcomes

## Outcome A — Context-dependent gating wins

If M2 or M3b clearly beats global methods under local drift while remaining competitive elsewhere:

> Continue with context-dependent temporal memory as the main research direction.

## Outcome B — Global adaptive mixing is enough

If M1 matches all more complex methods:

> Prefer the simpler global adaptive method and investigate theory/robustness rather than architecture complexity.

## Outcome C — Han ARW remains essentially optimal

If Han ARW nearly matches oracle performance across all relevant environments:

> Stop developing a new adaptive-memory method. The empirical headroom is insufficient.

## Outcome D — Only multi-timescale mixture wins

If M5 provides clear gains over two-timescale methods:

> Investigate learned instance-dependent memory horizon as the central method.

## Outcome E — No method helps on real data but synthetic gains are clear

> Add a second real chronological dataset with stronger natural temporal shift before making broad ML claims.

---

# 26. Autobidding follow-up gate

Do not implement new autobidding algorithms.

If a prediction method passes the previous gates:

1. freeze the CTR models;
2. produce CTR predictions chronologically;
3. feed every method into the same established bidding/pacing policy;
4. compare realized value at matched spend;
5. attribute differences only to CTR prediction/adaptation.

The final causal story should remain:

\[
\text{adaptive training}
\rightarrow
\text{better CTR prediction under shift}
\rightarrow
\text{better downstream autobidding}.
\]

---

# 27. Final deliverables from the AI agent

Produce:

1. fully runnable implementations of M1-M6;
2. tests for leakage and edge cases;
3. configuration files for all experiments;
4. one consolidated CSV/Parquet results table;
5. all required figures;
6. a Markdown report containing:
   - which methods worked;
   - where each method worked;
   - confidence intervals;
   - failures;
   - comparison against Han ARW and AdaMoE;
   - recommended method to continue developing;
7. exact reproduction commands.

The report must explicitly answer:

> Is there empirical evidence that adaptive combination of short- and long-term information can outperform strong recency-based temporal adaptation, and if so, under what kinds of distribution shift?

Do not declare a new research method successful unless the evidence is reproducible and the comparison against Han ARW is favorable.
