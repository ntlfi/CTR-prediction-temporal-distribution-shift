# P1/P2 findings

- P1/P2 methods: han_arw (Han et al. adaptive rolling window, PDF 3.5), diff_forgetting (Differentiable Forgetting, PDF 3.6), adamoe (AdaMoE-style closed-form mixture-of-experts, PDF 3.7).
- Best method overall on locked test by mean log loss: adamoe (log loss 0.6070).
- Best P1/P2 method: adamoe (log loss 0.6070).
- Han ARW selected windows used across prediction days: ['expanding', 'rolling_7'].
- Differentiable Forgetting learned half-life ranged 48.37-74.40 days (mean 72.74).
