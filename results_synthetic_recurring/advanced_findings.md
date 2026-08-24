# P1/P2 findings

- P1/P2 methods: han_arw (Han et al. adaptive rolling window, PDF 3.5), diff_forgetting (Differentiable Forgetting, PDF 3.6), adamoe (AdaMoE-style closed-form mixture-of-experts, PDF 3.7).
- Best method overall on locked test by mean log loss: expanding (log loss 0.4215).
- Best P1/P2 method: han_arw (log loss 0.4215).
- Han ARW selected windows used across prediction days: ['expanding', 'rolling_14', 'rolling_3', 'rolling_7'].
- Differentiable Forgetting learned half-life ranged 1.75-90.46 days (mean 62.75).
