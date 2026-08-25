# Preliminary findings

- Prediction days: 116 total (81 dev, 35 locked test).
- Best baseline on locked test by mean log loss: expanding (log loss 0.4391).
- Validation-selected window on dev period: h=expanding.
- Hindsight best fixed window took 3 distinct value(s) across the 35 test days: ['expanding', 'rolling_3', 'rolling_7'].
- The best horizon changed across test days -- motivates checking whether the adaptive baselines (Han ARW, Differentiable Forgetting) track that variation (see PDF section 5, 10).
