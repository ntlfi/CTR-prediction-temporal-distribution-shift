# Preliminary findings

- Prediction days: 27 total (19 dev, 8 locked test).
- Best baseline on locked test by mean log loss: rolling_7 (log loss 0.6072).
- Validation-selected window on dev period: h=rolling_7.
- Hindsight best fixed window took 3 distinct value(s) across the 8 test days: ['rolling_14', 'rolling_3', 'rolling_7'].
- The best horizon changed across test days -- motivates checking whether the adaptive baselines (Han ARW, Differentiable Forgetting) track that variation (see PDF section 5, 10).
