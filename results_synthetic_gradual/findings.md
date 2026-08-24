# Preliminary findings

- Prediction days: 116 total (81 dev, 35 locked test).
- Best baseline on locked test by mean log loss: rolling_14 (log loss 0.3543).
- Validation-selected window on dev period: h=rolling_14.
- Hindsight best fixed window took 1 distinct value(s) across the 35 test days: ['rolling_14'].
- One fixed horizon dominated every test day -- no empirical motivation yet for adaptive memory from this diagnostic alone (see PDF section 5, 10).
