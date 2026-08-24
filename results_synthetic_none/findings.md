# Preliminary findings

- Prediction days: 116 total (81 dev, 35 locked test).
- Best baseline on locked test by mean log loss: expanding (log loss 0.3107).
- Validation-selected window on dev period: h=expanding.
- Hindsight best fixed window took 1 distinct value(s) across the 35 test days: ['expanding'].
- One fixed horizon dominated every test day -- no empirical motivation yet for adaptive memory from this diagnostic alone (see PDF section 5, 10).
