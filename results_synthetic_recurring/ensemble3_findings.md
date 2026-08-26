# 3-way ensemble (M2/M5b/M5b-high-smooth) findings

- ensemble3 is a 3-way meta-gate (ensemble3.py) blending M2, M5b-default (smooth_reg=1e-3), and M5b-high-smooth (smooth_reg=0.1) -- built after the smooth_reg sweep found M5b-high-smooth beats every method on recurring drift but regresses on abrupt/local.
- ensemble3 locked-test log loss 0.4133; final-day mean weights [('m2', 0.09), ('m5b', 0.06), ('m5b_hs', 0.85)], top expert m5b_hs.
- M5b-high-smooth (standalone) locked-test log loss 0.4118.
- Best method overall on locked test (including ensemble3): m5b_high_smooth (log loss 0.4118).
