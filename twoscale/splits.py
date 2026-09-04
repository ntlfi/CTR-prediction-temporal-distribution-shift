"""Chronological split (plan section 3), scaled to the dataset's day count.

The plan's nominal split is days 1-60 / 61-81 / 82-116 of a 116-day log:
about 52% initial-training, 18% development, 30% locked test. No real CTR
dataset here has 116 days (Criteo has 31, Avazu ~10), so the same
*proportions* are applied to whatever ``n_days`` the data provides.

* ``train`` days: never predicted on or evaluated -- they only ever serve as
  history for the models fitted on later days.
* ``dev`` days: every hyperparameter and protocol choice is made here.
* ``test`` days: the locked test period, scored exactly once.

A minimum ``warmup`` is enforced so the earliest predicted day still has a
few days of history behind it even on a short dataset.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRAIN_FRAC = 60 / 116
DEV_FRAC = 21 / 116
TEST_FRAC = 35 / 116


@dataclass
class Split:
    train_days: np.ndarray   # history only, never predicted
    dev_days: np.ndarray     # hyperparameter selection
    test_days: np.ndarray    # locked, scored once
    eval_days: np.ndarray    # dev_days + test_days, sorted (every predicted day)


def make_split(n_days: int, warmup: int = 4,
               train_frac: float = TRAIN_FRAC, dev_frac: float = DEV_FRAC) -> Split:
    all_days = np.arange(n_days)
    n_train = max(warmup, int(round(n_days * train_frac)))
    n_dev = max(1, int(round(n_days * dev_frac)))
    n_train = min(n_train, n_days - n_dev - 1)

    train = all_days[:n_train]
    dev = all_days[n_train:n_train + n_dev]
    test = all_days[n_train + n_dev:]
    return Split(train_days=train, dev_days=dev, test_days=test,
                 eval_days=np.concatenate([dev, test]))
