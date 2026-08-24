"""Shared chronological day-split logic, used by every runner so P0/P1/P2
methods are compared on identical dev/test partitions (PDF section 4)."""
import numpy as np


def compute_splits(day: np.ndarray, warmup_days: int, test_frac: float):
    """Returns (eligible_days, dev_days, test_days).

    `eligible_days` skips the first `warmup_days` (too little history to be
    meaningful); the last `test_frac` of those are the locked test period.
    """
    eligible_days = np.arange(warmup_days + 1, day.max() + 1)
    n_test_days = max(1, int(round(len(eligible_days) * test_frac)))
    dev_days = set(eligible_days[:-n_test_days])
    test_days = set(eligible_days[-n_test_days:])
    return eligible_days, dev_days, test_days
