"""The 30-minute (1,800 s) label-availability rule, enforced identically
everywhere a training / selection / update cutoff reads labels.

A day-``k`` impression at ``sec_in_day = s`` produces its label at absolute
time ``k * 86400 + s + delay_sec``. At any cutoff ``t`` (an absolute time,
here always a day boundary ``cutoff_day * 86400``) the label is usable iff

    k * 86400 + s + delay_sec  <=  cutoff_day * 86400 .

`available_mask` applies exactly that. `day_matured_mask` is the special
case for a per-day summary whose earliest use is the next midnight
(``cutoff_day = k + 1``): ``s <= 86400 - delay_sec`` -- i.e. drop the
last ``delay_sec`` seconds of the day. Pending (not-yet-matured) labels
are never discarded from ``bank[d].y`` / ``bank[d].sec_in_day``; they are
simply excluded from any cutoff that precedes their arrival and are
picked up by the next cutoff that follows it.
"""
from __future__ import annotations

import numpy as np

DELAY_SEC = 1800
SECONDS_PER_DAY = 86_400


def available_mask(day_arr, sec_arr, cutoff_day: int, delay_sec: int = DELAY_SEC) -> np.ndarray:
    """Boolean mask: impressions whose label has matured by the start of
    ``cutoff_day`` (absolute cutoff ``cutoff_day * 86400``)."""
    day_arr = np.asarray(day_arr, np.int64)
    sec_arr = np.asarray(sec_arr, np.int64)
    return day_arr * SECONDS_PER_DAY + sec_arr + delay_sec <= int(cutoff_day) * SECONDS_PER_DAY


def day_matured_mask(sec_arr, delay_sec: int = DELAY_SEC) -> np.ndarray:
    """Boolean mask for one day's impressions matured by the *next*
    midnight -- the earliest cutoff at which that day's summary is used."""
    return np.asarray(sec_arr, np.int64) <= SECONDS_PER_DAY - delay_sec
