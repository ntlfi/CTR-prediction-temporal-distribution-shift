"""Chronological, leakage-safe CTR data with an explicit *within-day* time
axis -- the piece the two-timescale plan needs that the day-indexed loaders
elsewhere in the repo do not expose.

Two real datasets:

* **Criteo Attribution** (primary). ``timestamp`` is seconds from the start
  of the log; ``day = timestamp // 86400`` and ``sec_in_day = timestamp %
  86400`` give a true second-resolution within-day arrival order. Only
  pre-bid context (campaign + cat1..cat9) is used as features.

* **Avazu** (secondary, plan section 11 step 10). Native resolution is one
  hour (``hour`` = YYMMDDHH). ``day`` is the calendar day, ``sec_in_day`` is
  the hour-of-day in seconds. Only pre-bid context columns are kept; ``id``
  and ``hour`` are dropped so nothing hands the model the clock.

Both loaders return a :class:`Dataset` with row-aligned arrays, sorted by
(day, sec_in_day) so a day's slice is already in arrival order.
"""
from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher

SECONDS_PER_DAY = 86_400

CRITEO_CAT_COLUMNS = ["campaign"] + [f"cat{i}" for i in range(1, 10)]
CRITEO_RAW_COLUMNS = ["timestamp", "click"] + CRITEO_CAT_COLUMNS

AVAZU_CAT_COLUMNS = [
    "C1", "banner_pos", "site_id", "site_domain", "site_category",
    "app_id", "app_domain", "app_category",
    "device_id", "device_ip", "device_model", "device_type", "device_conn_type",
    "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
]


@dataclass
class Dataset:
    X: "scipy.sparse.csr_matrix"   # hashed pre-bid context features
    y: np.ndarray                  # click label {0, 1}
    day: np.ndarray                # integer calendar-day index (0-based)
    sec_in_day: np.ndarray         # arrival time within the day, in seconds [0, 86400)
    name: str                      # "criteo" / "avazu"

    def __post_init__(self):
        n = len(self.y)
        assert self.X.shape[0] == n == len(self.day) == len(self.sec_in_day)
        # arrival order within every day
        assert np.all(np.diff(self.day) >= 0), "rows must be day-sorted"

    @property
    def n_days(self) -> int:
        return int(self.day.max()) + 1

    def day_slice(self, d: int) -> slice:
        lo, hi = np.searchsorted(self.day, [d, d + 1])
        return slice(int(lo), int(hi))


def _hash_features(df: pd.DataFrame, columns, n_features: int):
    hasher = FeatureHasher(n_features=n_features, input_type="string")
    tokens = df[columns].astype(str)
    for col in columns:
        tokens[col] = col + "=" + tokens[col]
    return hasher.transform(tokens.values.tolist())


# --------------------------------------------------------------------------- #
#  Criteo Attribution                                                          #
# --------------------------------------------------------------------------- #
def load_criteo(tsv_path: str | Path, n_features: int = 2 ** 18,
                sample_frac: float = 1.0, seed: int = 0) -> Dataset:
    df = pd.read_csv(tsv_path, sep="\t", usecols=CRITEO_RAW_COLUMNS)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)
    df["day"] = (df["timestamp"] // SECONDS_PER_DAY).astype(np.int64)
    df["sec_in_day"] = (df["timestamp"] % SECONDS_PER_DAY).astype(np.int64)
    df = df.sort_values(["day", "sec_in_day"], kind="stable").reset_index(drop=True)
    X = _hash_features(df, CRITEO_CAT_COLUMNS, n_features)
    return Dataset(X=X, y=df["click"].to_numpy(np.int8),
                   day=df["day"].to_numpy(), sec_in_day=df["sec_in_day"].to_numpy(),
                   name="criteo")


# --------------------------------------------------------------------------- #
#  Avazu                                                                       #
# --------------------------------------------------------------------------- #
_AVAZU_SPLIT_FILES = ["train.csv", "valid.csv", "test.csv"]
_AVAZU_USECOLS = ["click", "hour"] + AVAZU_CAT_COLUMNS
_AVAZU_CHUNK = 2_000_000


def _avazu_handles(src: Path):
    src = Path(src)
    if src.is_dir():
        for f in _AVAZU_SPLIT_FILES:
            if (src / f).exists():
                with open(src / f, "rb") as fh:
                    yield fh
        return
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as zf:
            names = {Path(n).name: n for n in zf.namelist()}
            for f in _AVAZU_SPLIT_FILES:
                if f in names:
                    with zf.open(names[f]) as fh:
                        yield fh
        return
    raise FileNotFoundError(f"{src} is neither Avazu_x4.zip nor a dir of split CSVs")


def load_avazu(src: str | Path, n_features: int = 2 ** 18,
               sample_frac: float = 0.2, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    read_kw = dict(usecols=_AVAZU_USECOLS,
                   dtype={"click": np.int8, "hour": np.int64,
                          **{c: "string" for c in AVAZU_CAT_COLUMNS}})
    parts = []
    for fh in _avazu_handles(src):
        for chunk in pd.read_csv(fh, chunksize=_AVAZU_CHUNK, **read_kw):
            if sample_frac < 1.0:
                chunk = chunk.loc[rng.random(len(chunk)) < sample_frac]
            parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    if not len(df):
        raise FileNotFoundError(f"no rows read from {src}")

    h = df["hour"].to_numpy(np.int64)
    ts = pd.to_datetime({"year": 2000 + h // 1_000_000, "month": (h // 10_000) % 100,
                         "day": (h // 100) % 100, "hour": h % 100})
    day0 = ts.dt.normalize().min()
    df["day"] = (ts.dt.normalize() - day0).dt.days.to_numpy().astype(np.int64)
    df["sec_in_day"] = (ts.dt.hour.to_numpy() * 3600).astype(np.int64)
    df = df.sort_values(["day", "sec_in_day"], kind="stable").reset_index(drop=True)
    X = _hash_features(df, AVAZU_CAT_COLUMNS, n_features)
    return Dataset(X=X, y=df["click"].to_numpy(np.int8),
                   day=df["day"].to_numpy(), sec_in_day=df["sec_in_day"].to_numpy(),
                   name="avazu")


def load(source: str, path: str | Path, n_features: int = 2 ** 18,
         sample_frac: float = 1.0, seed: int = 0) -> Dataset:
    if source == "criteo":
        return load_criteo(path, n_features=n_features, sample_frac=sample_frac, seed=seed)
    if source == "avazu":
        return load_avazu(path, n_features=n_features, sample_frac=sample_frac, seed=seed)
    raise ValueError(f"unknown source {source!r}")
