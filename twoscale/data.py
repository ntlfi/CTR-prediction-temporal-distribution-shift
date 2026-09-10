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
_CRITEO_CHUNK = 2_000_000


def load_criteo(tsv_path: str | Path, n_features: int = 2 ** 18,
                sample_frac: float = 1.0, seed: int = 0) -> Dataset:
    if sample_frac < 1.0:
        # sub-sampled path (smoke tests, other repo scripts) -- unchanged so the
        # exact rows for a given (sample_frac, seed) are reproducible.
        df = pd.read_csv(tsv_path, sep="\t", usecols=CRITEO_RAW_COLUMNS)
        df = df.sample(frac=sample_frac, random_state=seed)
        df["day"] = (df["timestamp"] // SECONDS_PER_DAY).astype(np.int64)
        df["sec_in_day"] = (df["timestamp"] % SECONDS_PER_DAY).astype(np.int64)
        df = df.sort_values(["day", "sec_in_day"], kind="stable").reset_index(drop=True)
        X = _hash_features(df, CRITEO_CAT_COLUMNS, n_features)
        return Dataset(X=X, y=df["click"].to_numpy(np.int8),
                       day=df["day"].to_numpy(), sec_in_day=df["sec_in_day"].to_numpy(),
                       name="criteo")

    # full-data path: chunk-hash so the full ~16.5M-row string frame is never
    # materialised (FeatureHasher is per-row -> vstack of per-chunk hashes is
    # bit-identical to hashing the whole frame; the stable sort on
    # (day, sec_in_day) is the same permutation).
    from scipy.sparse import vstack as sparse_vstack
    hasher = FeatureHasher(n_features=n_features, input_type="string")
    # explicit int64 for the categorical columns -> per-chunk `.astype(str)` is
    # identical to whole-file inference (the Criteo attribution cat columns are
    # clean integers, no NaN), so no `"5"` vs `"5.0"` drift across chunks.
    _dt = {"timestamp": np.int64, "click": np.int8, **{c: np.int64 for c in CRITEO_CAT_COLUMNS}}
    x_blocks, clicks, ts = [], [], []
    for chunk in pd.read_csv(tsv_path, sep="\t", usecols=CRITEO_RAW_COLUMNS,
                             dtype=_dt, chunksize=_CRITEO_CHUNK):
        tok = chunk[CRITEO_CAT_COLUMNS].astype(str)
        for col in CRITEO_CAT_COLUMNS:
            tok[col] = col + "=" + tok[col]
        x_blocks.append(hasher.transform(tok.values.tolist()))
        clicks.append(chunk["click"].to_numpy(np.int8))
        ts.append(chunk["timestamp"].to_numpy(np.int64))
        del chunk, tok
    X = sparse_vstack(x_blocks, format="csr")
    del x_blocks
    y = np.concatenate(clicks)
    t = np.concatenate(ts)
    day = (t // SECONDS_PER_DAY).astype(np.int64)
    sec = (t % SECONDS_PER_DAY).astype(np.int64)
    order = np.lexsort((sec, day))                 # stable sort by (day, then sec_in_day)
    return Dataset(X=X[order], y=y[order].astype(np.int8),
                   day=day[order], sec_in_day=sec[order], name="criteo")


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
    """Chunk-hashed: each read chunk's category columns are hashed to a
    sparse block immediately and the strings dropped, so the full ~40M-row
    string frame is never materialised (peak memory ~one chunk + the
    accumulating CSR). ``FeatureHasher`` is per-row and deterministic, so
    ``vstack`` of the per-chunk hashes is bit-identical to hashing the
    concatenated frame; the per-chunk ``rng.random`` sub-sampling call
    sequence is unchanged, so a given ``(sample_frac, seed)`` gives the
    same rows as before."""
    from scipy.sparse import vstack as sparse_vstack

    rng = np.random.default_rng(seed)
    read_kw = dict(usecols=_AVAZU_USECOLS,
                   dtype={"click": np.int8, "hour": np.int64,
                          **{c: "string" for c in AVAZU_CAT_COLUMNS}})
    hasher = FeatureHasher(n_features=n_features, input_type="string")
    x_blocks, clicks, hours = [], [], []
    for fh in _avazu_handles(src):
        for chunk in pd.read_csv(fh, chunksize=_AVAZU_CHUNK, **read_kw):
            if sample_frac < 1.0:
                chunk = chunk.loc[rng.random(len(chunk)) < sample_frac]
            if not len(chunk):
                continue
            tok = chunk[AVAZU_CAT_COLUMNS].astype(str)
            for col in AVAZU_CAT_COLUMNS:
                tok[col] = col + "=" + tok[col]
            x_blocks.append(hasher.transform(tok.values.tolist()))
            clicks.append(chunk["click"].to_numpy(np.int8))
            hours.append(chunk["hour"].to_numpy(np.int64))
            del chunk, tok
    if not x_blocks:
        raise FileNotFoundError(f"no rows read from {src}")

    X = sparse_vstack(x_blocks, format="csr")
    del x_blocks
    y = np.concatenate(clicks)
    h = np.concatenate(hours)
    ts = pd.to_datetime({"year": 2000 + h // 1_000_000, "month": (h // 10_000) % 100,
                         "day": (h // 100) % 100, "hour": h % 100})
    day = (ts.dt.normalize() - ts.dt.normalize().min()).dt.days.to_numpy().astype(np.int64)
    sec = (ts.dt.hour.to_numpy() * 3600).astype(np.int64)
    order = np.lexsort((sec, day))          # stable sort by (day, then sec_in_day)
    return Dataset(X=X[order], y=y[order].astype(np.int8),
                   day=day[order], sec_in_day=sec[order], name="avazu")


def load(source: str, path: str | Path, n_features: int = 2 ** 18,
         sample_frac: float = 1.0, seed: int = 0) -> Dataset:
    if source == "criteo":
        return load_criteo(path, n_features=n_features, sample_frac=sample_frac, seed=seed)
    if source == "avazu":
        return load_avazu(path, n_features=n_features, sample_frac=sample_frac, seed=seed)
    raise ValueError(f"unknown source {source!r}")
