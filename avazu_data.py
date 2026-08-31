"""Load the Avazu CTR dataset as a chronological, leakage-safe feature matrix
-- the second real temporal benchmark called for by AMG-TP_Academic_LaTeX.pdf
section 5.3 (the "insufficient real evidence" decision-table outcome: synthetic
results are strong but Criteo's natural 31-day drift is too shallow for any
method to separate).

Source: the Kaggle "avazu-ctr-prediction" competition data, obtained via the
reczoo BARS mirror `reczoo/Avazu_x4` (https://huggingface.co/datasets/reczoo/
Avazu_x4). Avazu_x4 is ~40.4M rows of 10-day mobile-ad click-through logs,
randomly partitioned 8:1:1 into train/valid/test. The random split shuffles
row order but leaves the `hour` field (format YYMMDDHH) intact, so
concatenating the three parts and sorting by `hour` recovers the full
chronological stream.

Time is indexed in sub-day **blocks** rather than calendar days: 10 calendar
days leaves far too few blocks to evaluate temporal adaptation (~3 test days
after warmup). The default 2-hour block gives 120 blocks -- the same horizon
as the synthetic suite -- with the diurnal cycle (period ~12 blocks) inside
the window family's reach, making Avazu a genuine *recurring-drift* test on
real data, the regime where AMG-TP's learned persistence beta_t shows its
strongest reproducible synthetic win (S3). `block_hours=1` (240 blocks) is
also available for a finer-grained view.

Only pre-bid context is used as features: C1, banner_pos, the site_/app_/
device_ fields and the anonymized C14-C21 categories. `id` (row id, useless)
and `hour` (the time index itself) are dropped -- the temporal methods must
discover any periodicity from loss dynamics, not be handed the clock (the same
discipline that made the M5c explicit-phase-feature experiment a fair test).
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# every raw Avazu column except `id` and the `hour` timestamp
AVAZU_CAT_COLUMNS = [
    "C1", "banner_pos", "site_id", "site_domain", "site_category",
    "app_id", "app_domain", "app_category",
    "device_id", "device_ip", "device_model", "device_type", "device_conn_type",
    "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
]
_USECOLS = ["click", "hour"] + AVAZU_CAT_COLUMNS
_SPLIT_FILES = ["train.csv", "valid.csv", "test.csv"]


_STR_COLS = [c for c in AVAZU_CAT_COLUMNS]
_READ_KW = dict(usecols=_USECOLS,
                dtype={"click": np.int8, "hour": np.int64, **{c: "string" for c in _STR_COLS}})
_CHUNK = 2_000_000


def _iter_split_handles(src: Path):
    """Yield (name, file-handle) for each of train/valid/test.csv, from either
    the Avazu_x4.zip archive or a directory of extracted files."""
    src = Path(src)
    if src.is_dir():
        for f in _SPLIT_FILES:
            if (src / f).exists():
                with open(src / f, "rb") as fh:
                    yield f, fh
        return
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as zf:
            names = {Path(n).name: n for n in zf.namelist()}
            for f in _SPLIT_FILES:
                if f in names:
                    with zf.open(names[f]) as fh:
                        yield f, fh
        return
    raise FileNotFoundError(f"{src} is neither an Avazu_x4.zip nor a directory of split CSVs")


def _read_sampled(src: Path, sample_frac: float, seed: int) -> pd.DataFrame:
    """Chunked read of all three splits, uniformly subsampling each chunk so
    peak memory scales with sample_frac rather than the full 6 GB of CSV."""
    rng = np.random.default_rng(seed)
    parts = []
    for name, fh in _iter_split_handles(src):
        for chunk in pd.read_csv(fh, chunksize=_CHUNK, **_READ_KW):
            if sample_frac < 1.0:
                keep = rng.random(len(chunk)) < sample_frac
                chunk = chunk.loc[keep]
            parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    for c in _STR_COLS:
        df[c] = df[c].astype("category")
    return df


def _time_block(hour: np.ndarray, block_hours: int) -> np.ndarray:
    """YYMMDDHH integer -> running block index (block = `block_hours` hours)
    from the first block in the data."""
    h = np.asarray(hour, dtype=np.int64)
    ts = pd.to_datetime({
        "year": 2000 + h // 1_000_000,
        "month": (h // 10_000) % 100,
        "day": (h // 100) % 100,
        "hour": h % 100,
    })
    hrs = (ts - ts.min()).dt.total_seconds() // 3600
    return (hrs.to_numpy().astype(np.int64) // block_hours).astype(int)


def load_raw(src: str | Path, sample_frac: float = 1.0, seed: int = 0,
             block_hours: int = 2) -> pd.DataFrame:
    """Load Avazu, reassemble the chronological stream, add a `day` column
    (the time-block index; a block is `block_hours` hours).

    `sample_frac` uniformly subsamples rows (a fast preliminary run, and --
    unlike Criteo, whose full dataset makes seeds nearly degenerate -- a
    genuine source of per-seed variation). Chronological order and block
    boundaries are unaffected.
    """
    df = _read_sampled(src, sample_frac, seed)
    if not len(df):
        raise FileNotFoundError(f"no rows read from {src}")
    df["day"] = _time_block(df["hour"].to_numpy(), block_hours)
    df = df.sort_values(["day", "hour"], kind="stable").reset_index(drop=True)
    df = df.drop(columns=["hour"])
    return df


def numeric_context(df: pd.DataFrame) -> np.ndarray:
    """Dense per-example context for the gates (m2/m5b/amgtp), analogous to
    data.raw_numeric_features but built for Avazu's hex-hash category values,
    which have no numeric form: each column is factorised (sorted) to an
    integer code and normalised to [0, 1]. As with Criteo's anonymised ids
    these codes carry no ordinal meaning -- they are just extra numeric input
    the gate's L2 penalty can down-weight -- but the low-cardinality fields
    (banner_pos, device_type/conn_type, C1, C15/C16 ad size) do expose
    genuine structure.
    """
    out = np.zeros((len(df), len(AVAZU_CAT_COLUMNS)), dtype=np.float64)
    for j, col in enumerate(AVAZU_CAT_COLUMNS):
        codes, _ = pd.factorize(df[col], sort=True)
        cmax = codes.max()
        out[:, j] = codes / cmax if cmax > 0 else codes
    return out


def load_dataset(src: str | Path, n_features: int = 2**18, sample_frac: float = 1.0,
                 seed: int = 0, block_hours: int = 2):
    """Convenience wrapper mirroring data.load_dataset: (X, y, day)."""
    from data import hash_features
    df = load_raw(src, sample_frac=sample_frac, seed=seed, block_hours=block_hours)
    X = hash_features(df, columns=AVAZU_CAT_COLUMNS, n_features=n_features)
    return X, df["click"].to_numpy(), df["day"].to_numpy()
