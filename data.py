"""Load the Criteo attribution dataset and turn it into a chronological,
leakage-safe feature matrix for CTR prediction.

Only pre-bid-time context is used as features: the campaign id and the nine
anonymized contextual categories (cat1-cat9). Everything else in the raw file
(cost, cpo, click position/count, conversion fields, attribution, time since
last click) either happens after the impression or leaks the outcome, so it
is dropped.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher

SECONDS_PER_DAY = 86_400

RAW_COLUMNS = ["timestamp", "campaign", "click", "cat1", "cat2", "cat3", "cat4",
               "cat5", "cat6", "cat7", "cat8", "cat9"]
CAT_COLUMNS = ["campaign", "cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8", "cat9"]


def load_raw(tsv_path: str | Path, sample_frac: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """Load the raw tsv.gz, keep only pre-bid columns, add a `day` index.

    `sample_frac` uniformly subsamples rows (useful for a fast preliminary
    run); the chronological order and day boundaries are unaffected.
    """
    df = pd.read_csv(tsv_path, sep="\t", usecols=RAW_COLUMNS)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed).sort_values("timestamp")
    df["day"] = (df["timestamp"] // SECONDS_PER_DAY).astype(int)
    df = df.reset_index(drop=True)
    return df


def hash_features(df: pd.DataFrame, columns=CAT_COLUMNS, n_features: int = 2**18) -> "scipy.sparse.csr_matrix":
    """Hash categorical columns into a fixed-width sparse matrix.

    Each row becomes a list of "column=value" string tokens, matching the
    hashing-trick approach used in the dataset's original paper. `columns`
    defaults to the Criteo context columns but is reused as-is by
    synthetic_data.py for the drift-injection experiments.
    """
    hasher = FeatureHasher(n_features=n_features, input_type="string")
    tokens = df[columns].astype(str)
    for col in columns:
        tokens[col] = col + "=" + tokens[col]
    return hasher.transform(tokens.values.tolist())


def hash_indices(df: pd.DataFrame, columns=CAT_COLUMNS, vocab_size: int = 2**16) -> np.ndarray:
    """Per-column integer embedding indices in [0, vocab_size), for the
    neural (embedding + MLP) model used by sftl.py. Uses the same
    "column=value" token convention as hash_features, hashed with md5 for
    determinism regardless of PYTHONHASHSEED. Only unique values per
    column are hashed, not every row.
    """
    out = np.zeros((len(df), len(columns)), dtype=np.int64)
    for j, col in enumerate(columns):
        tokens = (col + "=" + df[col].astype(str)).to_numpy()
        uniques, inverse = np.unique(tokens, return_inverse=True)
        buckets = np.array(
            [int(hashlib.md5(u.encode()).hexdigest(), 16) % vocab_size for u in uniques], dtype=np.int64)
        out[:, j] = buckets[inverse]
    return out


def load_dataset(tsv_path: str | Path, n_features: int = 2**18, sample_frac: float = 1.0, seed: int = 0):
    """Convenience wrapper: returns (X, y, day) ready for the baselines."""
    df = load_raw(tsv_path, sample_frac=sample_frac, seed=seed)
    X = hash_features(df, n_features=n_features)
    y = df["click"].to_numpy()
    day = df["day"].to_numpy()
    return X, y, day
