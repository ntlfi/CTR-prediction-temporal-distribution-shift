"""Download and unpack the Criteo Attribution Modeling for Bidding dataset.

Usage:
    python download_data.py [--dest data]

Idempotent: skips the download/unzip if the target file already exists.

Note: the dataset's original host (go.criteo.net / the S3 mirror linked from
https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/) returns
404 as of 2026-08. If DATASET_URL below is still dead, download the file from
a mirror (e.g. the Kaggle copy: "criteo-attribution-modeling" by sharatsachin)
and place it at `<dest>/criteo_attribution_dataset.tsv.gz` yourself -- this
script will detect it and skip straight to the checksum check.
"""
import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = "http://go.criteo.net/criteo-research-attribution-dataset.zip"
ZIP_NAME = "criteo-research-attribution-dataset.zip"
TSV_NAME = "criteo_attribution_dataset.tsv.gz"
# Recorded from a known-good copy of the file; used to sanity-check whatever
# ends up at TSV_NAME, however it got there.
EXPECTED_SHA256 = "94ac7a465564349bc7ba008602211d5990a3c53cc133abc0aadef61ea2391a98"
EXPECTED_BYTES = 653015824


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / ZIP_NAME
    print(f"Downloading {DATASET_URL} -> {zip_path}")
    try:
        with urllib.request.urlopen(DATASET_URL) as response, open(zip_path, "wb") as out:
            shutil.copyfileobj(response, out)
    except Exception as e:
        raise SystemExit(
            f"Download failed ({e}). The original host may be down -- see the "
            f"note at the top of this script for a manual fallback."
        )
    return zip_path


def extract(zip_path: Path, dest_dir: Path) -> Path:
    tsv_path = dest_dir / TSV_NAME
    print(f"Extracting {TSV_NAME} -> {tsv_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(TSV_NAME, dest_dir)
    return tsv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="data", help="Directory to store the dataset in.")
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    tsv_path = dest_dir / TSV_NAME

    if not tsv_path.exists():
        zip_path = download(dest_dir)
        tsv_path = extract(zip_path, dest_dir)
    else:
        print(f"Found existing {tsv_path}, skipping download/extraction.")

    size = tsv_path.stat().st_size
    if size != EXPECTED_BYTES:
        print(f"WARNING: size mismatch ({size} bytes, expected {EXPECTED_BYTES}).")
    digest = sha256_of(tsv_path)
    if digest != EXPECTED_SHA256:
        print(f"WARNING: sha256 mismatch ({digest}). Proceeding anyway, but verify the source.")
    else:
        print("Checksum OK.")
    print(f"Dataset ready at {tsv_path}")


if __name__ == "__main__":
    main()
