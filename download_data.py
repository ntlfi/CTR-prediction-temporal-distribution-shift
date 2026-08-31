"""Download and unpack the real CTR datasets used in the project.

Usage:
    python download_data.py [--dataset criteo|avazu] [--dest data]

Idempotent: skips the download/unzip if the target file already exists.

criteo: the Criteo Attribution Modeling for Bidding dataset. Its original host
(go.criteo.net / the S3 mirror linked from https://ailab.criteo.com/
criteo-attribution-modeling-bidding-dataset/) returns 404 as of 2026-08. If
DATASET_URL below is still dead, download the file from a mirror (e.g. the
Kaggle copy: "criteo-attribution-modeling" by sharatsachin) and place it at
`<dest>/criteo_attribution_dataset.tsv.gz` yourself -- this script will detect
it and skip straight to the checksum check.

avazu: the Kaggle "avazu-ctr-prediction" data, pulled from the reczoo BARS
mirror `reczoo/Avazu_x4` on the Hugging Face Hub (no Kaggle auth needed). The
zip holds the random 8:1:1 split; avazu_data.py reassembles the chronological
stream from the intact `hour` field.
"""
import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = "http://go.criteo.net/criteo-research-attribution-dataset.zip"
AVAZU_URL = "https://huggingface.co/datasets/reczoo/Avazu_x4/resolve/main/Avazu_x4.zip"
AVAZU_ZIP = "Avazu_x4.zip"
AVAZU_EXPECTED_BYTES = 1306331539
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


def download_url(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    try:
        with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
            shutil.copyfileobj(response, out)
    except Exception as e:
        raise SystemExit(f"Download failed ({e}).")
    return dest


def fetch_avazu(dest_dir: Path):
    zip_path = dest_dir / "avazu" / AVAZU_ZIP
    if zip_path.exists():
        print(f"Found existing {zip_path}, skipping download.")
    else:
        download_url(AVAZU_URL, zip_path)
    size = zip_path.stat().st_size
    if size != AVAZU_EXPECTED_BYTES:
        print(f"WARNING: size mismatch ({size} bytes, expected {AVAZU_EXPECTED_BYTES}).")
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(Path(n).name for n in zf.namelist())
    print(f"Avazu ready at {zip_path} (contains {names}); avazu_data.py reads the zip directly.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["criteo", "avazu"], default="criteo")
    parser.add_argument("--dest", default="data", help="Directory to store the dataset in.")
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    if args.dataset == "avazu":
        fetch_avazu(dest_dir)
        return

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
