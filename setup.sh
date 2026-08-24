#!/usr/bin/env bash
# Create a virtualenv and install dependencies.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "Setup complete. Activate with: source .venv/bin/activate"
