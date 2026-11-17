#!/usr/bin/env bash
# Reproducible download of the official TACO v1.0 dataset (Zenodo DOI 10.5281/zenodo.3587843).
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data/raw
ZIP_PATH="data/raw/TACO.zip"
URL="https://zenodo.org/records/3587843/files/TACO.zip?download=1"

echo "Downloading TACO v1.0 (~2.7GB) from Zenodo..."
curl -L -C - -o "$ZIP_PATH" "$URL"

echo "sha256:"
shasum -a 256 "$ZIP_PATH"

echo "Extracting..."
unzip -q "$ZIP_PATH" -d data/raw/
rm -rf data/raw/__MACOSX

echo "Done. Dataset at data/raw/TACO/data/ (annotations.json + batch_1..batch_15/)."
