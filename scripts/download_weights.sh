#!/usr/bin/env bash
# Download the pretrained checkpoint to weights/checkpoint.pth and verify its sha256.
# usage: bash scripts/download_weights.sh
set -euo pipefail
cd "$(dirname "$0")/.."
RELEASE_URL="${RELEASE_URL:-https://github.com/LGECTO-AIL-VIL/SingleQuery-BHOI-Code/releases/download/v1.0.0}"   # base URL of the SingleQuery-BHOI release files
SHA256="770ffecd2264c8e1e50c214791c4aeb36d0a9b6d9852d336ec8ac627c06d33d5"                              # expected sha256 of checkpoint.pth
mkdir -p weights
wget -c -O weights/checkpoint.pth "$RELEASE_URL/checkpoint.pth"
echo "$SHA256  weights/checkpoint.pth" | sha256sum -c -
