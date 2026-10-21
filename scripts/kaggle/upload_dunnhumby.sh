#!/usr/bin/env bash
# Uploads the dunnhumby source CSVs, unmodified, as the private Kaggle dataset mitate-dunnhumby,
# which scripts/kaggle/kernel.sh attaches to dunnhumby runs (research ledger D26). The data is
# used for research only under dunnhumby's site terms and is never made public.
#
#   scripts/kaggle/upload_dunnhumby.sh
#
# Needs data/raw/dunnhumby/{transaction_data,product}.csv, matching experiments/raw_data.sha256.
set -euo pipefail
cd "$(dirname "$0")/../.."
user="${KAGGLE_USER:-rasinmuhammed}"
kaggle() { uv run --no-project --with kaggle==2.2.4 kaggle "$@"; }

grep 'data/raw/dunnhumby/.*\.csv$' experiments/raw_data.sha256 | shasum -a 256 -c
up=dist/kaggle/dunnhumby
rm -rf "$up"
mkdir -p "$up"
cp data/raw/dunnhumby/transaction_data.csv data/raw/dunnhumby/product.csv "$up/"
cat >"$up/dataset-metadata.json" <<JSON
{"title": "mitate-dunnhumby", "id": "$user/mitate-dunnhumby",
 "licenses": [{"name": "other"}],
 "subtitle": "dunnhumby The Complete Journey source files, private research copy"}
JSON
if [[ $(kaggle datasets status "$user/mitate-dunnhumby" 2>/dev/null) == *ready* ]]; then
  echo "mitate-dunnhumby already exists" >&2
else
  kaggle datasets create -p "$up"
fi
