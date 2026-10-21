#!/usr/bin/env bash
# Builds the files for running the large-store benchmark on Kaggle (ledger D21):
#   dist/kaggle/mitate-bench/      upload as ONE private Kaggle dataset named "mitate-bench"
#   dist/kaggle/mitate_bench.ipynb import as a Kaggle notebook, or push with kernel.sh
# The OTTO Parquet is the exact local file, so Kaggle samples from byte-identical input.
# Upload the first time with `kaggle datasets create -p dist/kaggle/mitate-bench -t`, and after a
# code change with `kaggle datasets version -p dist/kaggle/mitate-bench -t -m <message>` (-t keeps
# the Parquet as Parquet).
set -euo pipefail
cd "$(dirname "$0")/../.."

out=dist/kaggle/mitate-bench
rm -rf "$out"
mkdir -p "$out"

# the same paths scripts/kaggle/code_version.py hashes; the hash ships as CODE_VERSION
zip -qr "$out/mitate-code.zip" \
  mitate bench serve experiments/configs pyproject.toml uv.lock .python-version README.md \
  -x '*/__pycache__/*' '*.pyc' 'mitate/rankers/*'
python3 scripts/kaggle/code_version.py >dist/kaggle/CODE_VERSION
(cd dist/kaggle && zip -q mitate-bench/mitate-code.zip CODE_VERSION)
for f in data/otto/train.parquet data/otto/manifest.json data/otto/cobrowse-t*.parquet; do
  ln "$f" "$out/" 2>/dev/null || cp "$f" "$out/"  # hard link: the Parquet is 1.2 GB
done
cp scripts/kaggle/mitate_bench.ipynb dist/kaggle/
user="${KAGGLE_USER:-rasinmuhammed}"
cat > "$out/dataset-metadata.json" <<JSON
{"title": "mitate-bench", "id": "$user/mitate-bench",
 "licenses": [{"name": "CC-BY-4.0"}],
 "subtitle": "OTTO (CC BY 4.0) as Parquet plus Mitate benchmark code"}
JSON

du -sh "$out"/*
