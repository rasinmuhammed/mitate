#!/usr/bin/env bash
# Publishes the merged local results as the private Kaggle dataset mitate-results-<dataset>,
# which scripts/kaggle/kernel.sh attaches so the next run resumes from everything finished:
#
#   scripts/kaggle/publish_results.sh otto
set -euo pipefail
cd "$(dirname "$0")/../.."

ds=${1:-}
[[ $ds =~ ^(otto|hm|dunnhumby)$ ]] || { echo "usage: $0 otto|hm|dunnhumby" >&2; exit 2; }
[[ -f experiments/results/$ds/stores.json ]] || { echo "no results for $ds" >&2; exit 1; }
user="${KAGGLE_USER:-rasinmuhammed}"
kaggle() { uv run --no-project --with kaggle==2.2.4 kaggle "$@"; }

up=dist/kaggle/results-$ds
rm -rf "$up"
mkdir -p "$up/stage/experiments/results"
# built from the merged local results, so a failed or partial run can never shrink what the next
# run resumes from; OTTO stores ride along (CC BY 4.0), other datasets' stores stay on Kaggle
cp -R "experiments/results/$ds" "$up/stage/experiments/results/"
if [[ $ds == otto ]]; then
  mkdir -p "$up/stage/data/stores"
  cp -R data/stores/otto "$up/stage/data/stores/"
fi
(cd "$up/stage" && zip -qr ../mitate-results.zip .)
rm -rf "$up/stage"
cat >"$up/dataset-metadata.json" <<JSON
{"title": "mitate-results-$ds", "id": "$user/mitate-results-$ds",
 "licenses": [{"name": "other"}],
 "subtitle": "Finished Mitate benchmark results, for resuming the next run"}
JSON
if [[ $(kaggle datasets status "$user/mitate-results-$ds" 2>/dev/null) == *ready* ]]; then
  kaggle datasets version -p "$up" -m "results after $(date -u +%Y-%m-%dT%H:%MZ)"
else
  kaggle datasets create -p "$up"
fi
