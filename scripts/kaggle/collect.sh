#!/usr/bin/env bash
# Brings a finished Kaggle run home (research ledger D21):
#
#   scripts/kaggle/collect.sh otto
#
# 1. Downloads the output of the notebook mitate-bench-<dataset> (its results zip).
# 2. Checks that every hardware record in it names the same GPU, so a bucket never mixes
#    accelerators.
# 3. Merges the results into experiments/results/<dataset>, never replacing a local result, and
#    the store manifest entry by entry. OTTO stores go to data/stores/otto; H&M output holds
#    results only.
# 4. Publishes the merged local results for the next run to resume from (publish_results.sh).
# Exits non-zero, after merging, if the benchmark crashed rather than finishing or reaching the
# notebook's time budget.
set -euo pipefail
cd "$(dirname "$0")/../.."

ds=${1:-}
[[ $ds == otto || $ds == hm ]] || { echo "usage: $0 otto|hm" >&2; exit 2; }
user="${KAGGLE_USER:-rasinmuhammed}"
# a notebook Kaggle stops starting (an empty log and ERROR at once) can be replaced by a new one
kernel="${KAGGLE_KERNEL:-mitate-bench-$ds}"
kaggle() { uv run --no-project --with kaggle==2.2.4 kaggle "$@"; }

out=dist/kaggle/output-$ds
rm -rf "$out"
mkdir -p "$out"
# the output holds every result twice, as files and as one zip; the zip is one download, not
# thousands
kaggle kernels output "$user/$kernel" -p "$out" -q --file-pattern '^mitate-results\.zip$'
res=$out/mitate-results
unzip -q "$out/mitate-results.zip" -d "$res"
[[ -d $res/experiments/results/$ds ]] || { echo "no results for $ds in the output" >&2; exit 1; }
status=$(cat "$res/RUN_STATUS.json" 2>/dev/null || echo '{"exit_code": null}')
echo "run status: $status"

python3 - "$res/experiments/results/$ds/_hardware" <<'PY'
import json
import sys
from pathlib import Path

records = sorted(Path(sys.argv[1]).glob("*.json"))
gpus = {json.loads(p.read_text())["gpu"] for p in records}
for p in records:
    print(p.name, json.loads(p.read_text())["gpu"])
if len(gpus) != 1:
    sys.exit(f"runs used different GPUs: {sorted(gpus)}")
PY

rsync -a --ignore-existing --exclude stores.json "$res/experiments/results/$ds/" "experiments/results/$ds/"
# the store manifest grows with every run: merge it entry by entry, the newer run's entry winning
python3 - "$res/experiments/results/$ds/stores.json" "experiments/results/$ds/stores.json" <<'PY'
import json
import sys
from pathlib import Path

new, local = Path(sys.argv[1]), Path(sys.argv[2])
rows = {r["store"]: r for r in json.loads(local.read_text())} if local.exists() else {}
rows |= {r["store"]: r for r in json.loads(new.read_text())} if new.exists() else {}
local.write_text(json.dumps(sorted(rows.values(), key=lambda r: (r["bucket"], r["seed"])), indent=1))
PY
mkdir -p experiments/results/kaggle_logs
for f in "$res"/experiments/results/run_kaggle*.log; do
  [[ -e $f ]] && cp "$f" "experiments/results/kaggle_logs/$(date -u +%Y%m%dT%H%MZ)-${f##*/}"
done
if [[ $ds == otto && -d $res/data/stores/otto ]]; then
  rsync -a --ignore-existing "$res/data/stores/otto/" data/stores/otto/
fi

scripts/kaggle/publish_results.sh "$ds"
find "experiments/results/$ds" -path '*/test/*.json' | wc -l | xargs echo "finished (store, model) results now local:"
# finished (store, model) results stay valid after a crash; the crash itself must not go unseen
if [[ $status != *'"exit_code": 0'* && $status != *'"stopped_at_deadline": true'* ]]; then
  echo "The benchmark did not finish cleanly: read $res/experiments/results/run_kaggle_$ds.log" >&2
  exit 1
fi
