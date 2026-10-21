#!/usr/bin/env bash
# Pushes one private benchmark run to Kaggle, on a T4 GPU with internet on (research ledger D21).
#
#   scripts/kaggle/kernel.sh otto 30000        the OTTO 30K bucket
#   scripts/kaggle/kernel.sh hm 1000 3000      H&M, two buckets in one run
#   scripts/kaggle/kernel.sh otto stage2       Mitate on OTTO, once every bucket is finished
#
# Inputs: the private dataset mitate-bench (scripts/kaggle/package.sh); for hm, the H&M
# competition, whose rules must be accepted on kaggle.com first; and mitate-results-<dataset>
# (scripts/kaggle/collect.sh) once an earlier run has left results to resume from.
# Needs a Kaggle API token in ~/.kaggle/access_token.
set -euo pipefail
cd "$(dirname "$0")/../.."

usage() { echo "usage: $0 otto|hm BUCKET... | $0 otto|hm stage2" >&2; exit 2; }
ds=${1:-}
[[ $ds == otto || $ds == hm ]] || usage
shift
(($#)) || usage
stage=1
if [[ $* == stage2 ]]; then
  stage=2
  set --
else
  for b in "$@"; do [[ $b =~ ^[0-9]+$ ]] || usage; done
fi

user="${KAGGLE_USER:-rasinmuhammed}"
# a notebook Kaggle stops starting (an empty log and ERROR at once) can be replaced by a new one
kernel="${KAGGLE_KERNEL:-mitate-bench-$ds}"
kaggle() { uv run --no-project --with kaggle==2.2.4 kaggle "$@"; }

# the run must mount exactly the code packaged here, and that must be the code in this tree
version=$(python3 scripts/kaggle/code_version.py)
if [[ $(cat dist/kaggle/CODE_VERSION 2>/dev/null) != "$version" ]]; then
  echo "The code changed since scripts/kaggle/package.sh: repackage and version mitate-bench." >&2
  exit 1
fi

# the run resumes from the results dataset when one exists, and stops if that holds fewer
# finished results than this tree (a stale version still processing, or a failed publish)
sources="\"$user/mitate-bench\""
resume=0
if [[ $(kaggle datasets status "$user/mitate-results-$ds" 2>/dev/null) == *ready* ]]; then
  sources+=", \"$user/mitate-results-$ds\""
  resume=$(find "experiments/results/$ds" -path '*/test/*.json' 2>/dev/null | wc -l | tr -d ' ')
fi
echo "resume from mitate-results-$ds: $resume finished results expected"
# stage 2 also scores with another dataset's trained ranker when the config names one (D25)
if [[ $stage == 2 ]]; then
  transfer=$(uv run --no-project --with pyyaml==6.0.2 python -c "import yaml, sys
cfg = yaml.safe_load(open('experiments/configs/benchmark.yaml'))
print(cfg['datasets']['$ds'].get('transfer_from', ''))")
  if [[ -n $transfer ]]; then
    [[ -f experiments/results/$transfer/_ltr/full.json ]] ||
      { echo "stage 2 of $ds needs $transfer's ranker: finish $transfer stage 2 first" >&2; exit 1; }
    sources+=", \"$user/mitate-results-$transfer\""
    echo "ranker transfer from $transfer"
  fi
fi

dir=dist/kaggle/kernel-$ds
rm -rf "$dir"
mkdir -p "$dir"
python3 - "$ds" "$*" "$stage" "$version" "$resume" scripts/kaggle/mitate_bench.ipynb "$dir/mitate_bench.ipynb" <<'PY'
import json
import re
import sys

ds, buckets, stage, version, resume, src, dst = sys.argv[1:]
nb = json.load(open(src))
cell = next(c for c in nb["cells"] if "".join(c["source"]).startswith("# ---- settings"))
text = "".join(cell["source"])
values = {
    r'^DATASET = "\w+"': f'DATASET = "{ds}"',
    r"^BUCKETS = \[[^]]*\]": f"BUCKETS = [{', '.join(buckets.split())}]",
    r'^STAGE = "\d"': f'STAGE = "{stage}"',
    r"^RESUME = \d+": f"RESUME = {resume}",
    r"^CODE_VERSION = \S+": f'CODE_VERSION = "{version}"',
}
for pattern, line in values.items():
    text, n = re.subn(pattern, line, text, flags=re.M)
    if n != 1:
        sys.exit(f"the notebook's settings cell has no single line matching {pattern}")
cell["source"] = text.splitlines(keepends=True)
json.dump(nb, open(dst, "w"), indent=1, ensure_ascii=False)
PY

competition=""
[[ $ds == hm ]] && competition='"h-and-m-personalized-fashion-recommendations"'

cat >"$dir/kernel-metadata.json" <<JSON
{"id": "$user/$kernel", "title": "$kernel",
 "code_file": "mitate_bench.ipynb", "language": "python", "kernel_type": "notebook",
 "is_private": true, "enable_gpu": true, "enable_internet": true,
 "machine_shape": "NvidiaTeslaT4",
 "dataset_sources": [$sources], "competition_sources": [$competition], "kernel_sources": []}
JSON
cat "$dir/kernel-metadata.json"
kaggle kernels push -p "$dir" --accelerator NvidiaTeslaT4
echo "Follow with: kaggle kernels status $user/$kernel"
