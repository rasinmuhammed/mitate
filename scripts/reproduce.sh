#!/usr/bin/env bash
# Reproduce every number in docs/results.md from raw data.
#
#   scripts/reproduce.sh            # all stages
#   scripts/reproduce.sh bench      # one stage: data | repro | bench | online | efficiency | report
#
# Needs: uv, unzip, ~15 GB free disk, and a Kaggle account for OTTO (CC BY 4.0) with
# ~/.kaggle/kaggle.json in place. Every stage is resumable and skips finished work.
# Full benchmark compute is large (see docs/results.md, "Compute"); expect days on a laptop.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${CONFIG:-experiments/configs/benchmark.yaml}
NBR=https://raw.githubusercontent.com/liming-7/A-Next-Basket-Recommendation-Reality-Check/main/dataset
ML20M=https://files.grouplens.org/datasets/movielens/ml-20m.zip
AMZ=https://mcauleylab.ucsd.edu/public_datasets/data

fetch() {  # url dest
  [ -f "$2" ] || { mkdir -p "$(dirname "$2")"; curl -fL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"; }
}

stage_data() {
  uv sync --locked --group dev
  if [ ! -f data/raw/otto/otto-recsys-train.jsonl.zip ]; then
    uvx kaggle datasets download otto/recsys-dataset -f otto-recsys-train.jsonl.zip -p data/raw/otto
    uvx kaggle datasets download otto/recsys-dataset -f otto-recsys-test.jsonl.zip -p data/raw/otto
  fi
  fetch "$ML20M" data/raw/ml-20m/ml-20m.zip
  fetch "$NBR/instacart_history.csv" data/raw/nbr_reality_check/instacart_history.csv
  fetch "$NBR/instacart_future.csv" data/raw/nbr_reality_check/instacart_future.csv
  if [ ! -f data/amazon/events.parquet ]; then
    fetch "$AMZ/amazon_2023/raw/review_categories/Grocery_and_Gourmet_Food.jsonl.gz" \
      data/raw/amazon/Grocery_and_Gourmet_Food.jsonl.gz
    fetch "$AMZ/amazon_2023/raw/meta_categories/meta_Grocery_and_Gourmet_Food.jsonl.gz" \
      data/raw/amazon/meta_Grocery_and_Gourmet_Food.jsonl.gz
    fetch "$AMZ/amazon_v2/metaFiles2/meta_Grocery_and_Gourmet_Food.json.gz" \
      data/raw/amazon/v2018_meta_Grocery_and_Gourmet_Food.json.gz
  fi
  # verify whatever raw files are present (converted sources may have been removed to save disk)
  grep -E "  ($(ls data/raw/*/* 2>/dev/null | paste -sd'|' -))$" experiments/raw_data.sha256 \
    | shasum -a 256 -c -
  [ -f data/otto/train.parquet ] || uv run python -m bench.data.otto data/raw/otto data/otto
  [ -f data/raw/ml-20m/ml-20m/ratings.csv ] || unzip -q -o data/raw/ml-20m/ml-20m.zip -d data/raw/ml-20m
  [ -f data/amazon/events.parquet ] || uv run python -m bench.data.amazon data/raw/amazon data/amazon
}

stage_repro() {  # Gate 2: published baselines reproduced before anything else is trusted
  uv run python -m bench.repro.gp_topfreq_instacart data/raw/nbr_reality_check --tiebreak first
  uv run python -m bench.repro.ease_ml20m data/raw/ml-20m/ml-20m/ratings.csv
}

stage_bench() {
  uv run python -m bench.run "$CONFIG"
}

stage_online() {
  uv run python -m bench.online_validation
}

stage_efficiency() {  # fresh single-threaded processes; run on an otherwise idle machine
  uv run python -m bench.efficiency "$CONFIG" --dataset otto --seed 0
}

stage_report() {
  uv run python -m bench.analyze experiments/results/otto --metric ndcg@10
  uv run python -m bench.analyze experiments/results/otto --metric recall@20
  uv run python -m bench.report "$CONFIG"
}

stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(data repro bench online efficiency report)
for s in "${stages[@]}"; do
  echo "== $s"
  "stage_$s"
done
