# Mitate

**Chosen with a knowing eye.** A recommender for small online stores: sensible on day one from
the catalog alone, better as shopping data arrives, and honest about when it isn't helping.

見立て (*mitate*) is choosing something for someone with a discerning eye, the way a kimono
seller picks what will suit a customer. Here it means looking at a small shop's catalog and its
few shoppers and still choosing well.

## What it does

| Placement | Day one (catalog only) | Once shoppers arrive |
|---|---|---|
| Similar items | multilingual text embeddings of title, category, price band, description | same |
| Goes well with | LLM-suggested category pairings the merchant approved, then text-related products from other categories | measured co-purchase first |
| You may also like | similarity to this visit, then new arrivals | learned blend of 9 methods |
| Returning customers | as above | repurchases first, then the blend |

* **Shifts methods as data grows.** One LambdaMART ranker, trained across simulated stores of
  every size with store-size features, merges popularity, item-item (ItemKNN, RP3beta, EASE),
  iALS, pairwise-transition probes (PCTM, SeqRules), session kNN (VSTAN) and SASRec.
* **Business rules**: stock, delivery region (PIN prefix), pins, blocks, price sanity, category
  diversity. Rules apply instantly, without retraining.
* **Factual explanations**: "34 shoppers who bought X also bought this". Counts only, never
  invented, and never from fewer than five shoppers.
* **Measures itself**: holdout A/B tests on revenue per visitor with CUPED, team-draft
  interleaving, and a power calculator that says when a store is too small to test.
* **Privacy by construction** for India's DPDP Act: no personal fields accepted, HMAC pseudonyms,
  consent gating, a no-profile path for minors, erasure that retrains the model.
* **Runs on a laptop**: FastAPI + SQLite, CPU only, no external services.

## Results

The benchmark asks: *at each store size, which method wins, and does Mitate's blend match or beat
the best single method at every size?* Protocol **Mitate-GTS v1** ([protocol.md](docs/protocol.md)):
simulated stores from real data (5 per size, 1K–1M interactions), global temporal split,
full-catalog ranking, equal tuning budgets, paired tests with Holm correction.

> **Status (2026-10-21):** every baseline is finished on OTTO and H&M at all seven store sizes,
> and Mitate's own results on H&M are in [results.md](docs/results.md). Mitate on OTTO is running;
> the headline comparison appears here once both datasets are in, including every size where
> Mitate does not win.

What is already measured:

* **Pipeline validity (Gate 2).** Two published results reproduced: EASE on MovieLens-20M
  (Recall@20 0.3905 vs 0.391 published; Recall@50 0.5215 vs 0.521; NDCG@100 0.4202 vs 0.420) and
  GP-TopFreq on the NBR reality-check Instacart split (exact match to their code).
* **Leakage (Gate 3).** Predictions are bitwise identical when everything after the cut date is
  replaced with noise, for every model and the whole pipeline (enforced by the test suite).

## Quickstart (synthetic demo store)

```bash
uv sync --group demo
uv run --group demo python -m serve.demo            # builds a synthetic shop in data/demo
MITATE_DATA_DIR=data/demo MITATE_DEMO_SITE=data/demo/site uv run python -m serve run
# storefront:  http://127.0.0.1:8000/demo/
# dashboard:   http://127.0.0.1:8000/dashboard/   (private key in data/demo/keys.json)
```

The demo store is generated with [Misata](https://github.com/rasinmuhammed/misata) and is
labelled synthetic everywhere; it is never used for benchmark numbers. Press `X` on any page
to open the x-ray: per rail, the stage, sources, rules that held items back and the counted
evidence behind each reason. `?skin=thali|night|plain` swaps the host store's look to show the
widget inheriting it, and `?xray=1` opens with the x-ray on.

## Your own store

```bash
uv run python -m serve create-merchant "My Shop"     # prints a public and a private key once
MITATE_DATA_DIR=data/serve uv run python -m serve run
curl -X POST localhost:8000/v1/catalog/import/shopify -H "X-Mitate-Key: sk_..." \
     --data-binary @products_export.csv
```

```html
<script src="https://your-host/widget/mitate.js" defer data-key="pk_..." data-api="https://your-host"
        data-url-template="/products/{id}"></script>
<div data-mitate="similar" data-item="SKU1"></div>
<script>Mitate.setConsent("granted") /* from your consent banner */</script>
```

API: catalog import (Mitate CSV, Shopify and WooCommerce exports) and lookups, events,
recommend (single or batched), explain, status and daily totals, rules, complement-graph review,
experiments and reports, power calculator, shopper export and erasure. Interactive docs at `/docs` when the server runs.

## Reproduce

```bash
scripts/reproduce.sh          # data → reproductions → benchmark → online validation → efficiency → report
```

Needs a Kaggle account for OTTO (CC BY 4.0). The full benchmark is compute-heavy; see
[results.md](docs/results.md) for measured run times.

OTTO stores of 30K events and up, and all of H&M, run on Kaggle GPUs (research ledger D21, D22):

```bash
scripts/kaggle/package.sh            # code + OTTO Parquet as the private dataset mitate-bench
scripts/kaggle/kernel.sh otto 30000  # push one private run on a T4
scripts/kaggle/collect.sh otto       # fetch, check hardware, merge, save for the next run's resume
scripts/kaggle/kernel.sh otto stage2 # Mitate itself, once every bucket's baselines are finished
```

## Limitations

* Benchmark stores are slices of large marketplaces, not real small shops.
* The ranking benchmark so far uses OTTO (anonymous sessions, no item text); Amazon Grocery
  (2018 metadata) evaluates the similar and complements placements. Fashion (H&M) and grocery
  repeat purchase (Dunnhumby) need dataset terms accepted.
* Offline accuracy is not revenue. The online module measures revenue; its simulated validation
  is in results.md, and interleaving's sensitivity on real small-store traffic is unverified.
* The dense item-item models assume catalogs up to ~25K items.

## Documentation

[Explainer](docs/explainer.md) · [Protocol](docs/protocol.md) ·
[Store simulation](docs/store_simulation.md) · [Research ledger](docs/research_ledger.md) ·
[Licenses](docs/licenses.md) · [Privacy](docs/privacy.md) · [Model card](docs/model_card.md)

## License

Code: Apache-2.0. Benchmark data keeps its own licenses ([licenses.md](docs/licenses.md)); the
only pretrained artefact that ships is trained on OTTO (CC BY 4.0; attribution: OTTO GmbH & Co KG).
