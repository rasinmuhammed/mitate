# Model card: Mitate

Format after Mitchell et al., "Model Cards for Model Reporting" (FAT* 2019). Performance figures
are not repeated here; they live in [results.md](results.md), generated from stored results.

## Model details

* **What it is.** A recommendation engine for small online stores. Per store, it fits a set of
  candidate sources (popularity, item-item neighbourhoods, EASE, iALS, pairwise-transition
  models, session kNN, SASRec with full-softmax loss) and merges their candidates with one
  gradient-boosted ranker (LightGBM LambdaMART). Before a store has behavioural data it serves
  text-embedding similarity, a merchant-approved category complement graph (then text-related
  products from other categories), and new arrivals.
* **What ships.** Code under the PolyForm Noncommercial License 1.0.0, plus one pretrained
  artefact: the LambdaMART rankers in `mitate/rankers/`, dumped tree ensembles scored in pure
  numpy, with a `MANIFEST.json` recording their training stores, code version and checksums.
  Everything else is fitted on the merchant's own data inside their deployment.
* **Training data of the shipped ranker.** OTTO (CC BY 4.0) only: validation-week labels pooled
  from simulated stores of every size. Rankers fitted on non-commercial datasets (H&M,
  Dunnhumby, Amazon, Instacart, MovieLens) are benchmark artefacts and are never distributed
  ([licenses.md](licenses.md)).
* **Version.** 0.1.0, protocol Mitate-GTS v1.
* **Contact.** support@misata.studio

## Intended use

* Product-page "similar" and "goes well with" lists, a session or home-page "you may also like"
  list, and a returning-customer list, for stores with roughly 10² to 10⁴ products and 0 to
  ~10⁶ interactions.
* Ranking within a merchant's own catalog, under the merchant's own rules (stock, delivery
  regions, pins, blocks, price sanity, diversity).

**Out of scope.** Marketplaces with millions of items (the dense item-item models assume ≤ 25K
items); content or media recommendation; any decision about people (credit, hiring, pricing per
person); dynamic pricing; anything that needs a shopper's identity, demographics or sensitive
traits.

## Factors

* **Store size** is the factor the benchmark is built around: results are reported per size
  bucket, from 1K to 1M interactions before the test week.
* **Shopper history**: new vs returning shoppers, history length buckets, session length.
* **Items**: cold (first seen in the test week) vs warm items, popular head vs long tail.
* **Not used as factors or features, by design**: gender, age, religion, caste or any other
  sensitive attribute, inferred or supplied. There is no schema field through which they could
  arrive.

## Metrics

NDCG@10 (headline), Recall, MRR and HitRate at 5/10/20, repeat vs explore recall, catalog
coverage, novelty, average recommended popularity, exposure Gini. Per-shopper paired tests with
Holm correction for the headline claims; effect sizes and confidence intervals for every
comparison. Definitions: [protocol.md](protocol.md).

Online, the deployed metrics are incremental revenue per visitor and conversion against a
randomised holdout, with 95% intervals; clicks on recommendations are never reported as lift.

## Evaluation data

Simulated small stores carved from real logs ([store_simulation.md](store_simulation.md)):
OTTO (session-heavy general e-commerce) for all reported numbers so far. Planned: H&M (fashion),
Dunnhumby (grocery and repeat buying) and Amazon Reviews 2023 Grocery (text-rich; complement
ground truth from Amazon 2018 `also_buy`). Each dataset section of results.md states whether it
has been run.

## Quantitative analyses

See [results.md](results.md): per-size tables for every model, the crossover figure,
Mitate-vs-best-single tests (including the bucket sizes where Mitate does **not** win),
ablations, and measured latency and memory on a stated laptop.

## Ethical considerations

* **Privacy.** Designed for India's DPDP Act 2023 and Rules 2025: data minimisation at the
  schema, HMAC pseudonyms, consent gating, a non-personalised path for minors, erasure that also
  retrains the model, retention limits, no identifiers in logs ([privacy.md](privacy.md)).
* **Explanations** are counted facts, shown only when at least five distinct shoppers stand
  behind a count, so a reason cannot reveal one person's purchase.
* **Popularity feedback loops.** Recommenders that learn from their own exposure concentrate
  attention on a few products. Exposure Gini and coverage are reported per size; merchants can
  cap items per category and pin products. Benchmark data was itself collected under large
  marketplaces' own recommenders, so offline results inherit their exposure bias.
* **Children.** In stores selling to children, merchants should treat unknown ages as minors,
  which disables all personalisation.

## Caveats and recommendations

* Offline gains are not revenue. Run the built-in holdout test; the power calculator tells a
  store in advance whether it has enough traffic to conclude anything.
* Simulated stores are slices of large marketplaces, not real small shops; they keep the source
  data's time span, so they model *slow* shops rather than *young* ones.
* OTTO has no item text and no persistent customer ids, so content similarity and the
  returning-customer placement are not evaluated on it.
* gSASRec's published defaults collapse on small catalogs in our implementation (ledger D16); its
  reported numbers use tuned negatives.
* Neural training used Apple-silicon MPS kernels, which are not bitwise deterministic; neural
  numbers reproduce up to floating-point noise. Everything else is deterministic given the seed.
