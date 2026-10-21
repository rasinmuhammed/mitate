# Store simulation

The benchmark's question is how recommendation methods behave as a store grows. Real small-store
logs with a clear license do not exist publicly, so small stores are carved out of large real
datasets in a way that keeps what makes a shop a shop. The code is
[`bench/simulator.py`](../bench/simulator.py); every store is reproducible from
`(dataset, bucket, seed)`.

## Design goals

1. **Coherent assortment.** A small shop sells related products (one niche, one brand family), not
   a random 0.1% of a marketplace. Random item slices would destroy co-purchase structure and make
   every collaborative method look worse than it is in a real niche store.
2. **Intact shoppers.** Each simulated customer keeps their full behaviour inside the shop, so
   repeat purchases, session structure and history lengths stay realistic.
3. **Natural arrival of new customers and new items.** The test period receives new and returning
   shoppers and newly launched items at the dataset's own rates.
4. **No future information in the definition.** A store's assortment is chosen using only data
   before `t_test`.
5. **Variance, not one lucky draw.** Five independently seeded stores per size bucket.

## Procedure

For a size bucket `(n_interactions, n_items)` and a seed:

1. **Choose the assortment.**
   * *Taxonomy datasets* (H&M, Amazon, Dunnhumby): pick a random leaf category among items with at
     least 20 pre-test events; climb the category tree, adding whole sibling subtrees, until the
     subtree holds at least `n_items` active items; sample `n_items` uniformly from it. If even the
     top-level department is smaller, the store is that whole department; a store never mixes
     departments. If it then cannot reach its interaction target, the target assortment grows by
     1.5x from a new random leaf, at most three times, and otherwise the store is infeasible.
   * *Co-browse datasets* (OTTO, which has no item metadata): build an item graph from
     consecutive distinct events within sessions before `t_test` (undirected, top-20 neighbours per
     item by count). Start from a random active seed item and grow best-first: repeatedly add the
     frontier item with the largest total edge weight into the current set. The result is a
     tightly co-browsed assortment, the closest observable proxy for "one shop's range".
2. **Restrict the log** to the assortment. The shop never sees what its customers do elsewhere.
3. **Thin the traffic.** Customers (known users; anonymous sessions count as one-visit customers)
   are put in a seeded random order and admitted until the store has `n_interactions` events before
   `t_test`. All of an admitted customer's events are kept, before and after `t_test`.
4. **Widen if too quiet.** If the assortment's entire pre-test traffic is below the target, the
   assortment grows by 1.5× and steps 1–3 repeat, up to three times; otherwise the store is marked
   infeasible for that bucket and reported as missing (never silently replaced).
5. **Densify ids** and record statistics: items, pre-test and test events, known users, sessions,
   anonymous share, and items whose first event is in the test period (cold).

## Size grid

| Bucket (pre-test interactions) | 1K | 3K | 10K | 30K | 100K | 300K | 1M |
|---|---|---|---|---|---|---|---|
| Target catalog size | 200 | 400 | 800 | 1,500 | 3,000 | 6,000 | 12,000 |

The catalog grid is a modelling choice (assortment grows sub-linearly with traffic), not a measured
law; it is fixed in `experiments/configs/benchmark.yaml`.

## Known limitations

* **OTTO is anonymous.** It has sessions but no persistent customer ids, so every OTTO shopper is a
  one-visit customer and the returning-customer placement cannot be evaluated on it.
* **Co-browse assortments are defined by behaviour,** not by a merchant's merchandising decision.
  They are coherent in what shoppers consider together, which is what matters to recommenders, but
  they can mix product types a real shop would not.
* **Thinning preserves the dataset's time span.** A 1K-interaction store still spans the dataset's
  full history window (4 weeks for OTTO), so it is a slow shop, not a young one. A young store
  (short history) is a different axis; it is not simulated.
* **Marketplace behaviour leaks into store behaviour.** Shoppers in the source data saw a large
  marketplace's own recommendations; their in-shop sequences inherit that exposure bias.

## Released artefacts

`experiments/results/<dataset>/stores.json` lists every store's seed and statistics, and every
seed passed over with the reason (protocol §1, ledger D24). For OTTO
(CC BY 4.0) it also lists the item ids of each assortment. For other datasets only seeds and code
are released, per their licenses ([licenses.md](licenses.md)).
