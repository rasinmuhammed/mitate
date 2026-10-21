# Results

Every number on this page is read from a stored result file by `python -m bench.report`. A
table that has not been computed says *Not run yet*; nothing here is estimated or carried over
by hand. The protocol is in [protocol.md](protocol.md) and every decision that shaped it, with
its timing, is in the [research ledger](research_ledger.md).

**How to read the tables**

* Scores are test-window NDCG@10: the mean over stores, ± the half-width of a 95% t-interval
  across stores.
* "Mitate vs …" rows are paired tests on the same test units (shoppers for ranking, anchor
  products for placements), with a 95% interval on the difference and Holm-adjusted p-values
  across store sizes.
* "The baseline each store picks" is the baseline with the best score on that store's own
  validation data, which is the choice a merchant could make without seeing the test. "Oracle"
  is the best baseline on the test data itself, an upper bound no one can pick in advance.
* The means table weights every store equally; the paired tests weight every test unit
  equally. Where a store has very few test units (some 1K placement stores have one to four
  anchors), the two can disagree, and the paired test is the one to trust.
* Ranking stores use seeds 0–4, with spare seeds for a store that cannot be evaluated (ledger
  D24). Placement stores use seeds 1–5: seed 0 was the placement development store, where
  design decisions were made after looking at its numbers, and is never reported.

## 1. Ranking: which method wins at each store size (OTTO)

Store sizes are the number of shopper events before the test cut; catalog size grows with them
([store_simulation.md](store_simulation.md)). Above 10K events, tuning follows ledger D18. Every
ranking result on this page was computed on Kaggle (Tesla T4, ledger D21 and D23); each run's
hardware record is stored with its results.

{{otto_stores}}

{{otto_ndcg_10_headline}}

{{otto_figure}}

All models, NDCG@10:

{{otto_ndcg_10_models}}

All models, Recall@20:

{{otto_recall_20_models}}

### What drives the result

What each piece of Mitate contributes: Mitate minus a version of it with that piece removed,
paired over the same shoppers (positive means the piece helps; *n.s.* means not significant
after Benjamini-Yekutieli correction across all ablations of the dataset):

{{otto_ndcg_10_ablations}}

### Cost

Peak memory and wall time per model, measured one at a time on an otherwise idle Apple M1 (8 GB):

{{otto_efficiency}}

### Same stores, another machine

OTTO stores of 1K–10K events were first run on an Apple M1 and then rerun from scratch on Kaggle
(ledger D23): the same sampled stores (checked bit for bit, `bench.fingerprint`), the same
configuration and seeds. Only models finished on both machines are listed.

{{otto_replication}}

**Reading this table.** Models with nothing to tune and no randomness (VSTAN, SeqRules, the
Markov chain, popularity) agree to the last digit, which confirms the stores are identical. Tuned
models differ, even deterministic ones such as EASE and ItemKNN: floating-point results differ
slightly between machines, and that is enough to send the 30-trial search down a different path
after its first identical trials, to different hyperparameters. The differences therefore measure
tuning variance at a fixed budget. On a single small store they can reach several hundredths of
NDCG@10, which is why the comparisons above pair shoppers across all five stores of a size
instead of reading any one store.

## 2. Ranking: fashion, next basket (H&M)

Next-basket recommendation for returning customers: one basket per customer-day, test on the last
week of the competition's training file (ledger D22). The H&M data never leaves Kaggle; only the
scores below do. Small fashion stores see few buying shoppers in a test week, so read the paired
tests, and the shopper counts here, before the means:

{{hm_stores}}

{{hm_ndcg_10_headline}}

{{hm_figure}}

The figure plots a set of methods fixed in advance, one per family. Recent popularity was added
to that set after the H&M results were seen, because it is the strongest baseline here and
leaving it out would flatter Mitate.

All models, NDCG@10:

{{hm_ndcg_10_models}}

### What drives the result

What each piece of Mitate contributes: Mitate minus a version of it with that piece removed,
paired over the same shoppers (positive means the piece helps; *n.s.* means not significant
after Benjamini-Yekutieli correction across all ablations of the dataset):

{{hm_ndcg_10_ablations}}

### Returning customers: the rail as shipped

A returning customer's rail starts with repurchases: up to half of it is the products they have
ordered most often, the most recent first, and the ranker fills the rest. The benchmark calls the
same function the product does (`mitate.engine.repurchase_first`), on a 10-product rail: five
repurchases, then the ranker. "Ranker alone" is the headline above. This comparison was added
after the Dunnhumby baselines showed the repeat-purchase baseline leading at every store size
(ledger D27), so it sits beside the headline rather than replacing it.

{{hm_ndcg_10_returning}}

## 3. Ranking: grocery, next basket (Dunnhumby)

Grocery is repeat buying: across the whole dataset, 54% of the products in test-window baskets
had been bought by the same household before. Next-basket recommendation for 2,500 households
over two years, testing on the last 28 days (ledger D26).
The data has only 2,500 households, so stores stop at 30K events. Like H&M, the data never leaves
Kaggle; only the scores below do.

{{dunnhumby_stores}}

{{dunnhumby_ndcg_10_headline}}

{{dunnhumby_figure}}

All models, NDCG@10:

{{dunnhumby_ndcg_10_models}}

**Finding.** Grocery is where Mitate's ranker does not win. GP-TopFreq, a household's own most
frequent purchases, is the best single method at every size, and the ranker alone does not beat
it: no difference is significant after the Holm correction, but at 10K and 30K the ranker is about
0.01 behind on nearly every store (1 of 5 and 0 of 5 stores at or above it), and on Recall@20 it is
significantly behind at 30K. None of the ranker's nine sources recommends repurchases. Stage 2 on
this dataset ran on the Apple M1 rather than Kaggle (ledger D28).

### What drives the result

{{dunnhumby_ndcg_10_ablations}}

**Finding.** Unlike OTTO and H&M, the store-size features hurt here: the ranker trained without
them is better, significantly at 1K, 10K and 30K. With 20 stores to learn from, the ranker fits
differences between those particular stores. The shipped OTTO ranker scores the same as the one
trained on Dunnhumby itself (no difference is significant).

### Returning customers: the rail as shipped

A returning customer's rail starts with repurchases: up to half of it is the products they have
ordered most often, the most recent first, and the ranker fills the rest. The benchmark calls the
same function the product does (`mitate.engine.repurchase_first`), on a 10-product rail: five
repurchases, then the ranker. "Ranker alone" is the headline above. This comparison was added
after the Dunnhumby baselines showed the repeat-purchase baseline leading at every store size
(ledger D27), so it sits beside the headline rather than replacing it.

{{dunnhumby_ndcg_10_returning}}

**Finding.** With repurchases first, as the product serves returning customers, Mitate matches
GP-TopFreq at every size, within ±0.005 NDCG@10, with either ranker. With the shipped OTTO ranker
it is ahead at 30K before correction (+0.005) and level after it.

## 4. Product-page placements (Amazon Grocery)

The "Similar items" and "Goes well with" rails are scored against Amazon's own product-page
links (2018 metadata, McAuley lab): `also_view` for similar items, and `also_buy` for
complements. Raw `also_buy` is mostly same-kind products, which is the wrong goal for a
complements rail (ledger D19), so complements are also scored on the `also_buy` pairs that
cross leaf categories. Each store's anchors are split in half: tuned baselines and the
"store picks" choice use the first half, every reported number the second.

### Similar items (`also_view`)

{{placements_also_view}}

**Finding.** Mitate's similar-items rail is text similarity with e5-small, and it ties that
baseline at every size. No method beats it, and store behaviour adds nothing measurable here.

### Goes well with: pairs across categories (`also_buy`, different leaf category)

{{placements_also_buy_cross}}

**Finding.** At 1K and 10K too few product pairs have five co-buyers to count, so Mitate serves
its third tier, text similarity across categories, which is the best or tied-best method. At
100K, co-purchase evidence adds a small gain over that baseline that is significant after the
Holm correction. That gain was measured after the D20 change, so treat it as suggestive (below).

### Goes well with: every `also_buy` pair

{{placements_also_buy}}

**Finding.** Against every `also_buy` pair, Mitate is far behind plain text similarity: most
of those pairs are same-category substitutes, which this rail leaves out on purpose. This is the
cost of that design choice, reported rather than tuned away.

### Before ledger D20

After the results above were first computed on seeds 1–5, the demo store showed the complements
rail filling with substitutes (a second pan beside a pan). Tier 1 of the complements scorer
was changed to skip the anchor's own category, and the benchmark was rerun on the same seeds.
That change was made after those seeds' numbers had been seen, so both runs are published. The
tables above are after the change; these are before it. Similar items did not change.

Pairs across categories, before D20:

{{placements_d19_also_buy_cross}}

Every `also_buy` pair, before D20:

{{placements_d19_also_buy}}

## 5. Online measurement, validated in simulation

These tables check the online module's statistics against simulated stores where the true lift
is known. They do not measure Mitate's effect on any real store.

Revenue-per-visitor A/B estimates, plain and with pre-period adjustment (CUPED):

{{online_ab}}

False positives when the true lift is zero. A test is reported as reliable only when each group
has at least 355 × skewness² visitors (Kohavi et al., KDD 2014):

{{online_null}}

Does the power calculator ask for enough visitors?

{{online_power}}

Interleaving: how often a preference test picks the better ranker:

{{online_interleaving}}

## Not run

* H&M with the shipped OTTO ranker and the returning-customer rail: waiting for the Kaggle GPU
  quota to reset (ledger D28). The H&M rows above are final.

* Any real store. Offline accuracy is not revenue; the online module is how a store finds out.
