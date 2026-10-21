# Mitate explained

Plain-language notes on every part of Mitate: what it does, why it was chosen, what else was
considered, and a short spoken summary of each. Numbers that come from the
benchmark live in [results.md](results.md); this page explains ideas, not outcomes.

Contents: [the problem](#the-problem) · [measuring](#measuring-offline) ·
[the methods](#the-methods) · [the blend](#the-blend) · [the product](#the-product) ·
[measuring online](#measuring-online) · [privacy](#privacy)

---

## The problem

A small online shop has a few hundred to a few thousand products and, on day one, no shopper
history at all. Recommendation research mostly studies the opposite: millions of interactions
and a team to tune models. Mitate asks one question: **as a store's data grows from nothing,
which recommendation method works best, and can one system follow that curve automatically?**

> **In short.** "Most recommender papers assume big data. Small shops start with a
> catalog and nothing else, then get data slowly. I measured which methods win at each data size
> and built an engine that switches between them, with a learned blend that knows how big the
> store is."

---

## Measuring offline

### Simulated small stores

**What.** Real small-shop logs with a usable licence don't exist publicly, so small stores are cut
out of big real datasets: pick a coherent slice of the catalog (one category subtree, or items
shoppers browse together), keep only events on those items, then admit shoppers in random order
until the store has, say, 10,000 events before the test date. Five stores per size, seven sizes
from 1K to 1M events. Details: [store_simulation.md](store_simulation.md).

**Why.** A random 1% of Amazon isn't a shop. Its items have nothing to do with each other, which
would unfairly hurt every method that learns from items bought together.

**Alternatives.** Random item sampling (incoherent), random user sampling of the full catalog (a
huge catalog with sparse data, unlike any small shop), synthetic data (not real behaviour, so
never used for benchmark numbers).

> **In short.** "I carve small shops out of real logs: a related set of products, all
> the behaviour of the shoppers I admit, and I stop admitting shoppers when the store reaches its
> size. Several random stores per size, so I report spread, not a lucky draw."

### Temporal splits

**What.** One global cut date for everyone: train on everything before it, test on what happens
after. Hyperparameters are chosen on the week before the cut, then the model is retrained on
all data before the cut and tested. Every target in the test week counts ("successive"
targets).

**Why.** Leave-one-out (hold out each user's last item) lets the model train on events that
happen *after* other users' test events, which leaks the future: popular-next-week items look
predictable. *Time to Split* (RecSys 2025) shows the global split with successive targets is the
reliable choice and that validation should mirror it.

**Alternatives.** Leave-one-out (leaky), random splits (worse still), per-user temporal splits
(still mix timelines across users).

> **In short.** "One cut date for the whole store, like real deployment: you train on the
> past and serve the future. Leave-one-out quietly trains on the future."

### Ranking the full catalog

**What.** Every model scores every product for every test query; the top 20 are compared with
what the shopper actually did next.

**Why.** Scoring only 100 sampled negatives inflates numbers and can reverse model rankings
(Krichene & Rendle, 2020). Small catalogs make full ranking cheap anyway.

### Metrics

* **NDCG@k**: did the right items appear near the top? Each hit is worth `1/log2(rank+1)`, so
  rank 1 is worth 1, rank 3 is worth 0.5; divided by the best possible score so it lies in
  [0, 1]. The headline metric.
* **Recall@k**: share of the true next items that appear anywhere in the top k.
* **MRR**: 1 / rank of the first hit. **HitRate**: was there any hit.
* **Repeat vs explore** (Li et al., 2023): many "correct" predictions are items the shopper
  already looked at. Recall is split into repeat targets and new-item targets, so a method that
  only replays history can't hide behind a high average.
* **Beyond accuracy**: catalog coverage (does the store's whole range get shown?), novelty,
  average popularity of what is recommended, Gini of exposure (is it all going to a few items?).

All metrics are averaged per query, then per shopper, then over shoppers, so a single heavy
shopper can't dominate.

> **In short.** "NDCG rewards putting the right item high in the list, discounted by
> position. I also split hits into 'shopper had seen it' and 'new to them', because replaying
> history inflates accuracy without helping discovery."

### Fair comparison

* **Equal tuning budget.** Every tunable model gets the same number of Optuna trials (30),
  searching its own published hyperparameter space. Neural models also choose their training
  length on the validation week (early stopping).
* **Significance.** For each comparison, the per-shopper differences are tested with a paired
  t-test (primary) and a sign-flip permutation test (check), with a bootstrap confidence interval
  and an effect size (d_z = mean difference / SD of differences).
* **Multiple comparisons.** The headline claims ("Mitate ≥ best single method at each size") are
  one family, corrected with Holm. Full leaderboards use Benjamini–Yekutieli, which controls the
  false discovery rate even when the tests are correlated.
* **Why not Wilcoxon?** For mean effectiveness at recommender sample sizes it has inflated false
  positives (Urbano et al., SIGIR 2019); the t-test and permutation test hold their nominal rate.

> **In short.** "Same tuning budget for everyone, so the baselines are as strong as I
> could make them. Paired tests per shopper, Holm-corrected for the headline claims, and I report
> effect sizes and intervals, not just p-values."

### Leakage tests

**What.** Automated tests replace every event after the cut with random noise and require
*bitwise identical* predictions, for every model and for the whole pipeline (tuning, test
predictions and the blend's training data). Further tests check that histories precede query
times, that baskets never straddle the cut, and that no product is recommended before it exists.

**Why.** Leakage is the most common way recommender results go wrong (RecSys 2025 leakage study),
and it is silent. A perturbation test catches any path by which the future reaches a model,
including ones nobody thought of.

> **In short.** "If I scramble everything after the cut date and the predictions change
> by even one bit, a test fails. That catches leaks I didn't think to look for."

### Reproductions (Gate 2)

Before trusting the pipeline on new data, it reproduced two published results: EASE on
MovieLens-20M (Steck, 2019) and GP-TopFreq on Instacart (Li et al., 2023). Exact figures are in
[results.md](results.md).

---

## The methods

### Popularity, recency-weighted popularity

Recommend what most people interact with, optionally weighting recent events more (a half-life
in days). Embarrassingly strong in small stores, and the floor every method must beat.

### GP-TopFreq (repeat purchases)

"Personal top frequency": items this shopper bought most often, then global popularity. The
strongest simple baseline for grocery-like repeat buying (Li et al., 2023). Mitate's
returning-customer placement is built around the same idea.

### Item-item neighbourhoods: ItemKNN and RP3beta

**ItemKNN** scores an item by how similar it is to the items in the shopper's history, where
similarity is a shrunk cosine over "who interacted with both". **RP3beta** is a random walk item →
user → item with a penalty for popular items (exponent β), which pushes recommendations away from
the head of the catalog.

> **In short.** "People who interacted with this also interacted with that, with a
> correction so everything doesn't point to the bestsellers."

### EASE

**What.** A linear model that predicts each item's score from all other items:
`scores = history × B`, where B is an item × item matrix with a zero diagonal (an item may not
predict itself). B has a closed form: `B = I − P · diag(1/diag(P))` with
`P = (XᵀX + λI)⁻¹`. One matrix inverse, no iterations.

**Why.** On many benchmarks it matches or beats deep models (Steck, 2019), is deterministic, and
trains in seconds for catalogs up to ~20K items. Its cost is memory: an n × n dense matrix.

**Alternatives.** SANSA (sparse approximate EASE for huge catalogs; not needed at small-store
size and its sparse Cholesky dependency is GPL), EDLAE/DAN variants.

> **In short.** "EASE is ridge regression from the shopper's history to every item, with
> the rule that an item can't vote for itself. It has a closed-form solution, so training is one
> matrix inverse."

### iALS and BPR (matrix factorisation)

**iALS** learns a vector per user and per item so their dot product predicts interaction,
treating every non-interaction as a weak negative with a confidence weight, solved by alternating
least squares. Properly tuned it remains a top baseline (Rendle et al., 2022). **BPR** learns
the same vectors by ranking a positive above a random negative; it is in the benchmark because
the prompt asked for it, and is known to underperform in common implementations. Anonymous
sessions have no learned user vector, so their vector is computed from their items ("fold-in").

### Transition models: Markov chain, PCTM, SeqRules

**What.** Count which item follows which, within a window and weighted by distance, and score
candidates by the transitions from the shopper's recent items (recent ones weigh more).
**PCTM** and **SeqRules** are the tuned versions studied by Petrov et al. (RecSys 2026), who
found such simple pairwise probes beat tuned transformer recommenders on many datasets.

**Why it matters for Mitate.** These models need little data, train in milliseconds, explain
themselves ("viewed after X by 34 shoppers"), and are exactly the kind of method that might win
at small sizes. Whether and where transformers overtake them is one of the benchmark's
questions.

### Session kNN: VSTAN

Find past sessions similar to the current one (weighted toward recent items and recent
sessions), then recommend what those sessions contained. A strong, classic session baseline
(Ludewig et al., 2021), here compiled with numba.

### SASRec and its loss functions

**What.** A small transformer reads the shopper's sequence of items and produces a vector; the
dot product with each item's vector is that item's score. Self-attention lets it weight which
earlier items matter for the next one.

**The loss is what matters.** The original SASRec trained with binary cross-entropy against one
random negative. Later work showed the architecture was fine and the loss was the problem:

* **Cross-entropy over the full catalog** (SASRec+, Klenitskiy & Vasilev, RecSys 2023): a softmax
  over all items, so the true next item must beat every other item, not just one. Big gains,
  affordable when the catalog is small (as in a small shop).
* **gBCE** (gSASRec, Petrov & Macdonald, RecSys 2023): BCE with many negatives, with the positive
  term reweighted (exponent β, set by a parameter t and the sampling rate) to fix the
  over-confidence that negative sampling causes.
* **Sampled softmax with LiGR layers** (eSASRec, 2025): softmax over the positive and sampled
  negatives, with gated residual layers; reported to match much larger models.

In our small-store data, gBCE with many negatives collapsed to popularity; see ledger D16.

> **In short.** "SASRec is a small transformer over the shopper's sequence. The big
> lesson of the last few years is that its original loss was the bottleneck: full softmax
> cross-entropy makes it competitive again, and in a small catalog full softmax is cheap."

### Content similarity (catalog only)

**What.** Each product's title, category, price band and description become one text, which a
multilingual embedding model turns into a vector. Similar products have nearby vectors. With no
shopper data at all, this powers "similar items" and a reasonable "next item" from what the
shopper has viewed.

**Why multilingual.** Indian catalogs mix English with Indic languages and transliteration.
`multilingual-e5-small` (MIT, CPU-friendly) is the serving default; Qwen3-Embedding-0.6B and a
static embedding model are benchmarked against it.

**Alternatives.** TF-IDF (no semantics), large LLM embeddings (too slow on a CPU), a fine-tuned
LLM recommender (evidence of gains only at very short histories, and costly).

> **In short.** "On day one there is only the catalog, so I embed each product's text and
> recommend neighbours in that space. It's multilingual because Indian catalogs are."

---

## The blend

### Candidates, features, LambdaMART

**What.** Every source (popularity, ItemKNN, RP3beta, EASE, iALS, PCTM, SeqRules, VSTAN, SASRec)
proposes its top 30 items. For each proposed item, Mitate builds features: each source's score,
rank and normalised score; item popularity (all-time, 7 and 30 days), age, price band; the
shopper's history length, session length, time since last event; whether the shopper has seen
the item; and **store-size features**: log of events, items, users and sessions, density, and the
share of anonymous traffic. A gradient-boosted ranker (LightGBM LambdaMART) orders the
candidates.

**LambdaMART in one breath.** Gradient-boosted trees, trained so that swapping two items in the
list costs as much as that swap would change NDCG. It directly optimises the ranking metric
rather than per-item accuracy.

**Why store-size features.** The ranker is trained on validation data pooled from stores of
*every* size. Inside one store those features are constant, but across stores the trees learn
"at this size, trust transition models; at that size, trust SASRec". This is how one model
follows the crossover curve instead of hard-coding thresholds. The *no_volume* ablation
removes these features to test whether they matter.

**Why one pooled ranker.** A store with 1,000 events has too few labels to train a ranker of its
own. Pooling is also why the ranker is trained only on OTTO (CC BY 4.0): it is the one dataset
whose derived models can ship commercially ([licenses.md](licenses.md)).

**Ablations.** *rrf* replaces the learned ranker with reciprocal-rank fusion (sum of
1/(60 + rank) over sources, no training at all). *no_<source>* drops one source exactly: its
candidates and features are removed, as if it had never been built. The engine uses this to serve
stores below a source's data threshold with the ranker trained for that exact ablation.

> **In short.** "Every method nominates candidates; a LambdaMART ranker trained across
> stores of all sizes merges them, using features that tell it how big this store is. So it can
> learn to trust different methods at different sizes, and I can ablate each part exactly."

---

## The product

### Four placements

| Placement | Day one | With data |
|---|---|---|
| Similar items (product page) | text-embedding neighbours | same, popularity breaks ties |
| Goes well with (product page, cart) | merchant-approved category pairings, then text-related products from other categories | co-purchase evidence first |
| Next item (session, home) | content similarity to this visit, then new arrivals | the blend |
| Returning customer | as next item | repurchases first, then the blend |

The product on screen is treated as the shopper's latest view, so every history-based model
conditions on it, even for an anonymous shopper.

### The complement graph

A local LLM (Qwen3-4B through Ollama, offline) reads the store's category list and suggests which
categories go together (pans → lids). Suggestions are restricted to category names that exist;
anything else is discarded. **Nothing is served until the merchant approves it** in the
dashboard. Once orders exist, measured co-purchase outranks the graph. Every tier skips the
product's own category: shoppers do buy a second pan with a pan, but that is a substitute, and
substitutes belong in "Similar items" (ledger D20).

Slots that neither co-purchase nor an approved pairing can fill go to the products whose text is
closest to the product on screen *among other categories* (a coffee filter for filter coffee,
not another coffee). On Amazon Grocery development stores this was the strongest predictor of
cross-category co-purchase at every store size, and without it the rail was empty until a
product had five co-buyers (ledger D19). These products carry no reason, because no counted
fact supports one.

### Business constraints

Applied after scoring, in order: hard filters (blocked products and categories, out of stock,
not deliverable to the shopper's PIN prefix, implausible price relative to the product on screen,
already in the cart), then a per-category cap for diversity, then merchant pins at the top. A
pin never overrides a hard filter: a pinned product that is out of stock is not shown.

### Explanations

Each recommendation carries at most one reason, and every reason is a counted fact about the
product, never the ranker's internals: "34 shoppers who bought X also bought this", "viewed right
after X by 12 shoppers", "Like X: same category, same brand", "You bought this twice before",
"Popular in Mugs this week", "Featured by the store". Counts about other shoppers appear only
when at least five distinct shoppers stand behind them, so a reason can't reveal one person's
behaviour. If no fact holds, no reason is shown.

> **In short.** "Explanations are true statements with counts, chosen from the evidence
> for that product. If nothing true and non-identifying can be said, it says nothing."

### Serving

FastAPI with strict Pydantic schemas; SQLite (one file, no external services); one engine per
store, retrained in the background when events grow by max(200, 20%), the catalog changes, or a
shopper is erased; every version kept in a model registry that records which sources were
active. Rules and stock changes apply instantly without retraining. Latency and memory are
measured, not estimated ([results.md](results.md)).

---

## Measuring online

### Holdout A/B test

A random share of shoppers (say 10%) gets no recommendations; everyone else gets Mitate. The
assignment is a hash of the experiment and the shopper's pseudonymous id, so it is stable and
needs no storage. The report is **revenue per visitor and conversion**, with 95% intervals.
Clicks on recommendations are never reported as lift: a click on a recommended item may be a
purchase the shopper would have made anyway.

### CUPED

**What.** Revenue per visitor is noisy. Each shopper's spend *before* the experiment predicts
their spend during it, so subtracting the predictable part, `y − θ (x − mean(x))` with
`θ = cov(x, y) / var(x)`, removes variance without biasing the difference between groups (the
pre-period can't be affected by the treatment).

**When it helps.** Only when there is a pre-period signal. In our simulations it cuts variance
substantially for repeat grocery shoppers and barely at all for one-off buyers; both are
reported.

> **In short.** "CUPED is regression adjustment with pre-experiment spend. It can't bias
> the effect because the treatment can't change the past, and it shrinks the confidence interval
> when past spend predicts future spend."

### Interleaving

**What.** Instead of splitting shoppers, each shopper sees one list built from two rankers:
team-draft takes turns (coin flip each round), each ranker adding its best item not yet shown,
and remembers which "team" placed each slot. Clicks are credited to the team that placed the
item. The statistic is Δ = (wins_A + ties/2) / impressions − ½ over impressions with a click
(Chapelle et al., 2012), with a standard error that treats shoppers, not impressions, as
independent, and a 355·skew² check that the sample is large enough for the normal approximation.

**Why.** Every shopper compares both rankers, so it needs far less traffic than an A/B test. The
catch: it measures *preference between rankers*, not revenue. Mitate says so in every report.

> **In short.** "Interleaving merges two rankers' lists and sees whose items get clicked.
> It's much more sensitive than A/B testing, but it answers 'which ranker do shoppers prefer',
> not 'how much money did this make'."

### Power calculator

Given visitors per day and the smallest lift worth detecting, it computes how many visitors a
conclusive test needs and how long that takes, using the store's own revenue variance. Small
stores often can't detect a 5% lift within a month; the dashboard says so instead of running a
test that can't conclude.

### Validation

The estimators are checked in simulations where the true effect is known: bias, interval
coverage, false-positive rate under no effect, and power ([results.md](results.md)).

---

## Privacy

India's DPDP Act 2023 and the 2025 Rules: the merchant is the Data Fiduciary; Mitate processes on
its behalf. In code ([privacy.md](privacy.md)):

* Inbound schemas reject unknown fields, so names, emails and addresses can't be sent; there is
  no field for gender, age, religion or caste.
* Shopper and session ids are stored only as HMAC-SHA256 with a per-store secret.
* Without consent, nothing is stored and recommendations use only the product on screen. Minors
  (and, if the merchant chooses, unknown ages) get the same non-personalised path, because the Act
  forbids behavioural monitoring of children.
* Erasure deletes the shopper's events, suppresses future ones and retrains the model at once.
* Logs carry route names and timings only; ids travel in request bodies, never URLs.

> **In short.** "Personal data is minimised at the schema, pseudonymised at the door,
> gated on consent, never inferred, deleted on request including from the model, and absent from
> logs."
