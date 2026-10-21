# Evaluation protocol

This is the protocol of record. Every number in [results.md](results.md) names it as
**Mitate-GTS v1**. Changes require a new version number; results under different versions are
never mixed in one table. Evidence for each choice is in [research_ledger.md](research_ledger.md).

## 1. Stores

Benchmark units are simulated small stores carved from real datasets
([store_simulation.md](store_simulation.md)). A store is identified by `(dataset, bucket, seed)`
and is fully deterministic. Size buckets are defined by the number of interactions before
`t_test` (what a deployed model has seen): 1K, 3K, 10K, 30K, 100K, 300K, 1M. Five stores per
bucket: seeds 0–4, where a seed whose store cannot be sampled at that size, or has no
validation or no test query, is replaced by the next spare seed (5–14). Passed-over seeds and
their reasons are listed in `stores.json` (ledger D24).

## 2. Splits

Global temporal split with dataset-level cut points `t_val < t_test < t_end`:

| Phase | Fit on | Evaluate on | Targets |
|---|---|---|---|
| Tuning | events < `t_val` | `[t_val, t_test)` | **Last** eligible target per shopper |
| Test | events < `t_test` (train + validation, refit) | `[t_test, t_end)` | **Successive**: every eligible target |

| Dataset | `t_val` | `t_test` | `t_end` | Task |
|---|---|---|---|---|
| OTTO | 2022-08-14 22:00 UTC | 2022-08-21 22:00 UTC | 2022-08-28 22:00 UTC (end of the official train file) | next item |
| H&M | 2020-09-09 00:00 UTC | 2020-09-16 00:00 UTC | 2020-09-23 00:00 UTC (the day after the last day of `transactions_train.csv`) | next basket |

OTTO's official test week is not used: its sessions are truncated (ledger D15). H&M's test
window is the last week of the competition's training file, the same seven-day horizon the
competition predicts. H&M dates carry no time of day, so a basket is one customer-day and all
its events share the day's start (ledger D22).

Other datasets are added to this table as they are ingested.

* A query's history is every event of that shopper strictly before the target's timestamp,
  including window events before it (live histories). Events at the same second as the target are
  not history. Model parameters never see window events.
* **Next item** (sessions): sessions that *start* inside the window; consecutive duplicates
  collapsed; targets are positions 1..n−1 where a new item appears (the first item has no
  in-session context and is never a target).
* **Next basket** (returning customers): ORDER baskets of known shoppers inside the window;
  baskets that began before the window are excluded so no order straddles the cutoff.
* Sampling caps: up to 2,000 validation shoppers and 5,000 test shoppers per store (seeded); the
  realised counts are stored with each result.

## 3. Ranking

* Full-catalog ranking. No sampled negatives.
* Items not yet in the catalog at query time (first event later than the query) are masked.
* **Seen-item policy** is per dataset and stated: OTTO keeps previously seen items eligible
  (re-viewing and re-carting are legitimate), H&M keeps previously bought items eligible
  (basics are re-bought), and repeat vs explore are reported separately.
* Top-20 lists are stored for every query, model and store.

## 4. Metrics

Per target query, then averaged within the shopper ("unit"; anonymous sessions are their own
unit), then across units:

* Accuracy: NDCG, Recall (÷|T|), MRR, HitRate at k ∈ {5, 10, 20}. NDCG's ideal DCG uses
  min(k, |T|) relevant items.
* Repeat vs explore (Li et al. TOIS 2023): Recall_rep, Recall_expl, Hit_rep, Hit_expl at 10 and 20
  (each over the shoppers that have such targets), and RepR@10.
* Beyond accuracy, with item statistics from training data only: catalog coverage@10/20, novelty@10
  (mean self-information over distinct buyers), ARP@10, APLT@10 (long tail = outside the items
  covering 80% of training interactions), Gini@10, exposure entropy@10; ILD where items have
  categories or embeddings.
* Segments: new vs returning shoppers, history-length buckets (0, 1–2, 3–5, 6–10, 11–20, 21–50, 51+),
  session length, store-size bucket, cold vs warm target items.

## 5. Tuning

* Optuna TPE (multivariate, seeded with the store seed), run sequentially. Objective: validation
  NDCG@10.
* **Every tunable model gets the same number of trials** (30). Parameter-free models are evaluated
  once. Neural models tune their epoch count inside each trial with early stopping on validation
  (patience 5), and each trial has a wall-clock cap; caps are reported.
* The final model is refit on all data before `t_test` with the best parameters (neural models:
  the best epoch count).
* **Large stores (ledger D18).** At 30K interactions and above, every model gets 20 trials on the
  bucket's first store and the other stores reuse that configuration, re-selecting only the
  epoch count on their own validation window. From 100K, gSASRec, eSASRec and GRU4Rec are not run.
  These rules were fixed before any store above 10K was evaluated.
* Search spaces are the `space()` methods in `mitate/models/`, published with the results; best
  parameters per store are stored in `experiments/results/<dataset>/<store>/tuning/`.

## 6. The Mitate blend

* Candidate sources are the tuned single models listed in the config (`blend_sources`).
* Ranker: LambdaMART, trained on validation-window labels pooled across all stores of the dataset
  (every size bucket), with sources fitted before `t_val`. Its hyperparameters get the same
  30-trial budget, tuned on a store-level 80/20 split of the pooled groups.
* At test time sources are refit before `t_test`; the ranker is not retrained.
* Variants: full; reciprocal-rank fusion (no learning); without store-volume features; without
  each source (exact: candidates nominated only by the removed source are dropped and its
  features blanked, which equals rebuilding the blend without it). Business-constraint and
  complement-graph ablations apply only on datasets with categories.

## 7. Statistics

* Unit of analysis: the shopper (unit-averaged metric).
* Mitate vs each baseline, per store: paired t-test (primary), sign-flip permutation test
  (10,000 permutations, check), paired bootstrap 95% CI of the mean difference (10,000 resamples),
  effect size d_z = mean(d)/sd(d), and relative change.
* Confirmatory family ("Mitate ≥ the best single method in each bucket"): Holm correction.
  Exploratory leaderboards: Benjamini–Yekutieli.
* Per bucket: mean ± std and a t-interval across the five stores.
* Wilcoxon and sign tests are not used (see ledger D1).

## 8. Leakage guards (test suite)

`tests/test_leakage.py` and `tests/test_runner.py`:

1. `Log.until` refuses events at or after its horizon.
2. Every history event precedes its query time; every target lies in the window.
3. Model predictions are bitwise identical when every event at or after the cutoff is replaced by
   random noise, for each model and **for the whole runner** (tuning trials, test recommendations
   and the blend's LTR groups).
4. Baskets never straddle the cutoff.
5. No item is recommended before it exists.
6. Scoring never mutates a fitted model.

## 9. Efficiency

Measured separately on a stated machine (Apple M1, 8 GB, CPU only at serving) in fresh processes:
fit time, peak RSS, serialised model size, p50/p95 single-request latency, and throughput. See
[results.md](results.md).
