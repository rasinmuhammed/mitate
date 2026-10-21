# 02 — Sequential & session-based recommendation (next item for the current shopper)

Scope: next-item recommendation for a current, possibly anonymous, shopper in a SMALL store
(10^2–10^4 items, ~10^3–10^6 interactions, CPU serving, 8 GB M1 for training).
Review date: 2026-10-19. Every paper below was fetched (abstract page, HTML, PDF, or Crossref record)
unless it is listed under "Could not verify". All numbers are copied from the fetched source. Anything
I computed or inferred is labelled **[ours]**.

---

## TL;DR

1. On benchmark-sized data (≤ ~1M interactions), the strongest *verified* next-item models are
   **recency-weighted pairwise transition probes**: PCTM and Sequential Rules.
   On Amazon Beauty/Sports/Toys and ML-1M they beat a faithful eSASRec / sampled-softmax SASRec
   reproduction. They lose only on ML-20M (20M interactions). Mitate's whole operating range is below
   the point where the transformer won.
2. For anonymous **session** data in e-commerce, **session-kNN (SKNN / V-SKNN / STAN / VSTAN)** won
   or tied the neural models on 3 of 4 e-commerce datasets in the largest controlled comparison. The
   2024 (ECIR) and 2025 (RecSys) GNN re-evaluations by Shehzad & Jannach still find simple models
   competitive.
3. If a neural model is used, it should be **SASRec trained with full softmax cross-entropy** (CE). With
   ≤ tens of thousands of items, full softmax is cheap, so the negative-sampling fixes (gBCE, sampled
   softmax, TRON) mostly matter for large catalogues. Use it only behind a validation gate that it beats
   the pairwise envelope.
4. **Do not use the official GRU4Rec code**: its license is non-commercial. Most third-party GRU4Rec
   ports are badly broken. HSTU has no verified small-data advantage.
5. **Evaluation must use a global temporal split.** Leave-one-out (LOO) leaks future data. In the
   Helsinki study it inflated NDCG@10 by 16.5–90% and reshuffled model rankings.

---

## 1. Pairwise transition probes (PCTM, SeqRules) — **Choice: primary sequential engine**

### Evidence
- **Petrov, Chandar, Bennett, Bouchard, Lalmas, "Do Sequential Recommendation Benchmarks Really
  Require Higher-Order Sequence Modelling?"**
  - Venue and ID: RecSys 2026, arXiv **2608.19833**. The ID you gave is correct; submitted 20 Aug 2026.
  - URLs fetched: https://arxiv.org/abs/2608.19833 and https://arxiv.org/html/2608.19833
  - Code: https://github.com/spotify-research/sequential-capacity-probes (Apache-2.0, last push
    2026-08-18). I read `pctm.py`, `seqrules.py`, `docs/PROTOCOL.md` and `docs/HYPERPARAMETERS.md`.
  - Protocol: the eSASRec splits and evaluation code, full-catalogue ranking, seen items filtered, no
    sampled negatives. Hyperparameters were chosen on an inner holdout. The reproduction matches the
    reported eSASRec results within 0.0033 NDCG.
  - NDCG@10 (Table 1):

    | Data   | MC    | FMC+  | SASRec+ (SS) | eSASRec | SeqRules | PCTM  |
    |--------|-------|-------|--------------|---------|----------|-------|
    | Beauty | .0492 | .0531 | .0537        | .0524   | .0605    | **.0635** |
    | Sports | .0251 | .0271 | .0315        | .0324   | **.0371**| .0368 |
    | Toys   | .0548 | .0588 | .0575        | .0533   | .0730    | **.0738** |
    | ML-1M  | .1240 | .1206 | .1662        | .1739   | .1505    | **.1815** |
    | ML-20M | .1050 | .1034 | .1806        | **.1969** | .1115  | .1431 |

  - Abstract claims: the probe envelope beats eSASRec by 15–38% on the three Amazon sets and by 4.4%
    on ML-1M, and trails it by **27.3% on ML-20M**. At least one probe also beats the sampled-softmax
    SASRec reproduction by 9–28% on the other four datasets.
  - The authors' conclusion: these benchmarks are "poorly suited to measuring gains from higher-order
    sequence modelling". They propose recency-weighted pairwise probes as a validity check.
  - Your description was mostly right, with one correction: the probes beat tuned SASRec/eSASRec on
    4 of 5 datasets, **not** on ML-20M.
  - PCTM stands for "Probabilistic Collaborative Transition Model".
- **Han et al. (Michigan State and Airbnb), "An Embarrassingly Simple Graph Heuristic Reveals
  Shortcut-Solvable Benchmarks for Sequential Recommendation"**, arXiv 2605.07125 (May 2026,
  preprint). https://arxiv.org/abs/2605.07125
  - The heuristic starts from the last 1–2 items, expands a few-hop item-transition graph, and ranks
    by item-feature similarity.
  - It is competitive on 10 of 14 datasets and gains +38.10% and +44.18% NDCG@10 over the best
    baseline on Amazon Sports and CDs.
  - The authors name three "shortcuts": low-branching local transitions, feature-smooth transitions,
    and limited dependence on long histories. I read the abstract only.
- **Klenitskiy, Volodkevich, Pembek, Vasilev, "Does It Look Sequential?"**, RecSys 2024, arXiv
  2408.12008. https://arxiv.org/html/2408.12008
  - Shuffling test sequences changes NDCG@10 very little on RetailRocket (SASRec −2%, GRU4Rec −5%)
    and Diginetica (−7% / −16%). It changes a lot on OTTO (−28% / −37%) and MegaMarket (−45% / −56%).
  - So some e-commerce logs carry little order information, and **this varies by store**.

### Alternatives
MC (first-order Markov chain), FMC/FMC+, SASRec+, eSASRec, GRU4Rec, and session-kNN (section 2).

### Why it wins for small stores
- No embeddings and no SGD; it is closed-form sparse counting. It fits in seconds on CPU and is
  deterministic.
- It has about 5 hyperparameters, which suits the small validation sets a small store produces.
- Memory scales with the number of non-zero pairs, which is at most interactions × W. **[ours]** For
  1M interactions and W=10 that is ≤ 10M non-zeros, roughly 120 MB as float32 CSR.
- Each recommendation is interpretable as "people who viewed X then viewed Y".

### Caveats
- No published crossover curve exists. The only scale signal is "wins at ≤ ~1M interactions (ML-1M,
  Amazon), loses at 20M (ML-20M)".
- These are academic leave-one-out-style splits from the eSASRec release, not global temporal splits.
- In the paper, **repeat items are excluded** from recommendations. Stores with repeat purchases
  (consumables) need a repeat/reminder channel; make it a flag.
- Items absent from the transition graph get only the popularity background score, so this method
  has no cold-start answer.

---

## 2. Session-kNN family (SKNN, V-SKNN, STAN, VSTAN) — **Choice: anonymous-session engine / ensemble member**

### Evidence
- **Ludewig, Mauro, Latifi, Jannach, "Empirical analysis of session-based recommendation
  algorithms"**, UMUAI 31:149–181 (2021), arXiv 1910.12781. https://arxiv.org/pdf/1910.12781
  (pages 5–14 read).
  - Compares 12 algorithms, all tuned with 100 random-search iterations on MRR@20, using five
    contiguous time splits.
  - **E-commerce, Table 6:**
    - RETAIL (Retailrocket; 210k actions, 32k items): STAN is best on MAP@20 (0.0285), R@20 (0.4748),
      HR@20 (0.5938) and MRR@20 (0.3638). GRU4Rec reaches 0.5669 HR@20 and 0.3237 MRR@20.
    - DIGI (264k actions): SKNN is best on MAP@20 (0.0255). VSTAN reaches HR@20 0.4803 and MRR@20
      0.1837, against GRU4Rec 0.4639 / 0.1644.
    - ZALANDO (4.5M actions, 189k items): VSTAN is best on MAP@20 (0.0168) and HR@20 (0.5362).
      GRU4Rec has the best MRR@20 (0.3069).
    - RSC15 (5.4M actions, 29k items): this is the only dataset where neural methods (NARM, SR-GNN)
      edge ahead. The authors say the differences are "in most cases around or less than 1%".
  - **Cost:** training takes 0.13–0.20 min for kNN methods against 39–1577 min for neural methods,
    and prediction takes about 18–56 ms per request (Table 8).
  - Conclusion: nearest-neighbour heuristics are "in most cases" preferable. A user study found the
    heuristic recommendations were well accepted.
- **Ludewig & Jannach, "Evaluation of Session-based Recommendation Algorithms"**, arXiv 1803.09587
  (UMUAI 2018). https://arxiv.org/pdf/1803.09587 (pages 4–9 read). This is the source of the
  AR/MC/SR/SKNN/V-SKNN/S-SKNN/SF-SKNN definitions in the Implementation spec.
- **Shehzad & Jannach, "Performance Comparison of Session-based Recommendation Algorithms based on
  GNNs"**, ECIR 2024, arXiv 2312.16695. https://arxiv.org/abs/2312.16695
  - Eight GNN models were tuned. The simple kNN/SR models beat all of them on MRR, and were
    "only outperformed in three cases in terms of the Hit Rate".
- **Shehzad & Jannach, "Revisiting the Performance of Graph Neural Networks for Session-based
  Recommendation"**, RecSys 2025 (doi 10.1145/3705328.3748156).
  - Known only from a search snippet: simpler models "often competitive", and one NN baseline was
    strong on RSC15 but below average elsewhere. The full text returned ACM 403 (see "Could not
    verify").
- **Gusak et al. "Time to Split"**, RecSys 2025, arXiv 2507.16289, offers a counterpoint. On 8
  datasets (session plus long-history) with SASRec+, BERT4Rec and GRU4Rec, "SASRec+ is strongest on
  average, while GRU4Rec and SeqKNN often rank lowest".
  - So kNN dominance is **not universal**, especially on long user histories.

### Alternatives
SR (sequential rules), AR, item-kNN, GRU4Rec, NARM, SR-GNN, CT (Context Trees).

### Why it wins for small stores
- There is no trained model, only an inverted index (item → recent sessions). It handles anonymous
  sessions natively.
- It captures *whole-session* context, not just the last item.
- It adapts to trends because neighbours are sampled from the most recent sessions.
- It costs milliseconds per request on CPU.

### Caveats
- Prediction-time cost grows with sample size m and neighbours k, so cap both.
- The reference code (session-rec) has **no license file** and was last pushed 2021-11-05. Its code
  has bugs, observed in source:
  - In `vsknn.py`, `decay` is not reset per neighbour.
  - The `jaccard` similarity uses `time.clock()`, which was removed in Python 3.8.
  - In `vstan.py`, the IDF counts events, not sessions, so IDF values can go negative.
  - The documented defaults do not match the code.
- **Re-implement from the formulas; do not vendor the code.**

---

## 3. SASRec with cross-entropy vs BCE (Klenitskiy & Vasilev; Petrov & Macdonald) — **Choice: the only neural candidate, gated**

### Evidence
- **Klenitskiy & Vasilev, "Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?"**,
  RecSys 2023, arXiv 2309.07602. https://arxiv.org/pdf/2309.07602 (full PDF read).
  - Full-softmax CE ("SASRec+") beats BERT4Rec on 4 of 5 datasets; Steam is the exception.
  - NDCG@10:

    | Dataset | SASRec+ | SASRec (BCE, 1 negative) | BERT4Rec | Notes |
    |---|---|---|---|---|
    | ML-1M | 0.1821 | 0.1341 | 0.1537 | SASRec+ with 3000 negatives: 0.1857 |
    | Beauty (353,962 interactions) | **0.0327** | 0.0126 | 0.0187 | GRU4Rec-CE: 0.0163 |
    | Yelp | 0.0246 | — | 0.0223 | |
    | Steam | 0.0641 | — | **0.0662** | |

  - Training cost on ML-1M: SASRec+ took 540 s against 1409 s for BERT4Rec.
  - Sampled CE with N≈1000 negatives approaches full CE.
  - The code (github.com/antklen/sasrec-bert4rec-recsys23) has **no license**, so treat it as a
    reference only.
- **Petrov & Macdonald, "gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with
  Negative Sampling"**, RecSys 2023, arXiv 2308.07192. https://arxiv.org/pdf/2308.07192 (pages 1–9
  read).
  - Negative sampling with BCE causes overconfidence. With softmax/no sampling, SASRec ML-1M NDCG@10
    rises from 0.131 to 0.169 (+29%), and Steam rises from 0.0581 to 0.0721.
  - gSASRec beats BERT4Rec by +9.47% NDCG on ML-1M, with 73% less training time.
  - gBCE: `L = −1/(k+1) [ log σ(s⁺)^β + Σ_{i∈negatives} log(1−σ(s_i)) ]`.
  - β is set through `β = α(t(1 − 1/α) + 1/α)`, where α = k/(|I|−1) is the sampling rate.
  - Implement it as plain BCE on a transformed positive logit: `γ(s⁺) = log(1/(σ(s⁺)^−β − 1))`.
  - Official PyTorch code: github.com/asash/gSASRec-pytorch (Apache-2.0, last push 2024-02-23).
    Defaults are 256 negatives and t=0.75. ML-1M NDCG@10 is 0.1726 against 0.1259 for SASRec-BCE.

### Alternatives
gSASRec (gBCE), SASRec with sampled softmax, BERT4Rec, eSASRec, TRON.

### Why this choice for small stores
- With |I| ≤ ~50k, full-catalogue softmax is cheap on CPU/MPS. That removes the reason for gBCE or
  sampled negatives. **[ours]** At d=64 and 50k items, the logits for a 128×50 batch are 320M
  multiply-adds.
- Klenitskiy found full-softmax CE best or near-best everywhere.

### Caveats
- Under the Spotify protocol, SASRec-family models lost to the pairwise probes at the
  Amazon/ML-1M scale.
- SASRec overfits quickly on small sparse data (Klenitskiy notes this for Beauty and Yelp), so use
  early stopping and restore the best epoch.
- Gate it: ship only if it beats PCTM/SeqRules on a temporal validation split for that store.

---

## 4. eSASRec — **verified; strongest "simple transformer" if the neural path is ever needed**

### Evidence
- **Tikhonovich, Zelinskiy, Petrov, Spirina, Semenov, Savchenko, Kuliev, "eSASRec: Enhancing
  Transformer-based Recommendations in a Modular Fashion"**, RecSys 2025, arXiv **2508.06450**.
  The ID is correct.
  - URLs fetched: https://arxiv.org/abs/2508.06450 and https://arxiv.org/html/2508.06450
  - Code: github.com/blondered/transformer_benchmark (Apache-2.0, last push 2025-07-10).
  - eSASRec is SASRec's shifted-sequence objective, plus LiGR layers, plus sampled softmax.
  - Time-split "production-like" benchmark, NDCG@10:

    | Model | ML-20M | Kion | BeerAdvocate |
    |---|---|---|---|
    | eSASRec | 0.1563 | 0.1657 | **0.0650** |
    | HSTU | 0.1872 | **0.1694** | 0.0473 |
    | FuXi-α | **0.1883** | 0.1647 | 0.0472 |
    | SASRec + sampled softmax | 0.1527 | 0.1497 | 0.0381 |
    | SASRec vanilla (BCE, 1 negative) | 0.1192 | 0.1389 | 0.0222 |
    | **Popular-7-days** | **0.1671** | **0.142** | 0.0575 |

  - Coverage@10 on Kion: eSASRec 0.3003 against HSTU 0.2110.
  - The text says only eSASRec and HSTU stay on the accuracy–coverage Pareto front on all datasets.
  - **Your claim "roughly matches HSTU" is half-true.** HSTU is about 20% better on ML-20M NDCG and
    about 2% better on Kion. eSASRec is about 37% better on BeerAdvocate. They tie on Pareto
    optimality only once coverage is included.
  - **Notable: a 7-day popularity baseline beats eSASRec and every SASRec variant on ML-20M NDCG
    under the time split.**
  - Academic Amazon results: eSASRec reaches 0.052 NDCG@10 on Beauty, HSTU 0.044, and ActionPiece
    0.042 as reported in its own paper. This is the "+23%" claim. The authors note the gain over HSTU
    on Amazon "did not" hold under realistic settings.

### Caveats
No small-data study. RecTools implements it as `SASRecModel(transformer_layers_type=LiGRLayers,
loss="sampled_softmax")`.

---

## 5. GRU4Rec official implementation & third-party reproducibility — **Choice: do NOT use**

### Evidence
- **Hidasi & Czapp, "The Effect of Third Party Implementations on Reproducibility"**, RecSys 2023,
  arXiv 2307.14956 (doi 10.1145/3604915.3609487). https://ar5iv.labs.arxiv.org/html/2307.14956
  - Six third-party GRU4Rec ports were compared against the official code.
  - Out of the box, Recall@20 was lower by:
    - GRU4REC-pytorch: about 75–99% lower; the worst case in the paper is 99.63%.
    - GRU4Rec_Tensorflow: up to about 94% lower.
    - Recpack: up to about 51% lower.
    - Torch-GRU4Rec: up to about 23% lower.
    - KerasGRU4Rec: up to about 14% lower.
  - Microsoft Recommenders' version is "GRU4Rec in name only".
  - All ports lack shared embedding; none has momentum-Adagrad or logQ correction.
  - RecBole was **not** evaluated.
- **License of the official repos:**
  - The Theano repo (github.com/hidasib/GRU4Rec) has `license.txt`, which I fetched. It says
    **"Non-commercial use only."** Commercial use "requires a specific prior written permission
    from the copyright holders". The license is requested by email (licensing-gru4rec at yusp.com)
    and involves a fee.
  - The official PyTorch port (github.com/hidasib/GRU4Rec_PyTorch_Official, last push 2023-08-24)
    has **no LICENSE file**. Its README says "Final license is coming soon! Meanwhile, use the
    license of the original". In effect it is also non-commercial.
  - **Your recollection is confirmed: the license is restrictive.**
- Official PyTorch port limitations (README): session-based only, with hidden state reset per
  session; no cross-session BPTT. It is 1.5–4× slower than Theano.

### Why it loses for Mitate
- It is unlicensable for a commercial product without paying.
- Third-party ports are unreliable.
- On the e-commerce data in section 2, it rarely beats kNN.

---

## 6. TRON (OTTO) and the OTTO dataset — **Choice: not needed at small-catalogue scale**

### Evidence
- **Wilm, Normann, Baumeister, Kobow, "Scaling Session-Based Transformer Recommendations using
  Optimized Negative Sampling and Loss Functions"**, RecSys 2023, arXiv **2307.14906**.
  https://arxiv.org/html/2307.14906v2
  - Code: github.com/otto-de/TRON (MIT, last push 2025-05-28).
  - Dataset: github.com/otto-de/recsys-dataset (MIT, last push 2025-05-14).
  - Method: SASRec with uniform batch-wise and in-batch session-wise negatives, top-k (100) negative
    selection, and sampled softmax.
  - Recall@20 / MRR@20:

    | Model | Diginetica | Yoochoose | OTTO |
    |---|---|---|---|
    | GRU4Rec+ | 0.455 / 0.144 | 0.725 / 0.310 | 0.443 / 0.205 |
    | SASRec | 0.454 / 0.157 | 0.573 / 0.216 | 0.307 / 0.180 |
    | TRON XL-Negs | 0.541 / 0.182 | 0.732 / 0.302 | 0.472 / 0.219 |

  - Online A/B test against SASRec: CTR +18.14% (95% CI 16.63–19.67%), add-to-carts +23.85%.
  - TRON is compared only against BCE-SASRec and GRU4Rec+, not against kNN or full-softmax SASRec.
  - There is an internal inconsistency: the text claims "92% of SASRec's training speed", but the
    table shows −27.6% / −25.7% / −8.5% epochs per hour.
- OTTO is a dataset where order matters: shuffling cuts SASRec NDCG@10 by 28% and GRU4Rec by 37%
  ("Does It Look Sequential?").

### Why not for small stores
TRON solves negative sampling for a 1.8M-item catalogue. With ≤ 50k items, full softmax is feasible
and is the stronger choice per Klenitskiy & Vasilev.

---

## 7. HSTU / generative recommenders — **Choice: no**

### Evidence
- **Zhai et al., "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for
  Generative Recommendations"**, ICML 2024, arXiv 2402.17152. https://arxiv.org/abs/2402.17152
  - It targets "high cardinality, non-stationary streaming" data with tens of billions of daily
    actions.
  - It claims up to +65.8% NDCG over baselines on public and synthetic data, and a +12.4% online
    gain at 1.5T parameters.
  - Code: github.com/meta-recsys/generative-recommenders (Apache-2.0, last push 2026-10-17).
- Small-data evidence from 2025–2026:
  - Under eSASRec's time split (section 4), HSTU is best or near-best on ML-20M and Kion but loses
    to eSASRec on BeerAdvocate. Popular-7-days comes within about 11% of HSTU on ML-20M.
  - On Amazon (the small datasets), the eSASRec re-run gives HSTU 0.044 NDCG@10 on Beauty against
    eSASRec 0.052. Per the Spotify paper, both are below the pairwise probes (PCTM 0.0635); these
    are different splits, so treat this as indicative only **[ours]**.
  - Betello et al. (IEEE Access 2025, arXiv 2408.03873) found that SASRec "clearly dominate[s]" only
    once parameter counts become large.
  - **I found no study showing HSTU helps below ~1M interactions.**
- RecTools ships `HSTUModel` (Apache-2.0) if a check is ever needed.

---

## 8. Data leakage in sequential evaluation (RecSys 2025, doi 10.1145/3705328.3759329)

### Evidence
- Crossref (https://api.crossref.org/works/10.1145/3705328.3759329) resolves the DOI to:
  **Le, Liu, Medlar, Glowacka, "Don't Get Ahead of Yourself: A Critical Study on Data Leakage in
  Offline Evaluation of Sequential Recommenders"**, RecSys 2025, published 2025-09-07.
  - The ACM page returned 403. I read the study's findings in Yang Liu's PhD thesis (Univ. Helsinki,
    Aug 2025, §3.2, pp. 35–39), which includes it as Article II under the earlier title "Flood
    Warning: Data Leakage in Offline Evaluation of Sequential Recommenders" (under review then).
  - Thesis PDF: https://helda.helsinki.fi/server/api/core/bitstreams/aea49ea6-78e8-4b1f-884c-4b3f9fa99e66/content
- Findings:
  - The study covers 9 sequential models and 4 datasets (ML-1M, Yelp, Steam, Beauty).
  - It compares **temporal LOO** (per-user last item) against **split-by-timepoint LOO**. In the
    latter, a global cutoff is used; for active users, the item right after the cutoff is the test
    item and the item before it is validation. Later interactions are discarded, and inactive users
    are excluded from evaluation.
  - Going from temporal LOO to timepoint LOO drops NDCG@10/Recall@10 by **16.5% to 90.0%**. For
    example, unsampled NDCG@10 falls 90.0% on ML-1M and 78.3% on Beauty.
  - Subsampling the training set to equal size explains only part of the drop, which confirms
    leakage.
  - Model rankings change: "even the worst-performing recommender with temporal LOO can be the
    best-performing model" under timepoint LOO (Steam, unsampled).
  - **A general (non-sequential) recommender beat the best sequential model on most datasets** once
    leakage was removed.
- Corroborated by **Gusak et al., "Time to Split"**, RecSys 2025 (doi 10.1145/3705328.3748164),
  arXiv 2507.16289:
  - LOO's ranking correlation with a global-temporal-split (GTS) reference: Kendall 0.71 against
    0.91 for GTS-Random and 0.83 for GTS-Last.
  - Recommended: GTS with Last, Random or Successive targets, and global temporal validation, then
    retraining on train+validation. Retraining moved Beauty GT-Successive from 0.022 to 0.040.

### Implication
Mitate's offline evaluation must use a global time cutoff. LOO numbers are not comparable, and the
leakage inflates sequential models specifically.

---

## 9. What is strongest at SMALL scale (1K–100K interactions), and crossover evidence

- **No 2025–2026 paper fetched here measures accuracy as a function of dataset size for
  transformer vs Markov/kNN.** Searches turned up none. The only crossover-type signals:
  1. The Spotify probes beat eSASRec at Amazon scale (Beauty has 353,962 interactions in the
     BERT4Rec preprocessing) and at ML-1M (999,611 interactions). They lose by 27.3% at ML-20M
     (20,000,263 interactions). Dataset sizes are from the Klenitskiy & Vasilev Table 1.
     → Crossover lies somewhere between about 1M and 20M interactions *for MovieLens-like dense
     data*. Mitate's range sits entirely below it.
  2. Ludewig et al. 2021: kNN wins on RETAIL (210k actions) and DIGI (264k), and mostly on ZALANDO
     (4.5M). Neural edges ahead only on RSC15 (5.4M actions, 29k items), by about 1% or less. This is
     confounded by dataset characteristics; the authors call RSC15 unusual.
  3. eSASRec time split: a trivial **Popular-7-days** beats all SASRec variants on ML-20M. Recency
     popularity is a must-have baseline and fallback.
  4. Betello et al. 2025: SASRec dominance appears with large parameter counts, which small data
     cannot support.
- **At 1K–100K interactions there is no direct benchmark at all.**
  - **[ours]** By extrapolation, the order is: recent popularity → MC/SeqRules/PCTM → session-kNN
    ensemble. Neural models are unlikely to be estimable reliably at this size.
  - Below about 10K interactions, PCTM's τ smoothing toward popularity effectively makes it a
    popularity model with sparse transition boosts. That is the right degenerate behaviour.

---

## Tooling (licenses checked via GitHub API `license.spdx_id` and `pushed_at`, 2026-10-19)

| Repo / package | License | Last push | CPU/MPS | Notes |
|---|---|---|---|---|
| spotify-research/sequential-capacity-probes | Apache-2.0 | 2026-08-18 | MC/SeqRules/PCTM on CPU (scipy) | Reference impl. for probes. Full repro pins Linux and CUDA. |
| MTSWebServices/RecTools (moved from MobileTeleSystems), PyPI `rectools` 0.19.0 | Apache-2.0 | 2026-10-08 | PyTorch Lightning; CPU. MPS not verified. | SASRec/BERT4Rec/**HSTU**. `loss ∈ {softmax, BCE, gBCE, sampled_softmax}` (read in source); LiGR layers (eSASRec); ItemKNN/EASE/ALS wrappers. **Best maintained, permissive neural option.** |
| sb-ai-lab/RePlay, PyPI `replay-rec` 0.22.0 | Apache-2.0 | 2026-10-16 | "CPU, GPU, Multi-GPU". MPS not stated. | SASRec, BERT4Rec, TwoTower, ItemKNN. Spark and torch are optional extras. |
| asash/gSASRec-pytorch | Apache-2.0 | 2024-02-23 | PyTorch | Official gSASRec; small, unmaintained. |
| blondered/transformer_benchmark | Apache-2.0 | 2025-07-10 | — | eSASRec paper code (uses RecTools). |
| otto-de/TRON | MIT | 2025-05-28 | PyTorch Lightning | Large-catalogue negative sampling. |
| meta-recsys/generative-recommenders | Apache-2.0 | 2026-10-17 | GPU-oriented (not verified on CPU) | HSTU reference. |
| NVIDIA-Merlin/Transformers4Rec | Apache-2.0 | 2026-08-23 | NVIDIA-stack oriented | Not evaluated in depth. Heavy for an 8 GB M1. |
| RUCAIBox/RecBole, PyPI 1.2.1 | MIT | 2025-02-24 | PyTorch | Broad model zoo; activity slowing. Its GRU4Rec was not validated by Hidasi & Czapp. |
| pmixer/SASRec.pytorch | Apache-2.0 | 2026-09-17 | PyTorch | Simple SASRec reference. |
| rn5l/session-rec | **none (no LICENSE file)** | 2021-11-05 | Python 3.5-era, Theano/TF1 | SKNN/V-SKNN/STAN/VSTAN/SR reference; **reference only**. |
| hidasib/GRU4Rec (Theano) | **Non-commercial** | 2023-08-24 | — | Paid commercial license. |
| hidasib/GRU4Rec_PyTorch_Official | **none; defers to the non-commercial original** | 2023-08-24 | PyTorch | Same restriction. |
| antklen/sasrec-bert4rec-recsys23 | **none** | 2024-02-24 | — | Reference only. |
| recommenders-team/recommenders | MIT | 2026-10-19 | — | Its GRU4Rec is "GRU4Rec in name only" (Hidasi & Czapp). |

Non-commercial or unlicensed, so do not vendor: GRU4Rec (both), session-rec, antklen.

---

## Implementation spec

Notation:
- A training **sequence** is one session (anonymous) or one user's time-ordered events.
- History `h = (h_1,…,h_m)` is ordered oldest→newest. `|I|` is the catalogue size.
- `N^{(d)}_{ba}` is the number of times item `a` occurs exactly `d` positions after item `b` within
  the same training sequence.
- Default: consecutive duplicate events are collapsed, and events are sorted stably by (sequence,
  timestamp, row order), as in PROTOCOL.md.

### A. MC (sanity baseline)
`P(a|b) = N^{(1)}_{ba} / Σ_i N^{(1)}_{bi}`. Score with `b = h_m`. No smoothing.

### B. PCTM — Probabilistic Collaborative Transition Model (Petrov et al. 2026; verified against `pctm.py`)
1. **Distance-weighted counts:**
   `C_{ba} = Σ_{d=1}^{W} γ_d · N^{(d)}_{ba}`, with `C_{b·} = Σ_i C_{bi}`.
   `γ_d` comes from a count kernel:
   - `u` (uniform): `γ_d = 1`
   - `exp:r`: `γ_d = r^{d−1}`
   - `inv:p`: `γ_d = d^{−p}` (p=0.5 gives 1/√d; p=1 gives 1/d)
2. **Dirichlet-smoothed transition:**
   `P̂(a|b) = (C_{ba} + τ/|I|) / (C_{b·} + τ)`. τ > 0 is the concentration toward a uniform prior.
3. **Popularity background:**
   `p_pop(a) = (f_a + 1) / (Σ_i f_i + |I|)`, where `f_a` is the training event count.
4. **History kernel.** Let `L = min(m, L_max)` and index `j = 1` as the newest item. Weights are
   non-negative and normalised to sum to 1:
   - Exponential: `E(r)_j = r^{j−1} / Σ_{ℓ=1}^{L} r^{ℓ−1}`
   - Head–tail `H(k, ρ, r)`:
     - for `j ≤ k`: `w_j = ρ · r^{j−1} / Σ_{ℓ=1}^{k} r^{ℓ−1}`
     - for `j > k`: `w_j = (1−ρ)/(L−k)`
     - if `L ≤ k`, renormalise the head weights to sum to 1
   - `last`: `w_1 = 1`
   - `inv:p`: `w_j ∝ j^{−p}`
5. **Score:**
   `s(a|h) = Σ_{j=1}^{L} w_j · log P̂(a | h_{m−j+1}) + λ · log p_pop(a)`
6. **Efficient, rank-equivalent form (what the code does):**
   - Precompute the sparse matrix `E_{ba} = log(1 + |I|·C_{ba}/τ)` over non-zeros only.
   - Then `s_a = (w ⊙ onehot(history)) · E[:, a] + λ · log p_pop(a)`.
   - Candidates with no evidence from any history item get 0 + λ·log p_pop(a). They are filled from
     a popularity-ordered background list (ascending log-pop if λ<0, descending if λ>0).
   - The dropped term `log(τ/|I| / (C_{b·}+τ))` depends only on b, so it is constant across
     candidates.
7. Exclude items in the history (paper default; make it a flag for repeat-purchase stores).
8. Tie-break deterministically. The paper uses SHA-256 of the external item ID.
9. Tuned values in the paper (Table 2 / HYPERPARAMETERS.md):

   | Data | count kernel | τ | history kernel | λ | L_max |
   |---|---|---|---|---|---|
   | Beauty | inv:10:.5 | 11,250 | exp:.6 | −.04 | 50 |
   | Sports | inv:10:.5 | 15,000 | tail:10:.8:.7 | 0 | 50 |
   | Toys | exp:20:.7 | 15,000 | tail:7:.8:.6 | 0 | 50 |
   | ML-1M | inv:20:1 | 150 | tail:7:.8:.6 | −.4 | 200 |
   | ML-20M | exp:10:.5 | 225 | tail:7:.8:.6 | −.5 | 200 |

   Note that **λ ≤ 0** in every tuned configuration, i.e. a popularity *penalty*. **[ours]**
   λ = −1 would make the score a pointwise-mutual-information (lift) style score.
10. **Suggested Mitate search grid [ours, not from the paper]:**
    - W ∈ {3, 5, 10, 20}
    - kernel ∈ {inv p∈{.5, 1}, exp r∈{.5, .7}}
    - τ on a log grid ∈ |I| × {0.01, 0.1, 0.3, 1, 3}
    - history ∈ {last, exp .4/.6/.8, tail:7:.8:.6}
    - λ ∈ {−.5, −.2, 0, .2}
    - Tune on a global-time validation cut.

### C. SeqRules — Sequential Rules (Ludewig & Jannach 2018; sparse variant per Petrov et al. 2026 `seqrules.py`)
1. **Rule matrix:** `R_{ba} = Σ_{d=1}^{W} g(d) · N^{(d)}_{ba}`, with `g` chosen from:
   - `same`: 1
   - `div`: `1/d`
   - `quadratic`: `1/d²`
   - `log`: `1/log10(d + 1.7)`
   - `linear`: `max(0, 1 − 0.1·d)` (the code applies this up to d ≤ 100)
2. **Optional IDF on source rows:**
   - `idf_b = minmax( ln(U / f_b) )`, where U is the number of sequences and `f_b` is the event count.
   - `R_{b·} ← idf_b · R_{b·}`, computed before pruning.
3. **Optional row pruning:** keep the top-K entries per source row, breaking ties by ascending item
   ID.
4. **Score:** `S(a|h) = Σ_{j=1}^{L} v(j) · R_{h_{m−j+1}, a}`, where `v(1) = 1` and
   `v(j) = g_v(j)` for j ≥ 2. `g_v` is from the same family; the tuned values used `div` or
   `quadratic`.
5. Exclude history items. Fill the remaining slots with background items.
6. Tuned values in the paper:

   | Data | W | g | K | IDF | L | v |
   |---|---|---|---|---|---|---|
   | Beauty | 10 | log | 100 | no | 10 | quadratic |
   | Sports | 30 | log | ∞ | no | 20 | div |
   | Toys | 30 | log | ∞ | no | 10 | div |
   | ML-1M | 10 | div | 200 | yes | 10 | quadratic |
   | ML-20M | 20 | div | 100 | yes | 5 | quadratic |

7. **Original SR (Ludewig & Jannach 2018, Eq. 3)** is the special case with L = 1 (last item only)
   and `g(d) = 1/d`. It is normalised by the total rule weight leaving `h_m`, which does not change
   the ranking.

### D. Session-kNN family (Ludewig & Jannach 2018, Eqs. 5–7; STAN/VSTAN per session-rec code)
Notation:
- `s` is the current session.
- `N_s` is the top-k neighbour sessions, chosen from a candidate pool. The pool is the m most recent
  past sessions that share ≥ 1 item with s, found via an item→sessions inverted index; use
  `sampling=recent`.
- `1_n(i) = 1` if session n contains item i.

Variants:
- **SKNN:**
  - `score(i,s) = Σ_{n∈N_s} sim(s,n)·1_n(i)`, with binary cosine `sim = |s∩n| / √(|s|·|n|)`.
  - The 2018 paper used k = 500 and noted m ≈ 1000 recent sessions sufficed.
- **V-SKNN** (vector-multiplication SKNN):
  - The current session is a real-valued vector; the last item has weight 1 and earlier items decay.
  - Position weights for position p of L (session-rec code): `same` = 1, `div` = p/L,
    `quadratic` = (p/L)², `log` = 1/log10((L−p)+1.7), `linear` = 1 − 0.1(L−p) (clip at 0 [ours]).
  - `sim(s,n) = Σ_{i∈s∩n} w(i) / |s|` (`vec` similarity), or the weighted-cosine variant.
  - Sequence-aware scoring multiplies each neighbour's contribution by `ws(step)`. Here `step` is
    the recency rank (1 = newest) of the most recent current-session item that also appears in n.
    `ws ∈ {div: 1/step, quadratic: 1/step², log: 1/log10(step+1.7), linear: 1−0.1·step}`.
  - Optional IDF: `sim ← sim·(1 + λ_idf·idf(i))`, with `idf(i) = ln(#sessions / #sessions containing
    i)`. **Count sessions, not events; session-rec's VSTAN gets this wrong.**
- **S-SKNN:** `score = Σ_n sim(s,n)·w_n(s)·1_n(i)`, where `w_n(s) = x/|s|` and x is the position of
  the most recent item of s that appears in n.
- **SF-SKNN:** same as SKNN, but `1_n(s_last, i) = 1` only if the consecutive pair (s_last, i) was
  observed in some training session (and i ∈ n).
- **STAN** (Garg et al., SIGIR 2019; formulas from session-rec `stan.py`):
  - Current-item weight: `w(p) = exp((p − L)/λ_spw)`, using the last occurrence of each item.
  - Similarity: `sim(s,n) = [Σ_{i∈s∩n} w(i)] / (√(Σ_{i∈s} w(i)²)·√|n|) · exp(−(t_now − t_n)/λ_snh)`,
    where `t_n` is the last event time of n.
  - Keep the top-k neighbours by sim.
  - Item-in-neighbour weight: `d(i,n) = exp(−|pos_n(i) − pos*_n| / λ_inh)`. Here `pos*_n` is the
    last position in n of any item shared with s, and `pos_n(i)` is the last position of i in n.
  - `score(i) = Σ_{n∈N_s, i∈n} sim(s,n)·d(i,n)`.
  - Code defaults: λ_spw = 1.02, λ_snh = 5 days, λ_inh = 2.05, sample_size = 5000.
- **VSTAN** (Ludewig et al. 2021) is STAN plus V-SKNN's sequence-aware scoring plus IDF:
  - `score(i) = Σ_n sim'(s,n) · d(i,n) · (1 + λ_idf·idf(i)) · exp((i*+1−L)/λ_ipw)`
  - `i*` is the 0-indexed position in s of the most recent item shared with n. Code defaults are
    λ_ipw = 1.02 and λ_idf = 5.
  - Pick i* as the max position; the reference code's choice depends on iteration order (a bug).
- Common flags:
  - `remind`: allow or suppress items already in s.
  - Normalise scores by the maximum score, so they can be blended with other models.

**Serving note [ours]:** keep `item → deque(recent session ids)` plus `session id → (items,
last_ts)` in memory. A request costs O(m + k·avg_session_len).

---

## Recommendations for Mitate

1. **Default next-item model: PCTM.** Run SeqRules alongside it and pick per store, or take the
   "pairwise envelope" (the better on validation). Implement both on `scipy.sparse`; there is no
   torch dependency. Retrain nightly in seconds.
2. **Anonymous-session context: VSTAN (or V-SKNN)**, re-implemented from the spec with the bug
   fixes noted above.
   - Blend it with PCTM via a validation-fitted linear weight on normalised scores.
   - For logged-in users, feed PCTM the cross-session history; VSTAN uses only the current session.
3. **Always ship a recency-popularity fallback** (Popular-last-N-days). Use it for cold sessions and
   as the background fill. It beat SASRec variants under a time split in the eSASRec paper.
4. **Neural path is opt-in and gated:**
   - Use RecTools `SASRecModel(loss="softmax")`, Apache-2.0, on CPU/MPS.
   - Enable it only for stores with roughly ≥ 1M interactions **and** only if it beats the pairwise
     envelope on a global-temporal validation split.
   - Use gBCE or sampled softmax only if |I| makes full softmax too slow; this is unlikely at
     Mitate's scale.
   - Skip HSTU, TRON, GRU4Rec and BERT4Rec.
5. **Diagnostic before sequential modelling:** run the "Does It Look Sequential?" shuffle test per
   store. If shuffling the test histories barely changes NDCG, sequence order adds little; use
   co-occurrence (AR / item-kNN) or set PCTM's history kernel to uniform.
6. **Evaluation protocol:**
   - Global temporal split with Last or Successive targets.
   - Validation on an earlier global cut, then retrain on train+validation.
   - Full-catalogue metrics only; never sampled metrics and never LOO.
   - Report MC, popularity and the pairwise envelope as mandatory baselines.
7. **Licensing:** vendor nothing from GRU4Rec, session-rec or antklen. The permissive options are
   RecTools, RePlay, gSASRec-pytorch, the Spotify probes (Apache-2.0) and TRON (MIT).

---

## Could not verify

- **"Don't Get Ahead of Yourself" (doi 10.1145/3705328.3759329):**
  - Title, authors and venue were confirmed via Crossref. The ACM page and PDF returned 403.
  - The findings above come from the authors' pre-publication version ("Flood Warning…") as
    summarised in Yang Liu's thesis. The published numbers may differ.
  - I found no arXiv version.
- **Shehzad & Jannach, RecSys 2025 GNN revisit (doi 10.1145/3705328.3748156):** ACM returned 403.
  The only claim used comes from a search snippet.
- **Latifi, Jannach, Ferraro, "Sequential recommendation: A study on transformers, nearest neighbors
  and sampled metrics"** (Information Sciences 609, 2022): bibliographic record confirmed via
  citations only. Content not read, so no findings are used.
- **Han et al. 2605.07125:** abstract only. The exact heuristic, the full dataset list and the code
  were not read.
- **"Cold-Starts in Generative Recommendation: A Reproducibility Study"** (arXiv 2603.29845):
  abstract only. Whether it includes HSTU, and its small-data results, are unknown.
- **The SASRec (2018) claim that Markov-chain methods win on very sparse data:** seen in a search
  snippet only; not fetched.
- **A blog's claim that MARS needs ≥ 34 interactions per item:** unverified; not used.
- **Direct crossover evidence:** no 2025–2026 study of transformer vs Markov/kNN accuracy against
  dataset size was found. The crossover statements above are inferences from cross-dataset results.
- **Last-commit dates:** these are GitHub `pushed_at`, which counts any branch push. They are not
  release dates.
- **Platform support:** MPS support for RecTools and RePlay is not stated in their docs. CPU support
  for meta-recsys/generative-recommenders and Merlin was not tested.
- **Minor discrepancy:** the Spotify paper text reports +21.3% and +14.8% on Beauty and Sports. Its
  Table 1 implies +21.2% and +14.5% **[ours]**.
