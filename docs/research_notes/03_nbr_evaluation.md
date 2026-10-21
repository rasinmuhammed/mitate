# 03: Next-basket / repeat-purchase recommendation, and offline + online evaluation methodology

Scope: (A) next-basket (NBR) and repeat-purchase baselines for Mitate; (B) the offline benchmark protocol (splits, metrics, statistics, leakage, tuning budgets) and the online experiment module (interleaving, CUPED, power).
Review date: 2026-10-19. Every number below was read from the URL listed with it (PDFs were downloaded and text-extracted; code repos were cloned). Anything I could not open is listed under "Could not verify".

Notation: R@k = Recall@k, N@k = NDCG@k, PHR@k = personalized hit ratio. "Repeat item" = item the user bought in any basket strictly before the target basket. "Explore item" = any other item.

---

## A1. NBR reality check: baselines, repeat/explore metrics, and a pipeline-validation target

**Choice:** GP-TopFreq is the floor every Mitate basket model must beat, and repeat/explore-decomposed metrics are reported for every model. We reproduce GP-TopFreq on the authors' preprocessed Instacart to validate our metric code before trusting any other number.

**Evidence:**
- **Li, Jullien, Ariannezhad, de Rijke. "A Next Basket Recommendation Reality Check". ACM TOIS (arXiv v2, 9 Mar 2023).** https://arxiv.org/abs/2109.14233 (PDF fetched). Code and preprocessed data: https://github.com/liming-7/A-Next-Basket-Recommendation-Reality-Check (cloned, HEAD `ca1a58e`, 2022-06-28).
  - **Baselines (Sec. 3.2.1):**
    - G-TopFreq: the k most popular items in the dataset.
    - P-TopFreq: the user's k most frequent historical items. It contains repeat items only and may leave slots empty. At k=10 it fills 96.70% of Instacart slots, at k=20 it fills 90.48% (Table 2).
    - GP-TopFreq: "first uses P-TopFreq to fill a basket, and then uses G-TopFeq to fill any remaining slots."
  - **Code-level definitions (`methods/g-p-gp-topfreq/*.py`):**
    - Personal counts are item occurrences over all history baskets. Ties keep first-occurrence order (Python stable sort over dict insertion order).
    - The global list `instacart_pop.csv` is counts over the history baskets of all 19,435 users. I recomputed it and the top-10 matched exactly.
    - Lists are padded to length 100.
  - **Conventional metrics (Eqs. 1–3):**
    - Recall@K = |P∩T|/|T|.
    - NDCG@K = Σ_{k≤K} p_k/log2(k+1) ÷ Σ_{k=1}^{min(K,|T|)} 1/log2(k+1).
    - PHR@K = fraction of users with P∩T ≠ ∅.
    - All are averaged over test users. Precision and F1 are dropped because K is fixed, which makes them proportional to recall.
  - **IMPORTANT, code ≠ paper for NDCG.** `evaluation/metrics.py::get_NDCG` computes IDCG over **all |T|** truth items (`for i in range(len(truth_list))`), not min(K,|T|). My recomputation below shows the published Table 4 NDCG values match the **code** version, not Eq. 2. To reproduce their numbers we must implement `ndcg_li2023` (IDCG over |T|) alongside the standard `ndcg` (IDCG over min(K,|T|)).
  - **Repeat/explore metrics (Eqs. 4–6):**
    - RepR = (1/N) Σ_j |P^rep_j|/K, and ExplR = (1/N) Σ_j |P^expl_j|/K.
    - Recall_rep = (1/N_r) Σ_{j∈U_r} |P_j∩T^rep_j|/|T^rep_j|, and PHR_rep = (1/N_r) Σ_{j∈U_r} φ(P_j, T^rep_j).
    - Recall_expl and PHR_expl are defined the same way over U_e with T^expl.
    - N_r (N_e) = number of users whose truth basket contains at least one repeat (explore) item. Users without any are excluded from that average.
    - Code check (`model_performance.py`): Recall_rep is computed against the **full predicted top-K list** (`get_Recall(truth_repeat, pred, size)`), not only against P^rep. The two are identical, because a repeat truth item can only be hit by a repeat prediction.
  - **Exact Instacart preprocessing (`preprocess/Instacart.py`, the authoritative version):**
    1. Read `orders.csv` (user_id, order_number, order_id, eval_set) and concat `order_products__prior.csv` + `order_products__train.csv`. Inner-merge on order_id and drop NaN, which removes Kaggle "test" orders because they have no products.
    2. **User sampling is deterministic, not random:** keep `user_id <= int(0.1 * n_users)`, i.e. the first 10% of user IDs.
    3. Keep users with **3 ≤ #orders ≤ 50**. The paper's phrase "basket size between 3 and 50" means the number of baskets per user. Baskets are deduplicated item sets per order_number.
    4. The last order is the target ("future") basket. All earlier orders are history.
    5. **Item filter:** keep items with ≥ **17** occurrences in the history baskets of the sampled users. The filter is applied to history and target baskets alike.
    6. Re-index users and items.
    - Result (Table 1): **13,897 items, 19,435 users**, avg basket size 9.61, avg 13.21 baskets/user, repeat ratio 0.597, explore ratio 0.403. I verified 19,435 users with no empty target basket and 2–49 history baskets in `jsondata/`.
  - **Split (`keyset_fold.py`):** users are shuffled (`random.shuffle`, **no seed**) into 72% train / 8% val / 20% test, users disjoint, repeated 5 times. Keyset files are not in the repo. Reported numbers are the mean (std) over the 5 folds. This is **not** a temporal split; the paper justifies that in Sec. 3.1.
  - **Table 4, Instacart (mean (std) over 5 user splits):**

    | Method | R@10 | N@10 | PHR@10 | R@20 | N@20 | PHR@20 |
    |---|---|---|---|---|---|---|
    | G-TopFreq | 0.0710 | 0.0811 | 0.4542 | 0.0990 | 0.0962 | 0.5248 |
    | P-TopFreq | 0.3260 | 0.3378 | 0.8449 | 0.4307 | 0.3939 | 0.8957 |
    | **GP-TopFreq** | **0.3269** (0.0008) | **0.3383** (0.0007) | 0.8463 | **0.4354** (0.0007) | **0.3961** (0.0003) | 0.9011 |
    | UP-CF@r | 0.3506 | 0.3631 | 0.8652 | 0.4591 | 0.4222 | 0.9079 |
    | **TIFUKNN** | **0.3601** (0.0015) | **0.3721** (0.0008) | 0.8642 | **0.4709** (0.0015) | **0.4323** (0.0003) | 0.9097 |
    | DNNTSP | 0.3330 | 0.3412 | 0.8525 | 0.4423 | 0.4000 | 0.9042 |
    | Dream | 0.0712 | 0.0805 | 0.4551 | 0.0997 | 0.0957 | 0.5304 |

    The other datasets at R@10 (GP-TopFreq / TIFUKNN) are TaFeng 0.1211 / 0.1251 and Dunnhumby 0.2356 / 0.2396. On TaFeng, DNNTSP is best (R@10 0.1526).
  - **Table 5, Instacart repeat/explore:**
    - GP-TopFreq: Recall_rep@10 0.5388, Recall_expl@10 0.0014, PHR_rep@10 0.9085, PHR_expl@10 0.0046. At @20: Recall_rep 0.7193 and Recall_expl 0.0084.
    - TIFUKNN: Recall_rep@10 0.5931, Recall_expl@10 0.0018. At @20: Recall_rep 0.7747 and Recall_expl 0.0108.
    - G-TopFreq @10: Recall_rep 0.1002, Recall_expl 0.0382, PHR_rep 0.4138, PHR_expl 0.1575.
  - Significance in Tables 4–5 uses a paired t-test with p < 0.05. Repeat items account for "over 97%" of the performance of TIFUKNN, UP-CF@r, Sets2Sets and DNNTSP on Dunnhumby and Instacart (Sec. 5.4).
  - TIFUKNN Instacart hyperparameters used (README): `900 0.9 0.7 0.9 3 20`. These map to k=900 neighbours, r_b=0.9, r_g=0.7, α=0.9, m=3 groups, topk=20 (argument order checked in `tifuknn_new.py::main`).
- **My own pipeline-validation run** (scratch scripts on the repo's preprocessed `dataset/instacart_history.csv`, `jsondata/instacart_future.json` and `instacart_pop.csv`; deterministic, **all 19,435 users**, no split needed because TopFreq has no training):

  | Method | R@10 | N@10 (code IDCG=|T|) | N@10 (Eq. 2 IDCG=min(K,|T|)) | PHR@10 | R@20 | N@20 (code) | PHR@20 |
  |---|---|---|---|---|---|---|---|
  | G-TopFreq | 0.0727 | 0.0822 | n/a | 0.4543 | 0.1006 | 0.0970 | 0.5258 |
  | P-TopFreq | 0.3294 | 0.3412 | n/a | 0.8409 | 0.4315 | 0.3959 | 0.8921 |
  | GP-TopFreq | 0.3304 | 0.3417 | 0.3866 | 0.8422 | 0.4361 | 0.3979 | 0.8971 |

  - The mean over 5 random 20% test-user subsets (seeds 0–4) for GP-TopFreq: R@10 0.3282, N@10 (code) 0.3407, PHR@10 0.8409. All are within about 0.005 absolute of Table 4.
  - The Eq. 2 normaliser gives 0.3866, far from the published 0.3383. This confirms the code NDCG produced the table.
  - My all-user G-TopFreq repeat/explore values match their Table 5 **exactly** at @10 and @20: 0.1002 / 0.0382 / 0.4138 / 0.1575 and 0.1335 / 0.0602 / 0.4767 / 0.2251.
  - GP-TopFreq: Recall_rep@10 0.5429 vs 0.5388 and PHR_rep 0.9072 vs 0.9085. RepR@10 = 0.9664, vs 96.70% slot usage in Table 2.
  - **Acceptance test for Mitate's metric module:** on this data, all-user GP-TopFreq must give R@10 = 0.3304, ndcg_li2023@10 = 0.3417 and PHR@10 = 0.8422, each ±0.0005. G-TopFreq Recall_rep@10 must give 0.1002. These are deterministic, so any deviation is a bug in our code.

**Alternatives:** P-TopFreq alone (leaves empty slots, so it is unfair to compare); G-TopFreq alone (the only baseline in many deep-NBR papers, and far too weak).

**Why:** On high-repeat grocery data GP-TopFreq is within about 0.03 R@10 of the best published deep models (Table 4), and repeat items carry over 97% of the accuracy of the strong models. For small stores with consumables, the repeat channel is the product.

**Caveats:**
- The protocol is a random **user** split with each user's last basket as target. It does not respect a global timeline (see B1 and B5). Use it only to validate the pipeline against published numbers, never as Mitate's protocol of record.
- Instacart has no absolute timestamps. Orders carry `order_number`, `order_dow`, `order_hour_of_day` and `days_since_prior_order` only (schema confirmed in secondary sources; Kaggle page not fetched). A true global temporal split is impossible on Instacart. TaFeng (`TRANSACTION_DT`) and the Dunnhumby "Let's Get Sort-of-Real" 50k data (`SHOP_DATE`) used in the repo do have dates.
- Instacart preprocessing differs across papers: TIFU-KNN uses 8,000 items and 19,935 users; TREx uses 29,399 items and 19,210 users. Never compare absolute numbers across papers without identical preprocessing.

---

## A2. TIFU-KNN (primary personalized repeat+collaborative model)

**Choice:** TIFU-KNN, implemented exactly as the reference code, tuned with the same budget as every other model.

**Evidence:**
- **Hu, He, Gao, Zhang. "Modeling Personalized Item Frequency Information for Next-basket Recommendation". SIGIR 2020.** DOI 10.1145/3397271.3401066. https://arxiv.org/abs/2006.00556 (PDF fetched). Code: https://github.com/HaojiHu/TIFUKNN (not cloned; I read the copy vendored in the reality-check repo, `methods/tifuknn/tifuknn_new.py`).
  - **Algorithm, paper Sec. 3.2:**
    1. Each basket is a multi-hot vector over items.
    2. Partition a user's t history baskets into m groups. Group size x = ⌈t/m⌉, and the first group has size t − x(m−1).
    3. Within a group, the j-th vector is weighted r_b^(x−j) and the group vector is the mean of the weighted vectors.
    4. The i-th group vector is weighted r_g^(m−i). The user vector **u** is the mean of the weighted group vectors (the PIF vector).
    5. Find the k nearest neighbours of the target user by **Euclidean distance** on **u**. u_n is the mean of the neighbours' vectors.
    6. Score P = α·u_t + (1−α)·u_n and recommend the top-s entries.
  - **Code differences to match exactly (`tifuknn_new.py`):**
    - The within-basket decay is `r_b^(num_vec − idx)` over the **whole** history index. It is not reset per group.
    - Groups are averaged (`sum/base_num_vec_each_block`).
    - The group weight is `r_g^(group_size − 1 − idx)`, and the final vector is divided by `real_group_size`. When t < m, each basket is its own group.
    - The neighbour pool is **train users only**: `NearestNeighbors(n_neighbors=k, algorithm='brute')`, i.e. sklearn default Minkowski p=2.
    - The output is the argsort top-100.
  - **Hyperparameters (Table 3):**

    | Dataset | k | m | r_b | r_g | α |
    |---|---|---|---|---|---|
    | Instacart | 900 | 3 | 0.9 | 0.7 | 0.9 |
    | Dunnhumby | 900 | 3 | 0.9 | 0.6 | 0.2 |
    | TaFeng | 300 | 7 | 0.9 | 0.7 | 0.7 |
    | ValuedShopper | 300 | 7 | 1 | 0.6 | 0.7 |

  - **Search grid (Sec. 5.1.4):**
    - k ∈ {100, 300, …, 1300}.
    - r_b, r_g ∈ {0.1, …, 1}.
    - α ∈ {0, 0.1, …, 1}.
    - m ∈ {2, …, 7}; ValuedShopper uses {3, 7, …, 23}.
    - Selected on validation users (10% of the training users).
  - Original protocol: 5-fold user split (4 train / 1 test), last basket as target.
  - Instacart TIFU-KNN R@10 0.3952, R@20 0.4875, N@10 0.3825, N@20 0.4384. PersonTopFreq R@10 is 0.3426. These come from different preprocessing than A1.
- **Garcarz, Pal, Praat. "[RE] Modeling Personalized Item Frequency Information for Next-basket Recommendation". arXiv 2402.17925 (2024).** https://arxiv.org/abs/2402.17925 (PDF fetched).
  - The reproduction confirms TIFU-KNN > TopPersonal on every dataset, with original hyperparameters.
  - Instacart R@10 0.4308 (+0.0356 vs reported) and R@20 0.4962. The authors attribute the deviation to preprocessing.
  - Tmall and Taobao were tuned with Optuna.

**Alternatives:** UP-CF@r (Faggioli et al., UMAP 2020; not fetched), about 0.01 below TIFU-KNN on Instacart R@10 in A1. TIFU-KNN-TA and TAIW (A3).

**Why:** Its repeat component is a time-decayed personal frequency vector, so it sits strictly above GP-TopFreq. It is O(users × items) and trivially fast at small-store scale, and it is consistently top or tied-top in independent re-evaluations (A1, A3).

**Caveats:**
- The kNN step needs enough users. With a few hundred customers k=900 is meaningless, so scale k with |users| in the search space.
- Euclidean distance on unnormalised vectors favours users with similar history length.
- Reproduced numbers vary by ±0.05 R@10 with preprocessing.

---

## A3. Newer repeat/explore and NBR work (2022–2026), and stronger simple methods

**Choice:** Add TREx-Rep (a closed-form repeat scorer) as a second simple baseline. Treat TAIW as the "time-aware" upgrade if Mitate has timestamps and a known prediction time. Do not adopt neural NBR (ReCANet, DNNTSP, SAFERec) as defaults.

**Evidence:**
- **Ariannezhad, Jullien, Li, Fang, Schelter, de Rijke. "ReCANet: A Repeat Consumption-Aware Neural Network for Next Basket Recommendation in Grocery Shopping". SIGIR 2022, pp. 1240–1250.** DOI 10.1145/3477495.3531708 (via Crossref). UvA DARE record fetched: https://dare.uva.nl/id/5d0a13aa-7de9-4d1f-b78e-da9859f79041. Full text not fetched.
  - Across 6 grocery datasets, "over 54% of NBR performance in terms of recall comes from repeat items", which are about 1% of the catalogue on average.
  - The model recommends repeat items only.
- **Li, Liu, Jullien, Ariannezhad, Yates, Aliannejadi, de Rijke. "Are We Really Achieving Better Beyond-Accuracy Performance in Next Basket Recommendation?" SIGIR 2024.** https://arxiv.org/abs/2405.01143 (PDF fetched). Code: https://github.com/lynEcho/TREX.
  - **TREx-Rep (Eqs. 3–6):**
    - RepS_u(i) = E^u_i · RepI(i).
    - RepF(i) = Σ_users (repurchase frequency of i)^α ÷ (#users who bought i at least once).
    - RepI(i) = RepF(i) + mean(RepF)/N_i.
    - E^u_i = Σ_j β^(T − l_j) over the positions l_j of i in the user's T baskets.
    - Grid: α ∈ {0, 0.1, …, 1}, β ∈ {0.7, 0.75, …, 1.0}.
    - The paper does not define "repurchase frequency" more precisely than this. Check the repo before implementing.
  - Instacart (their preprocessing) R@10 / N@10 / PHR@10:

    | Method | R@10 | N@10 | PHR@10 |
    |---|---|---|---|
    | GP-TopFreq | 0.3150 | 0.3343 | 0.8460 |
    | TIFUKNN | 0.3456 | 0.3657 | 0.8639 |
    | ReCANet | 0.3490 | 0.3699 | 0.8668 |
    | TREx-Rep | 0.3476 | 0.3661 | 0.8655 |

    TREx-Rep is statistically tied with the best (paired t-test).
  - On Dunnhumby, TREx-Rep R@10 0.1815 is best, with TIFUKNN 0.1763.
  - Key warning: the decoupled repeat + explore design gets "better" diversity and fairness scores while accuracy stays almost entirely in the repeat items. The authors argue this exposes flaws in how beyond-accuracy metrics are evaluated for NBR.
- **Li, Ariannezhad, Yates, de Rijke. "Masked and Swapped Sequence Modeling for Next Novel Basket Recommendation in Grocery Shopping". RecSys 2023.** https://arxiv.org/abs/2308.01308 (PDF fetched).
  - Defines next **novel** basket recommendation (explore items only) and finds "limited progress" by existing NBR methods on it. Proposes BTBR (a bidirectional transformer with masking and swapping).
  - Relevant if Mitate wants a separate "discover" slot.
- **Liu, Li, Aliannejadi, de Rijke. "Repeat-bias-aware Optimization of Beyond-accuracy Metrics for Next Basket Recommendation". ECIR 2025.** https://arxiv.org/abs/2501.06362 (PDF fetched).
  - Optimizing diversity or item fairness without accounting for repeat bias can push NBR models to recommend **more** repeat items.
  - Proposes model-agnostic re-ranking that mitigates repeat bias "at acceptable Recall loss".
- **Romanov, Lashinin, Ananyeva, Kolesnikov. "Time-Aware Item Weighting for the Next Basket Recommendations" (TAIW). RecSys 2023.** https://arxiv.org/abs/2307.16297 (PDF fetched). Code: https://github.com/alexeyromanov-hse/time_aware_item_weighting.
  - Uses timestamps and inter-basket intervals (Hawkes-process style).
  - Leave-one-basket protocol: last basket = test, penultimate = validation.
  - R@10 TaFeng TAIW 0.1565 vs TIFU-KNN 0.1360 vs TIFU-KNN-TA 0.1503. Dunnhumby 0.1791 vs 0.1653. TaoBao 0.1177 vs 0.0749, where GP-Pop gets 0.1116.
  - Cites TIFU-KNN-TA (Naumov et al., "Time-Dependent Next-Basket Recommendations", ECIR 2023; not fetched) and SLRC (WWW 2019) as other strong time-aware baselines.
- **Lashinin, Krasilnikov, Milogradskii, Ananyeva. "SAFERec: Self-Attention and Frequency Enriched Model for Next Basket Recommendation". arXiv 2412.14302 (Dec 2024).** https://arxiv.org/abs/2412.14302 (PDF fetched).
  - Transformer plus a frequency module, "up to 8%" R@10 gain on one dataset (TaoBao: 0.1619 vs UP-CF 0.1503).
  - On Dunnhumby it is not best: DNNTSP has 0.1276 vs SAFERec 0.1256.
  - Protocol: leave-one-basket, half of users val and half test, Optuna with **50 trials per model**, paired t-test with Bonferroni.
- **Kolesnikov, Lashinin, Pechatov, Kosov. "Next Period Recommendation Reality Check". arXiv 2110.05589.** https://arxiv.org/abs/2110.05589 (PDF fetched).
  - The next-period task (all purchases in the next time window, no explicit basket) is closest to a store's weekly email.
  - Found a "strong repetitive consumption pattern" and that novel-item prediction "is still questionable".

**Alternatives:** DNNTSP, Sets2Sets, Dream, Beacon, CLEA (all ≤ TIFU-KNN on high-repeat data in A1). ReCANet (repeat-only and neural; TREx-Rep matches it at a fraction of the cost).

**Why:**
- Recent work agrees that frequency × recency of personal purchases is the dominant signal.
- Simple closed-form scorers (GP-TopFreq, TIFU-KNN, TREx-Rep, TAIW) tie or beat neural models while being cheap, deterministic and explainable ("you buy this every ~2 weeks").

**Caveats:**
- Every paper here uses user-split or leave-last-basket protocols, so none of the reported gaps are measured under a global timeline.
- Explore performance is near zero for all methods. Recall_expl@10 is at most 0.02 on Instacart. Explore recommendations need their own evaluation slice (Recall_expl, PHR_expl) or they look like pure cost.

---

## B1. Data splitting: global temporal split ("Time to Split") adapted to baskets

**Choice:**
- Use a global temporal split (GTS) with test cutoff T_test and a validation cutoff T_val < T_test (global-temporal validation).
- Target selection is **Successive**: every post-cutoff basket is a target, the input is all history before that basket, and the model state is frozen at the cutoff. Average within each user first, then across users.
- **Last** (each user's last post-cutoff basket) is the cheap deterministic variant.
- Retrain on train+val before test.

**Evidence:**
- **Gusak, Volodkevich, Klenitskiy, Vasilev, Frolov. "Time to Split: Exploring Data Splitting Strategies for Offline Evaluation of Sequential Recommenders". RecSys 2025, pp. 874–883, DOI 10.1145/3705328.3748164.** https://arxiv.org/abs/2507.16289 (PDF fetched). Code: https://github.com/monkey0head/time-to-split.
  - **Definitions:**
    - GTS assigns all interactions after T_test to the holdout. For test and validation inference, all events before the target are used as input, "regardless of whether they occur before or after the global split point".
    - Targets: Last, First, Random, Successive, or All (the whole holdout as one set).
    - Validation options: Global Temporal (cutoff T_val), Last Training Item (LTI), User-Based (UB).
  - **Their main setting:**
    - T_test = the 0.9 quantile of interactions.
    - Sequences of length 1 are removed.
    - For sequences starting after the cutoff, the first item is excluded as a target.
    - The All target is applied only to sequences starting before the cutoff.
    - Successive metrics are averaged per user first.
    - GT and UB validation use the Last target. UB samples 1024 users.
    - Metrics are unsampled. RePlay is used for metrics.
  - **Agreement with GTS-Successive (Table 6, Kendall τ for HR@10 / NDCG@10):** LOO 0.71 / 0.71, GTS Last 0.83 / 0.83, **GTS Random 0.91 / 0.91**, GTS First 0.70 / 0.62, GTS All 0.57 / 0.43.
  - "We do not recommend using the First item after the global timepoint as a target": the time gap before the First target is much larger.
  - The All target "significantly deviates from the next-item prediction objective".
  - GT validation aligns best with GTS test. LTI and UB "lag only slightly behind" but share LOO's drawbacks.
  - **Retraining (Table 8, SASRec+ NDCG@10):** GT-Successive on Beauty goes 0.022 → 0.040 (+81.8%), and on Diginetica 0.149 → 0.160. "Retraining on combined training and validation data is important for achieving optimal deployment performance", but it does not substantially change model ranking.
  - Survey: 77.3% of 75 RecSys/SIGIR/CIKM 2022–24 sequential-recommendation papers used LOO, and only 6.7% used GTS with next-item targets.
- **Meng, McCreadie, Macdonald, Ounis. "Exploring Data Splitting Strategies for the Evaluation of Recommendation Models". RecSys 2020.** https://arxiv.org/abs/2007.13237 (PDF fetched).
  - Strategies compared: leave-one-last item, leave-one-last basket, temporal global. Two datasets incl. TaFeng and 7 models.
  - Over 230 hyperparameter variants of NeuMF, VBCAR and Triple2Vec, Kendall τ between strategies ranges **0.5284–0.7630**.
  - Best-practice list: (1) report the split with train/val/test statistics and filtering; (2) "report performance under temporal global splitting… should be the default"; (3) release the splits.

**Alternatives:** Leave-last-basket per user (the NBR standard; leaks the global timeline; see B5). Random user split (A1; leaks). Sliding-window or prequential "timeline scheme" (Ji et al., B5): more faithful but costs k retrains.

**Why:** A store owner deploys a model at time T and it serves until the next retrain. Only GTS simulates that. For baskets, the analogue of next-item targets is next-basket targets after the cutoff.

**Caveats:**
- Time to Split studies **next-item** prediction. Its finding that "First is biased" is driven by session gaps. In NBR the first post-cutoff basket is a legitimate target, since that is exactly the next shop, but it carries a longer inter-purchase gap.
- Successive covers all post-cutoff baskets, so it is the safer default. Report Last as well and check rank agreement. This adaptation is my inference, not a result from the paper.
- Small stores often have few post-cutoff baskets. Choose T_test by a minimum count of test users and baskets, not a fixed quantile, and report the test-period length.
- Instacart cannot be globally time-split (A1 caveat).

---

## B2. Sampled vs full-catalogue metrics

**Choice:** Always rank the **full** catalogue (minus items not purchasable at the target time). Never use sampled negatives.

**Evidence:**
- **Krichene, Rendle. "On Sampled Metrics for Item Recommendation". KDD 2020 (best paper).**
  - DOI 10.1145/3394486.3403226 (via Crossref). Also CACM DOI 10.1145/3535335.
  - Fetched version: the IJCAI 2021 extended abstract, https://www.ijcai.org/proceedings/2021/0651.pdf (DOI 10.24963/ijcai.2021/651).
  - Sampled metrics "are inconsistent with their exact counterpart… do not persist relative statements… not even in expectation".
  - With small sample sizes, metrics collapse toward AUC. AUC is the only metric consistent under sampling.
  - Corrections help, but "sampling should be avoided for metric calculation".
- **Hidasi, Czapp. "Widespread Flaws in Offline Evaluation of Recommender Systems". RecSys 2023, DOI 10.1145/3604915.3608839.** https://arxiv.org/abs/2307.14951 (PDF fetched).
  - Negative sampling during testing is listed as one of four widespread flaws. It can change model ordering.

**Why:** Small-store catalogues are around 10^2–10^4 items, so full ranking is cheap. There is no reason to sample.

---

## B3. Statistical testing, multiple comparisons and effect sizes

**Choice:**
- **Unit of analysis:** the user. Use the per-user metric under Successive, averaged within user first.
- **Primary test:** a two-sided **paired t-test** on per-user differences between each candidate and the reference model.
- **Secondary test:** a **paired randomisation (sign-flip permutation) test** with 10^4 or more permutations, for non-mean statistics or very skewed per-user metrics.
- **Interval:** a **95% paired bootstrap CI** (percentile; BCa optional) on the mean difference, with B ≥ 10,000.
- **Do not use** Wilcoxon or the sign test.
- **Multiplicity:**
  - **Holm** (FWER) across the family of candidate-vs-champion comparisons for a ship decision.
  - **Benjamini–Yekutieli** (FDR under dependency) for exploratory many-model leaderboards.
- **Always report** the mean difference, relative change, the CI and the paired effect size d_z.

**Evidence:**
- **Urbano, Lima, Hanjalic. "Statistical Significance Testing in Information Retrieval: An Empirical Analysis of Type I, Type II and Type III Errors". SIGIR 2019, DOI 10.1145/3331184.3331259.** https://arxiv.org/abs/1905.11096 (PDF fetched).
  - Simulation over more than 500 million p-values with 25, 50 and 100 topics.
  - t-test and permutation "maintain errors at the nominal α level remarkably well".
  - The bootstrap-shift test has "a clear bias towards small p-values": 2-tailed Type I error is 0.059 at nominal 0.05.
  - Wilcoxon and sign tests are "unreliable for hypotheses about mean effectiveness, specially with large sample sizes".
  - "We strongly recommend the use of the t-test for typical hypotheses pertaining to mean system effectiveness, and the permutation test for others".
- **Ihemelandu, Ekstrand. "Statistical Inference: The Missing Piece of RecSys Experiment Reliability Discourse". PERSPECTIVES @ RecSys 2021.** https://arxiv.org/abs/2109.06424 (PDF fetched).
  - Of 111 RecSys 2019–20 algorithm papers, **59%** report no significance test, 11% use an unspecified test, 23% name a test, and 7% use CIs.
  - Summarises the conflict between Urbano (t-test) and Parapar (Wilcoxon/sign).
  - Notes that RecSys sample sizes (more than 1,000 users) make tiny effects "significant".
- **Ihemelandu, Ekstrand. "Inference at Scale: Significance Testing for Large Search and Recommendation Experiments". SIGIR 2023 short.** https://arxiv.org/abs/2305.02461 (PDF fetched).
  - On ML-100K, ML-25M and Amazon-Video recommendation runs, Wilcoxon and sign tests show "significantly higher Type-1 error rates for large sample sizes". Bootstrap, randomisation and t-test stay close to nominal.
  - Power differences vanish at large n. "We recommend the sign and Wilcoxon tests should not be used". Effect size should drive judgements of practical importance.
- **Ihemelandu, PhD dissertation, Boise State, 2024. "Best Practices for Offline Evaluation for Top-N Recommendation: Candidate Set Sampling and Statistical Inference".** https://scholarworks.boisestate.edu/td/2238/ (abstract page fetched).
  - Benjamini–Yekutieli "exhibited the lowest error rate and greater power than the Bonferroni test" and is recommended as the default for comparing multiple systems.
  - Holm is not discussed in the abstract.
- **Benjamini, Yekutieli. "The control of the false discovery rate in multiple testing under dependency". Annals of Statistics 29(4), 2001, pp. 1165–1188.** DOI 10.1214/aos/1013699998 (metadata via Crossref only).
- **Sakai. "Statistical Significance, Power, and Sample Sizes: A Systematic Review of SIGIR and TOIS, 2006-2015". SIGIR 2016.** https://static.aminer.org/pdf/20170130/pdfs/sigir/wsby64dh28ioeolu9rzgt5dvmsjmahi3.pdf (fetched).
  - Paired-t effect size is **ês = |t|/√n**, which equals mean(d)/sd(d), i.e. Cohen's d_z. The unpaired version is ês = |t|·√((n1+n2)/(n1·n2)).
  - Researchers "should report p-values and effect sizes"; saying "significant at α=0.05" invites dichotomous thinking.
  - The paired t-test share is rising (Wilcoxon falling). Over half of papers with tests report neither p-values nor test statistics.

**Alternatives:**
- Wilcoxon signed-rank: higher Type I error at large n (Urbano 2019; Ihemelandu 2023).
- Bootstrap *test* (shift method): fine for CIs, anti-conservative for p-values at small n.
- Bonferroni: dominated by Holm, which is uniformly more powerful under the same FWER guarantee.
- Fully Bayesian analysis (Carterette and Sakai recommend it, per Ihemelandu 2021): defer.

**Why:**
- The t-test is calibrated even at n=25 (Urbano).
- Small stores may have only 10^2–10^3 test users, which is exactly where test choice matters.
- At larger n everything is significant, so the CI and effect size carry the decision.

**Caveats:**
- Per-user NBR metrics are bounded and zero-inflated. The t-test is still recommended for means.
- Use the permutation test as a robustness check, and flag disagreements.
- Per-user observations are independent only under the user-level average. Do not treat each Successive basket as an independent sample; cluster by user.

---

## B4. Beyond-accuracy metrics

**Choice:** Report catalogue coverage, novelty (mean self-information), ILD (when item features exist), ARP, APLT, Gini and Shannon entropy of exposure, plus RepR/ExplR (A1). Popularity counts and "long tail" sets are computed from **training data only**.

**Evidence:**
- **Catalogue coverage.** Ge, Delgado-Battenfeld, Jannach. "Beyond accuracy: evaluating recommender systems by coverage and serendipity". RecSys 2010, pp. 257–260, DOI 10.1145/1864708.1864761. https://web-ainf.aau.at/pub/jannach/files/Conference_RecSys2010.pdf (fetched).
  - Catalog coverage = |∪_{j=1..N} I_L^j| / |I| over the N recommendation lists observed.
  - Defines a weighted variant that counts only useful items.
- **Novelty.** Zhou, Kuscsik, Liu, Medo, Wakeling, Zhang. "Solving the apparent diversity-accuracy dilemma of recommender systems". PNAS 107(10), 2010, DOI 10.1073/pnas.1000488107. https://arxiv.org/abs/0808.2670 (fetched).
  - Self-information I_α = log2(u/k_α), where u = #users and k_α = item degree. Mean top-L surprisal I(L) averages over items in the list, then over users.
  - Also defines personalization h(L) = mean over user pairs of 1 − q_ij(L)/L.
- **Intra-list similarity.** Ziegler, McNee, Konstan, Lausen. "Improving recommendation lists through topic diversification". WWW 2005, pp. 22–32, DOI 10.1145/1060745.1060754. https://archives.iw3c2.org/www2005/proceedings/docs/p22.pdf (fetched).
  - ILS(P) = Σ_{b_k∈P} Σ_{b_e∈P, b_k≠b_e} c°(b_k, b_e) / 2, with c° ∈ [−1, 1]. Higher means less diverse. It is permutation-insensitive.
- **Popularity bias.** Abdollahpouri, Burke, Mobasher. "Managing Popularity Bias in Recommender Systems with Personalized Re-ranking". FLAIRS 2019. https://arxiv.org/abs/1901.07555 (fetched).
  - ARP = (1/|U_t|) Σ_u Σ_{i∈L_u} φ(i)/|L_u|, where φ(i) = # training interactions.
  - APLT = (1/|U_t|) Σ_u |L_u ∩ Γ|/|L_u|.
  - ACLT = (1/|U_t|) Σ_u Σ_{i∈L_u} 1(i ∈ Γ).
  - Γ (the long tail) = items outside the short head. The short head is the most popular items that together account for 80% of ratings.
- **Gini, entropy, tail percentage.** RecBole 1.2.1 metric docs: https://recbole.io/docs/recbole/recbole.evaluator.metrics.html (fetched).
  - Gini@K = Σ_{i=1}^{|I|} (2i − |I| − 1)·P(i) / (|I| Σ_i P(i)), with P sorted ascending and P(i) = exposure count of item i in all top-K lists.
  - ShannonEntropy@K = −Σ_i p(i) log p(i).
  - TailPercentage uses a configurable tail ratio, default 0.1.
  - RecBole's NDCG uses IDCG over min(|R(u)|, K).

**Why:** The store owner's levers are catalogue exposure (coverage, Gini) and the "you might also like" vs "buy again" balance (ExplR).

**Caveats:**
- TREx (A3) shows these metrics can be gamed by filling non-repeat slots with diverse but irrelevant items while keeping accuracy. Always pair them with Recall_expl and PHR_expl, which show whether explore slots are ever right.
- Long-tail definitions (80/20 vs 10%) change APLT a lot. Fix one definition in config.

---

## B5. Leakage: what goes wrong and how to test for it

**Choice:** Every benchmark run executes an automated leakage-test suite (the checklist below) and fails closed.

**Evidence:**
- **RecSys 2025 paper, DOI 10.1145/3705328.3759329.**
  - Crossref and OpenAlex metadata: "Don't Get Ahead of Yourself: A Critical Study on Data Leakage in Offline Evaluation of Sequential Recommenders". Authors Huy Hoang Le, Yang Liu, Alan Medlar, Dorota Glowacka. RecSys '25, pp. 1164–1168.
  - OpenAlex marks it closed access. dl.acm.org returned HTTP 403, and no arXiv version was found, so **I could not read its leakage list**.
  - Closest accessible source: Yang Liu's PhD thesis, Univ. Helsinki 2025, https://helda.helsinki.fi/server/api/core/bitstreams/aea49ea6-78e8-4b1f-884c-4b3f9fa99e66/content (fetched). It summarises the same group's manuscript "Flood Warning: Data Leakage in Offline Evaluation of Sequential Recommenders" (Le, Liu, Glowacka, Medlar; then under review). The RecSys 2025 paper is likely, but not confirmed to be, its published form.
  - **Leakage mode in the thesis: temporal leave-one-out.** For a target at time t, other users' interactions after t are used in training.
  - **Proposed fix: "split-by-timepoint LOO".** With a global cut-off, for each "active" user (at least one interaction after the cut-off), the item just before the cut-off is validation and the first item after is test. Later interactions are discarded. "Inactive" users go to train and validation only. The cut-off is chosen to maximise test size.
  - **Findings:** NDCG@10 and recall@10 drop 16.5%–90.0% when switching from temporal LOO to split-by-timepoint LOO across 4 datasets and 9 sequential models.
    - Subsampling training data to equal size does not explain the drop. For example, ML-1M unsampled NDCG@10 falls −90.0%, or −88.9% after size matching.
    - Model rankings flip: the worst model under T-LOO can be the best under ST-LOO.
    - A general (non-sequential) recommender beat the best sequential model on unsampled NDCG on most datasets once leakage was removed.
- **Ji, Sun, Zhang, Li. "A Critical Study on Data Leakage in Recommender System Offline Evaluation". ACM TOIS, DOI 10.1145/3569930.** https://arxiv.org/abs/2010.11060 (v4 fetched). Code: https://github.com/putatu/dataLeakageRec.
  - Defines data leakage as not observing the global timeline.
  - **Random-split-by-ratio, random-split-by-user and leave-one-out all leak.** Only split-by-timepoint does not (Table 2).
  - All four models (BPR, NeuMF, SASRec, LightGCN) recommend "future items", i.e. items released after the test instance's timestamp, among their top-20.
  - Adding leaked future data changes HR@20 and NDCG@20 by between −38.1% and +89.5% (Table 5), making model order "unpredictable".
  - Proposes a "timeline scheme": prequential or sliding-window evaluation that uses only the items available at prediction time.
- **Hidasi & Czapp, RecSys 2023 (B2):** four flaws.
  1. Dataset–task mismatch.
  2. Overzealous preprocessing.
  3. Information leaking through time. With non-time-separated LOO, test transitions shared with train are 0.4868–0.6756 vs 0.3969–0.4500 under time split (Fig. 2b).
  4. Negative sampling.
  - Also flags **timestamp collisions** (day-resolution timestamps plus presorting by item ID create artificial sequences).
- **Sun. "Take a Fresh Look at Recommender Systems from an Evaluation Standpoint". SIGIR 2023, DOI 10.1145/3539618.3591931.** https://arxiv.org/abs/2210.04149 (fetched).
  - 59.1% of the surveyed papers do not observe the global timeline.
  - Their Popularity baseline is "ill-defined": counts over the whole training set include interactions after the test time.
- **Time to Split (B1):** LOO "permits the overlap between training and test periods, which leads to temporal leakage". LTI and UB validation leak into validation, not into test.
- **Reality-check code (A1):** by inspection, the item-frequency filter (≥17) and the G-TopFreq counts use **all** sampled users' history baskets. Under a user split these include baskets dated after other test users' target baskets, which is mild global-timeline leakage. This is acceptable for reproducing their numbers, but must not appear in Mitate's protocol.

**Leakage test checklist** (each is an automated assertion in the benchmark harness):

1. **Global cutoff.** max(timestamp in training interactions) < T_test. With validation, the training max is < T_val during tuning.
2. **Item popularity / global statistics are time-bounded.** Every global statistic is computed only from events < T_test: G-TopFreq counts, TREx RepF, novelty k_α, ARP φ, long-tail Γ, item-frequency filters, k-core filters and ID vocabularies. Test: recompute with test events removed from the raw log and assert identical artefacts.
3. **Preprocessing filters are fit on train only.** Min-count, k-core, deduplication and ID maps are fit on train; test events with unknown items or users are mapped to OOV and not silently dropped. Test: run the pipeline twice, once on the full log and once on the log truncated at T_test, and assert identical training matrices.
4. **Per-target history strictly precedes the target.** For each target basket B_t, the input history contains only baskets with timestamp < ts(B_t). Ties on equal timestamps are treated as the same basket and never as history (Hidasi's collision issue). Test: assert max(history ts) < target ts for every evaluation row.
5. **Repeat/explore labels use the same history.** The repeat set for B_t is ∪ of the user's baskets strictly before B_t. It never includes B_t or later baskets. Test: recompute repeat labels independently and assert equality.
6. **Model state frozen at T_test.** Under Successive targets the model may consume the user's post-cutoff baskets as *input*, but parameters and neighbour pools must not be refit on them. This covers TIFU-KNN neighbour vectors, popularity and TREx item features. Test: hash model artefacts before and after test inference.
7. **kNN pool hygiene.** The target user is never their own neighbour, and neighbour vectors are built only from data < T_test (TIFU-KNN's reference code uses train users only). Test: assert that the neighbour index set excludes the query user.
8. **No future items.** Recommended items must exist in the catalogue (first-seen or published date ≤ target time), per Ji et al. Test: count recommended items whose first appearance is > target time. It must be 0, or they are masked.
9. **Tuning never sees test.** Hyperparameter search, early stopping and model selection use only validation (< T_test). The test set is touched once per final configuration. Test: the harness raises if the test loader is opened before final retraining, and the log records every access.
10. **Retrain-on-train+val is explicit.** The final model is refit on all data < T_test with the selected hyperparameters, and the configuration hash is recorded.
11. **No sampled negatives** in metrics (B2). Test: assert the candidate set equals the full purchasable catalogue at target time.
12. **Basket construction.** One order maps to one basket. Orders are not split across the cutoff (assign by order timestamp). Returns or cancellations recorded after T_test do not alter training baskets. Test: assert every order_id lies entirely on one side of the cutoff.
13. **User and ID leakage across splits** (for user-split reproduction mode only). Train, val and test user sets are disjoint. Keysets are persisted, because the reference code shuffles without a seed.
14. **Duplicate and bot traffic.** Identical (user, ts, item) rows are collapsed before splitting, so a duplicate cannot sit in both train and test. Test: assert there are no (user, item, ts) rows shared between train and test.
15. **Feature leakage (if side features exist).** Prices, stock and category popularity are as-of the target time, not the latest snapshot.

**Alternatives:** Prequential or sliding-window evaluation (Ji et al. timeline scheme). It is more faithful and k× more expensive. Optional for small data, where retraining is cheap.

---

## B6. Hyperparameter-budget fairness

**Choice:**
- Every model, including baselines, gets the **same number of Optuna TPE trials**: 50 by default, with no smaller budget for cheap models.
- Use the same search-space documentation, the same validation metric (N@10 on GT validation) and the same seeds.
- Retrain the best configuration on train+val.
- Parameter-free baselines (G/P/GP-TopFreq) are reported as is.
- Store the full trial history.

**Evidence:**
- **Rendle, Zhang, Koren. "On the Difficulty of Evaluating Baselines: A Study on Recommender Systems". arXiv 1905.01395 (2019).** https://arxiv.org/abs/1905.01395 (fetched).
  - On ML10M, a carefully tuned vanilla MF baseline beat both its own previously reported results and every newly proposed method. Reported baselines included RSVD 0.8256 RMSE and BPMF 0.8197. Their tuned Bayesian MF reached 0.7633, and their best combined model 0.7485.
  - Findings "are questionable unless they were obtained on standardized benchmarks where baselines have been tuned extensively".
- **Shehzad, Jannach. "Everyone's a Winner! On Hyperparameter Tuning of Recommendation Models". RecSys 2023, DOI 10.1145/3604915.3609488.** https://web-ainf.aau.at/pub/jannach/files/Conference_RecSys_2023_Repro.pdf (fetched).
  - With under-tuned baselines, every one of seven recent deep models can be shown to "win".
  - Baseline tuning is "barely documented". Their own runs used 10–50 tuning iterations per model.
- **Ferrari Dacrema, Cremonesi, Jannach. RecSys 2019.** https://arxiv.org/abs/1907.06902 (fetched). Baselines were tuned by Bayesian search (scikit-optimize), **35 cases per algorithm, the first 5 random**.
- **Akiba et al. "Optuna: A Next-generation Hyperparameter Optimization Framework". KDD 2019.** https://arxiv.org/abs/1907.10902 (fetched). It supports TPE and pruning.
  - Current `TPESampler` docs (https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html, fetched): `n_startup_trials=10`, `n_ei_candidates=24`, and `multivariate` auto-selects True for single-objective. With `n_jobs>1`, replicas "may suggest the same values" unless reseeded.
- SAFERec (A3) uses 50 Optuna trials per model.
- Fang et al., arXiv 2408.07630 (abstract fetched), compare seven HPO algorithms for top-N but name no winner in the abstract.

**Why:** Unequal tuning is the most common way simple baselines "lose". A fixed trial budget makes the comparison falsifiable.

**Caveats:**
- TPE is stochastic. Fix the sampler seed and run sequentially (or reseed per worker) for reproducibility.
- Equal trials is not equal compute. Also log wall-clock time.

---

## B7. Online measurement for the experiment module

**Choice:**
- **Default online test: team-draft interleaving** of the two rankers on the same widget. Credit = clicks or add-to-carts on each team's items. Aggregate with Chapelle's Δ_AB and test per-user preference shares.
- **Use A/B only** for business KPIs (revenue/visitor, conversion). Apply **CUPED** (pre-period same-metric covariate, 1–2 weeks), **capped/winsorised revenue**, and an **always-valid / sequential** analysis if the owner will peek.
- Show the owner a pre-launch **power calculator** that states honestly when an A/B test is infeasible for their traffic.

**Evidence:**
- **Radlinski, Kurup, Joachims. "How Does Clickthrough Data Reflect Retrieval Quality?" CIKM 2008, pp. 43–52, DOI 10.1145/1458082.1458092.** https://cs.cornell.edu/~tj/publications/radlinski_etal_08b.pdf (fetched).
  - Introduces Team-Draft Interleaving (Alg. 2). Each round, the team with fewer picks (or the RandBit winner on a tie) appends its highest-ranked item not yet in I. Clicks are credited to the item's team: h_a and h_b. If h_a > h_b, the user prefers A.
  - None of 8 absolute metrics reliably reflected quality, while both interleaving tests did.
  - A randomly clicking user yields equal preference in expectation. Balanced interleaving does not have that property.
- **Chapelle, Joachims, Radlinski, Yue. "Large-Scale Validation and Analysis of Interleaved Search Evaluation". ACM TOIS 30(1), Article 6, 2012, DOI 10.1145/2094072.2094078.** https://www.cs.cornell.edu/people/tj/publications/chapelle_etal_12a.pdf (fetched).
  - Δ_AB = (wins(A) + ½·ties(A,B)) / (wins(A) + wins(B) + ties(A,B)) − 0.5. Queries without clicks are ignored.
  - Interleaving reduced the number of queries needed by "over an order of magnitude" vs Clicks@1 in all but one pair. Absolute metrics "typically need orders of magnitude more data".
  - Attributed to the fact that interleaving "is a paired test, paired on both queries and users".
- **Bi, Castells, Gilbert, Galperin, Tardif, Ahuja. "Debiased Balanced Interleaving at Amazon Search". CIKM 2022.** https://cdn.amazon.science/a9/c8/c9016a1c47caac6a634768e7491d/debiased-balanced-interleaving-at-amazon-search.pdf (fetched). Reports 10 online e-commerce experiments with a **60× sensitivity gain over A/B**.
- **Bi, Li, Gao, Edge, Ahuja. "Interleaved Online Testing in Large-Scale Systems". WWW '23 Companion.** https://cdn.amazon.science/c1/4d/7945330e47539fdd870cb5c73613/interleaved-online-testing-in-large-scale-systems.pdf (fetched).
  - Cites a "sensitivity boost ranging from 10 to 100 times".
  - Adds multi-system ordering with false-discovery control and A/B-effect-size estimation from interleaving.
- **Zhang, Deng, Du, Gao, He, Katariya et al. "Harnessing the Power of Interleaving and Counterfactual Evaluation for Airbnb Search Ranking". KDD 2025.** https://arxiv.org/abs/2508.00751 (fetched).
  - Uses competitive-pair team-draft: one coin flip, a pair from each team per round, identical items unassigned. It gave **50× speedup** vs A/B in production. Counterfactual evaluation gave up to **100×**.
  - **Limitations** include set-level optimisation (diversity), shared surfaces, and "not straightforward" continuous outcomes like revenue.
  - Case study: interleaving showed a significant negative result while the A/B conversion result was neutral.
- **Deng, Xu, Kohavi, Walker. "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data" (CUPED). WSDM 2013, pp. 123–132, DOI 10.1145/2433396.2433413.** https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf (fetched).
  - Ŷ_cv = Ȳ − θ(X̄ − E[X]), with θ = cov(Y,X)/var(X). The variance factor is (1 − ρ²).
  - About 50% variance reduction at Bing.
  - The same metric from the pre-period is the best covariate. A 1–2 week pre-period works well. Never use covariates the treatment can affect.
  - Users without pre-period data get a missing-indicator covariate, equivalent to stratification.
- **Kohavi, Deng, Longbotham, Xu. "Seven Rules of Thumb for Web Site Experimenters". KDD 2014.** https://exp-platform.com/Documents/2014%20experimentersRulesOfThumb.pdf (fetched).
  - For skewed metrics, the minimum n per variant for the mean to be approximately normal is **355·s²**, where s = skewness (use when |s| > 1).
  - At Bing, Revenue/User had |s| = 17.9, needing 114k users and detecting a 4.4% change at 80% power. Capped revenue had |s| = 5.2, needing 9.7k users at 10.5%.
  - "At a commerce site, the skewness for purchases/customer was >10 and for revenue/customer >30". Capping at $10/user/week cut skewness from 18 to 5.3 and detected changes 30% smaller.
- **Johari, Pekelis, Walsh. "Always Valid Inference" (arXiv title: "Always Valid Inference: Bringing Sequential Analysis to A/B Testing"; PDF title: "… Continuous Monitoring of A/B Tests").** https://arxiv.org/abs/1512.04922 (fetched).
  - Fixed-horizon p-values are "wholly unreliable" under continuous monitoring. Always-valid p-values and CIs (mixture SPRT) fix this.
  - Implemented in a commercial A/B platform.

**Alternatives:**
- A/B only: needs about 10–100× more traffic for ranking comparisons, per the evidence above.
- Counterfactual / off-policy evaluation from logs: needs logged propensities. Airbnb's counterfactual approach needs a serving-side counterfactual ranking. Defer.
- Multileaving for more than 2 rankers: not reviewed here.

**Why:**
- Small stores have little traffic. Illustratively, with RPV coefficient of variation σ/μ = 10 (plausible given skewness above 10; not measured here), detecting a 5% RPV lift at α=0.05 and power 0.8 needs n ≈ 15.7·(σ/Δ)² = 15.7·(10/0.05)² ≈ 628,000 visitors per arm.
- Interleaving's paired design is the only realistic way such stores get a significant answer about ranker quality within weeks.

**Caveats:**
- Interleaving measures engagement preference on a single surface, not revenue (Airbnb limitation).
- It is biased against rankers that deliberately trade conversion for diversity (Airbnb case study). It does not suit slot-structured basket widgets where both rankers mostly show the same repeat items: identical items carry no credit, so expect many ties.
- Keep A/B (with CUPED and capping) for the final business decision.

---

## Exact definitions (to implement)

Notation:
- U = evaluated users. For user u: target basket T_u (a set), ranked prediction P_u = (p_1, …, p_K), history H_u = ∪ of baskets strictly before T_u.
- T_u^rep = T_u ∩ H_u and T_u^expl = T_u \ H_u.
- I = catalogue purchasable at target time. φ(i) = training interaction count. k_i = #training users who bought i. N_U = #training users.
- Under Successive targets, compute every per-user metric as the mean over that user's targets, then average across users.

**Accuracy (per user, then mean over U):**
- `recall@K = |P_u[:K] ∩ T_u| / |T_u|`
- `precision@K = |P_u[:K] ∩ T_u| / K`
- `hit/PHR@K = 1[|P_u[:K] ∩ T_u| > 0]`
- `dcg@K = Σ_{r=1..K} 1[p_r ∈ T_u] / log2(r+1)`
- `ndcg@K = dcg@K / Σ_{r=1..min(K,|T_u|)} 1/log2(r+1)` is the standard form (Li et al. Eq. 2, RecBole).
- `ndcg_li2023@K = dcg@K / Σ_{r=1..|T_u|} 1/log2(r+1)` reproduces the reality-check code and tables only.
- `mrr@K = 1/rank of first hit within K, else 0`

**Repeat/explore (Li et al. 2023):**
- `RepR@K = mean_u |P_u[:K] ∩ H_u| / K` and `ExplR@K = 1 − RepR@K`. If a model returns fewer than K items, use |P_u| in place of K and log that separately.
- `Recall_rep@K = mean over U_r of |P_u[:K] ∩ T_u^rep| / |T_u^rep|`, where U_r = {u : |T_u^rep| > 0}.
- `PHR_rep@K = mean over U_r of 1[|P_u[:K] ∩ T_u^rep| > 0]`.
- `Recall_expl@K` and `PHR_expl@K` are defined the same way with T_u^expl over U_e = {u : |T_u^expl| > 0}.
- Always report |U_r| and |U_e|.

**Beyond-accuracy (training-only statistics):**
- `coverage@K = |∪_u P_u[:K]| / |I|` (Ge et al. 2010)
- `novelty@K = mean_u mean_{i∈P_u[:K]} log2(N_U / k_i)` (Zhou et al. 2010). Items with k_i = 0 are cold; use log2(N_U / 1) and count them separately.
- `ILD@K = mean_u [ 2/(K(K−1)) Σ_{a<b} (1 − sim(i_a, i_b)) ]`. This is our normalised distance form of Ziegler's ILS = Σ_{a≠b} sim/2. State which similarity is used: cosine on category or text embeddings.
- `ARP@K = mean_u mean_{i∈P_u[:K]} φ(i)`
- `APLT@K = mean_u |P_u[:K] ∩ Γ| / K`, where Γ = items outside the short head. The short head is the smallest set of most-popular items covering 80% of training interactions (Abdollahpouri et al. 2019).
- `Gini@K = Σ_{j=1..|I|} (2j − |I| − 1)·c_(j) / (|I|·Σ_j c_(j))`, with c_(j) the exposure counts sorted ascending and items with zero exposure included (RecBole).
- `Entropy@K = −Σ_i p_i log p_i`, with p_i = c_i / Σ c.
- `Personalization@K = mean over user pairs of 1 − |P_u[:K] ∩ P_v[:K]|/K` (Zhou et al.). Sample pairs if |U| is large.

**Statistics (candidate B vs reference A; d_u = m_B(u) − m_A(u), n = |U|):**
- Paired t: `t = mean(d) / (sd(d)/√n)`, df = n−1, two-sided p.
- Effect size: `d_z = mean(d)/sd(d) = t/√n` (Sakai 2016). Also report the relative change `mean(d)/mean(m_A)`.
- Paired bootstrap CI:
  - Resample users with replacement B = 10,000 times. Recompute mean(d*).
  - The 95% CI is the [2.5, 97.5] percentiles. Use a fixed RNG seed.
  - Same users for both systems, i.e. resample the pairs.
- Paired randomisation test:
  - For r = 1..R (R ≥ 10,000), draw s_u ∈ {±1} and compute `mean(s_u·d_u)`.
  - `p = (1 + #{|stat_r| ≥ |stat_obs|}) / (R + 1)`.
- Holm (FWER), for m comparisons:
  - Sort p_(1) ≤ … ≤ p_(m).
  - Reject H_(1..k), where k is the largest index with p_(j) ≤ α/(m − j + 1) for all j ≤ k.
  - Adjusted `p̃_(j) = max_{l≤j} min(1, (m − l + 1)·p_(l))`.
- Benjamini–Yekutieli (FDR under arbitrary dependence):
  - c(m) = Σ_{i=1..m} 1/i.
  - Reject H_(1..k) for the largest k with `p_(k) ≤ k·α / (m·c(m))`.
- Kendall τ between model rankings from two protocols, for split-sensitivity reports (Meng et al.; Time to Split).

**Online:**
- Team-draft interleaving, per impression: run Radlinski Alg. 2. h_A = clicks on TeamA items and h_B = clicks on TeamB items. Outcome is win A / win B / tie (impressions with no clicks are ignored).
- `Δ_AB = (wins_A + ½ ties) / (wins_A + wins_B + ties) − 0.5` (Chapelle et al. 2012).
- Test: aggregate per user to the share of A-wins among decided impressions, then a one-sample t or sign-flip test against 0.5. This is the user-clustered version; impression-level binomial tests overstate n.
- CUPED: `θ = cov(Y, X)/var(X)`, pooled over both arms. `Y_cv = Y − θ(X − mean(X))`. Compare arm means of Y_cv. The variance factor is 1 − ρ². Add a missing-pre-period indicator.
- Sample size per arm, two-sided, normal approximation: `n = 2·(z_{1−α/2} + z_{1−β})²·σ²/Δ²`, which is ≈ 15.7·σ²/Δ² at α = .05 and power .8 (standard derivation). Also require `n ≥ 355·s²` when |skew| s > 1 (Kohavi et al. 2014).
- Capping: winsorise per-user revenue at a pre-registered quantile of the pre-period distribution, e.g. the 99th percentile.

---

## Recommendations for Mitate

1. **Protocol of record:**
   - Global temporal split. T_test is chosen so that at least N_min test users have a post-cutoff basket. T_val is placed before it with the same rule.
   - Successive basket targets, user-averaged, with Last also reported.
   - Hyperparameters selected on GT-validation, then the model is retrained on everything < T_test.
   - Full-catalogue ranking. Report the split statistics (users, baskets, period lengths) every run.
2. **Pipeline-validation mode:** a separate "reality-check" mode that loads the liming-7 preprocessed Instacart and reproduces all-user GP-TopFreq R@10 0.3304, `ndcg_li2023@10` 0.3417, PHR@10 0.8422 and G-TopFreq Recall_rep@10 0.1002. These are exact and deterministic. Optionally also reproduce TIFU-KNN under their 72/8/20 user split with the README hyperparameters. Expect R@10 ≈ 0.360 ± 0.002 (5-fold std 0.0015). The keysets are unseeded, so match the mean, not individual folds.
3. **Baseline ladder:**
   - G-TopFreq → P-TopFreq → **GP-TopFreq (floor)** → **TIFU-KNN** → **TREx-Rep** (+ GP-style fill for explore slots).
   - TAIW or TIFU-KNN-TA once timestamps and the prediction time are reliable.
   - Any neural model must beat TIFU-KNN under the same budget and split.
4. **Metrics in every report:**
   - R/N/PHR @10 and @20.
   - Recall_rep, Recall_expl, PHR_rep, PHR_expl, RepR, with |U_r| and |U_e|.
   - Coverage, novelty, ARP, APLT, Gini.
   - ILD only where item features exist.
5. **Statistics:**
   - Per-user paired t-test as primary, sign-flip permutation as a check, and a paired bootstrap 95% CI with d_z.
   - Holm across the candidate-vs-champion family, and BY for exploratory leaderboards.
   - No Wilcoxon or sign tests.
   - Ship decisions require the CI lower bound > a pre-registered minimum effect, not just p < .05.
6. **Leakage suite:** implement the 15 checks in B5 as tests that run in CI on a synthetic log with planted leaks. Each check must fail on its planted leak and pass on clean data.
7. **Tuning:** Optuna TPE, an equal 50-trial budget per tunable model, a fixed seed, sequential execution, and the trial log persisted. TIFU-KNN's k range is scaled to |train users|.
8. **Online module:**
   - Team-draft interleaving (competitive-pair variant) for ranker selection on browse/recommendation widgets.
   - A/B with CUPED, capped revenue and always-valid sequential p-values for business sign-off.
   - The power calculator refuses to launch an A/B test whose required n exceeds roughly 8 weeks of store traffic, and suggests interleaving or a proxy metric instead.

---

## Could not verify

- **RecSys 2025 leakage paper (DOI 10.1145/3705328.3759329):**
  - Title and authors were confirmed via Crossref and OpenAlex. The full text is closed: dl.acm.org returned 403, and there is no arXiv copy. **Its specific list of leakage modes was not read.**
  - The B5 checklist is built from Ji et al., Hidasi & Czapp, Sun, Time to Split, and the Liu thesis's summary of the same group's "Flood Warning" manuscript.
  - It is unconfirmed that the RecSys paper equals that manuscript.
- **Krichene & Rendle KDD 2020 full paper:** only the IJCAI 2021 extended abstract was read. The KDD DOI comes from Crossref.
- **ReCANet full text:** the staff.fnwi.uva.nl PDF host did not resolve (ENOTFOUND). Only the UvA DARE abstract page was read. The DOI 10.1145/3477495.3531708 was confirmed via Crossref. One search result's DOI (…3531940) is wrong: Crossref says it belongs to "AutoGSR".
- **TIFU-KNN original repo** (github.com/HaojiHu/TIFUKNN) was not cloned. The algorithm details come from the paper and the copy vendored in the reality-check repo.
- **TREx's exact definition of "repurchase frequency"** in RepF: the paper text is underspecified. Check https://github.com/lynEcho/TREX (not fetched).
- **UP-CF@r (Faggioli et al., UMAP 2020), TIFU-KNN-TA (Naumov et al., ECIR 2023), SLRC (WWW 2019):** cited by fetched papers, but not fetched themselves.
- **Holm (1979), "A simple sequentially rejective multiple test procedure":** JSTOR did not load and Crossref did not return the record. The procedure in "Exact definitions" is the standard one, but the citation details are unverified.
- **Sakai, "Statistical reform in information retrieval?" (SIGIR Forum 2014):** not found. The Sakai SIGIR 2016 review was used instead.
- **Netflix tech blog posts on interleaving:** both returned 403. No Netflix numbers are used.
- **Kohavi et al. 2009, "Controlled experiments on the web: survey and practical guide":** the PDF URL returned HTML. The 15.7·σ²/Δ² sample-size formula is therefore given as a standard derivation, not a citation.
- **Instacart schema:** confirmed only via secondary sources. The Kaggle data page requires login.
- **Optuna `create_study` default sampler:** not stated on the fetched TPESampler page. Set `sampler=TPESampler(seed=…)` explicitly.
- **Fang et al. (arXiv 2408.07630):** the abstract does not name the best HPO algorithm, so no conclusion is drawn from it.
- **"Interleaving 10–100× more sensitive than A/B for small-traffic stores":** no fetched source studies small stores specifically. The 10–100×, 50× and 60× figures all come from large platforms (Amazon, Airbnb, Yahoo/Bing-scale search). How well they transfer to low-traffic stores with sparse clicks is unverified.
