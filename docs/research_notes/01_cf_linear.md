# 01: CF, item-item, linear-autoencoder and MF baselines

Scope: collaborative-filtering baselines for Mitate (small stores: 10^2 to ~3x10^4 items, ~10^3 to ~10^6 interactions).
Review date: 2026-10-19. Every number below was read from the URL listed with it. Anything I could not open is listed under "Could not verify".

Notation: R@k = Recall@k, N@k = NDCG@k.
- "Strong generalization" (SG) means train, validation and test users are disjoint. A held-out user's history is folded in at inference time.
- "Weak generalization" (WG) means each user's interactions are split, so test users also appear in training.

---

## 0. Reproducibility context (why these baselines)

**Choice:** Mitate's default comparison set is tuned EASE, RP3beta, ItemKNN, iALS and popularity. A neural model is accepted only if it beats this set under the same split and the same tuning budget.

**Evidence:**
- **Ferrari Dacrema, Cremonesi, Jannach. "Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches". RecSys 2019.** https://arxiv.org/abs/1907.06902
  - 18 top-n algorithms from top venues; only 7 were reproducible "with reasonable effort".
  - 6 of those 7 were often beaten by simple heuristics (nearest-neighbour or graph-based).
  - The 7th did not consistently beat a well-tuned non-neural linear ranking method.
  - Extended journal version: "A Troubling Analysis of Reproducibility and Progress in Recommender Systems Research", ACM TOIS, DOI 10.1145/3434185, issued 2021-01. Metadata only, via https://api.crossref.org.
- **Rendle, Krichene, Zhang, Anderson. "Neural Collaborative Filtering vs. Matrix Factorization Revisited". RecSys 2020.** https://arxiv.org/abs/2005.09683 and https://research.google/pubs/neural-collaborative-filtering-vs-matrix-factorization-revisited/
  - With proper hyperparameter selection, a dot product "substantially outperforms" NCF's learned MLP similarity.
  - Learning a dot product with an MLP is non-trivial.
  - MLP scoring is too costly for production retrieval.
- **Anelli, Bellogín, Di Noia, Jannach, Pomo. "Top-N Recommendation Algorithms: A Quest for the State-of-the-Art". UMAP 2022.** https://arxiv.org/pdf/2203.01155v2
  - Setup: 10 algorithms, TPE (HyperOpt) with 20–50 trials each, tuned on validation nDCG@10, repeated 80/20 hold-out.
  - Datasets of small-store size: Amazon Digital Music 5-core (14,354 users / 10,027 items / 145,523 interactions) and Epinions (8,485 users / 8,463 items / 300k interactions). ML-1M was also used.
  - Top three on every metric came from EASE, MF2020, SLIM, RP3beta and UserKNN, "with EASE always taking a top rank". Neural models (NeuMF, MultVAE) were never best.

  | Dataset | Metric | RP3beta | EASE | SLIM | UserKNN | iALS | ItemKNN | MultVAE |
  |---|---|---|---|---|---|---|---|---|
  | AMZm | nDCG@10 | 0.085 | 0.083 | 0.081 | 0.081 | 0.073 | 0.071 | 0.054 |
  | ML-1M | nDCG@10 | 0.315 | **0.336** | 0.335 | 0.315 | 0.306 | 0.292 | 0.294 |
  | Epinions | nDCG@10 | 0.163 | 0.164 | 0.156 | 0.164 | 0.110 | 0.138 | 0.149 |

- **Shevchenko et al. "From Variability to Stability: Advancing RecSys Benchmarking Practices". KDD 2024.** https://arxiv.org/html/2402.09766v2
  - Setup: 11 algorithms (incl. ItemKNN, ALS, BPR, SLIM, EASE, MultiVAE, LightGCN) on 30 datasets.
  - Global temporal 80/10/10 split. Optuna TPE on nDCG@10, then retrained on train plus validation.
  - Quote: "EASE is a clear winner with respect to all considered aggregation strategies"; "no distinct second-place algorithm".
  - The paper also flags "the significant computational demands of EASE".
- **Benigni, Ferrari Dacrema, Jannach. "Diffusion Recommender Models and the Illusion of Progress". ACM TORS 2026.** https://arxiv.org/pdf/2505.09364
  - Only 25% of the reported results of 9 SIGIR'23/'24 diffusion recommenders were fully reproducible.
  - Tuned simple baselines beat them. On Amazon-Books (clean), ItemKNN R@20 = 0.1525, SLIM 0.1561 and RP3beta 0.1493, against the best diffusion variant at 0.1188.
  - iALS was weak there (0.0983).
  - EASE could not run on that dataset because it needed more than 64 GB of memory.
- **Said, Bellogín. "Reproducibility in Recommender Systems: A Survey". Accepted at ACM TORS, arXiv July 2026.** https://arxiv.org/abs/2607.26074
  - Covers 51 RecSys reproducibility-track papers from 2020–2025.
  - Finds a narrow set of datasets and protocols and "a gap between the conceptual definition of reproducibility and its implementation".
  - It names no winning baseline.

**Why it matters for small stores:** the small, sparse datasets in these studies (AMZm, Epinions, Amazon-Books) are where neural models do worst and neighbourhood or linear models do best.

**Caveat:** the rankings depend on the protocol.
- iALS ranks medium in generic-TPE studies (Quest, Diffusion) but near the top when tuned by experts (Rendle 2022, §2).
- UserKNN beat ItemKNN in the Quest study.

---

## 1. EASE (primary model)

**Choice:** EASE (Steck 2019), closed form `B = I − P·diagMat(1/diag(P))` with `P = (XᵀX + λI)⁻¹`. It is a dense n×n solve with one hyperparameter, λ.

**Evidence:** **Steck. "Embarrassingly Shallow Autoencoders for Sparse Data". WWW 2019.** https://arxiv.org/pdf/1905.03375 (Table 1, read from the PDF)

Protocol: the Liang et al. 2018 Mult-VAE setup ("results reproduced from [13]"), strong generalization, metrics R@20, R@50, N@100. Standard errors are about 0.002 (ML-20M), 0.001 (Netflix) and 0.001 (MSD).

| Dataset (users / items / interactions) | Model | R@20 | R@50 | N@100 |
|---|---|---|---|---|
| ML-20M (136,677 / 20,108 / ~10M) | EASE | 0.391 | 0.521 | 0.420 |
| | EASE with negative weights clipped (≥0) | 0.373 | 0.499 | 0.402 |
| | Mult-VAE | 0.395 | 0.537 | 0.426 |
| | SLIM | 0.370 | 0.495 | 0.401 |
| | WMF | 0.360 | 0.498 | 0.386 |
| | popularity | 0.162 | 0.235 | 0.191 |
| Netflix (463,435 / 17,769 / ~57M) | EASE | **0.362** | **0.445** | **0.393** |
| | Mult-VAE | 0.351 | 0.444 | 0.386 |
| | SLIM | 0.347 | 0.428 | 0.379 |
| | WMF | 0.316 | 0.404 | 0.351 |
| MSD (571,355 / 41,140 / ~34M) | EASE | **0.333** | **0.428** | **0.389** |
| | Mult-VAE | 0.266 | 0.364 | 0.316 |
| | WMF | 0.211 | 0.312 | 0.257 |
| | SLIM | did not finish | | |

Other points from the paper:
- Best λ was about 500 (ML-20M), 1,000 (Netflix) and 200 (MSD). On the smaller ML-10M weak-generalization experiment, λ = 3000.
- About 60% of the learned weights are negative. Clipping them to zero drops accuracy to roughly SLIM's level.
- Training time on an AWS instance with 64 GB RAM and 16 vCPUs: under 2 minutes on Netflix and under 20 minutes on MSD.

**Exact reproduction protocol (pipeline-validation target).** Verified in Liang's code, https://raw.githubusercontent.com/dawenl/vae_cf/master/VAE_ML20M_WWW2018.ipynb (Apache-2.0):
- Binarize with `rating > 3.5`.
- Keep users with at least 5 interactions (`min_uc=5`, `min_sc=0`).
- Shuffle users with `np.random.seed(98765)`, then hold out 10,000 validation and 10,000 test users. MSD holds out 50,000 users per the iALS paper.
- Drop validation and test items not seen in training.
- For each held-out user, 20% of interactions become targets (`test_prop=0.2`, seed 98765, only users with at least 5 items). The other 80% are the fold-in input.
- Metrics:
  - Recall@k = hits / **min(k, |test items|)**. This is truncated recall, not plain recall.
  - NDCG@k uses binary relevance, with IDCG computed over min(n, k).
- Expected result: EASE ≈ 0.391 / 0.521 / 0.420 on ML-20M. ML-20M is the cheapest of the three to reproduce: n = 20,108 items, so G is about 1.6 GB in float32 or 3.2 GB in float64.

**Alternatives considered:**
- SLIM: slower, and worse in the table above.
- Mult-VAE: a near-tie on ML-20M, clearly worse on MSD and Netflix.
- Newer LAE variants: see §3.

**Why it wins for small stores:**
- It is the most consistent top performer across independent studies (Quest, KDD'24, Diffusion paper).
- It has a single hyperparameter.
- It is deterministic.
- For n ≤ 30k items, the n×n matrix stays within a few GB. For n ≤ 10k items it is at most about 400 MB in float32 and trains in seconds.

**Caveats:**
- Memory is O(n²) and the solve is O(n³). EASE runs out of memory on large catalogs (over 64 GB on Amazon-Books in the Diffusion paper).
- It has a strong popularity bias (see DAN, §3).
- It has no cold-start for new items: they need side information (see L3AE, §3).

---

## 2. iALS / MF (second family; implicit library or irspack)

**Choice:** iALS with Rendle's tuning recipe: unobserved weight α₀, frequency-scaled regularization ν = 1, and large embedding dimension d. Use it as the MF baseline and for low-latency serving through a user embedding.

**Evidence:** **Rendle, Krichene, Zhang, Koren. "Revisiting the Performance of iALS on Item Recommendation Benchmarks". RecSys 2022, pp. 427–435.** https://arxiv.org/pdf/2110.14037 (PDF read). Venue confirmed at https://research.google/pubs/revisiting-the-performance-of-ials-on-item-recommendation-benchmarks/

Protocol:
- Same Liang et al. splits: hold out 10,000 users (ML-20M) and 50,000 (MSD), fold in 80% of each held-out user's history, evaluate on the other 20%.
- Tuned on the validation holdout. Mean of 10 repeats.
- No training on evaluation users. Their embeddings come from a closed-form least-squares solve at evaluation time (8 block sweeps with the block solver).

| Benchmark | λ | α₀ | epochs T | d | iALS R@20 | R@50 | N@100 | Previous iALS (Liang 2018) | EASE |
|---|---|---|---|---|---|---|---|---|---|
| ML-20M | 0.003 | 0.1 | 16 | 2048 | 0.395 | 0.532 | 0.425 | 0.360 / 0.498 / 0.386 | 0.391 / 0.521 / 0.420 |
| MSD | 0.002 | 0.02 | 16 | 8192 | 0.309 | 0.415 | 0.368 | 0.211 / 0.312 / 0.257 | **0.333 / 0.428 / 0.389** |

Other results and recipe points:
- On ML1M and Pinterest (sampled metrics) with d capped at 192: λ = 0.007 / 0.02, α₀ = 0.3 / 0.007.
- With ν = 1, quality keeps improving as d grows. With ν = 0 it plateaus.
- d = 512 already beats EASE on ML-20M.
- Only EASE beats iALS on MSD. RecVAE, H+Vamp and RaCT beat it on ML-20M.

**Alternatives:**
- **BPR-MF.** "Revisiting BPR: A Replicability Study of a Common Recommender System Baseline", Milogradskii et al., RecSys 2024. https://arxiv.org/html/2409.14217v2
  - Tuned BPR reached N@100 of 0.4012 (ML-20M) and 0.3289 (MSD), against EASE at 0.4215 and 0.3907 (their runs).
  - Library implementations were much worse:

    | Library | ML-20M N@100 | MSD N@100 |
    |---|---|---|
    | Cornac | 0.3691 | 0.3114 |
    | implicit | 0.2880 | 0.2567 |
    | LightFM | 0.1843 | 0.0575 |
    | RecBole | 0.1785 | 0.0117 |

  - What matters for BPR: plain SGD beat Adam and RMSProp; adaptive negative sampling helps; item-bias usefulness depends on the dataset; d = 1024.
  - On a global time split of ML-20M, tuned BPR (0.1730) ≈ Mult-VAE (0.1722) ≈ EASE (0.1698), and the lead is not significant.
  - Conclusion: BPR is not worth carrying unless we implement it ourselves. Do not use implicit's BPR as a baseline.
- **PureSVD / SVD-AE:** see §3.

**Why it wins for small stores:**
- Training cost is linear in interactions.
- It produces compact user and item vectors for ANN serving and cold users (fold-in).
- It is a different inductive bias from item-item models, which makes it useful in an ensemble or as a check.

**Caveats:**
- It is very sensitive to tuning. With generic TPE tuning it was mid-pack (Quest: AMZm nDCG@10 0.073 vs EASE 0.083) or poor (Diffusion paper on Amazon-Books: R@20 0.0983 vs ItemKNN 0.1525).
- The tuning path must use log grids over (λ, α₀) and double d until quality plateaus.
- **implicit's ALS has no ν or α₀.** Its parameters are `factors=100, regularization=0.01, alpha=1.0, iterations=15, …` (https://benfred.github.io/implicit/api/models/cpu/als.html), so Rendle's recipe is not directly available there.
- **irspack's `IALSRecommender` implements α₀, reg and ν (default ν = 1) and cites the Rendle paper** in its source: https://raw.githubusercontent.com/tohtsky/irspack/main/src/irspack/recommenders/ials.py

---

## 3. Newer linear / closed-form models (2020–2026): which, if any, to add

**Choice:**
- Add EDLAE/DLAE (a one-line change to EASE) and DAN normalization (two exponents) as tunable variants of the EASE solver.
- Do not add the graph-filter family (GF-CF, Turbo-CF, ChebyCF, BSPM) or SVD-AE.

**Evidence:**
- **EDLAE.** Steck, "Autoencoders that don't overfit towards the Identity", NeurIPS 2020. https://proceedings.neurips.cc/paper_files/paper/2020/file/e33d974aae13e4d877477d51d8bafdc4-Paper.pdf (Table 1, PDF read)
  - Low-rank (k = 1000) EDLAE on the Liang protocol, R@20 / R@50 / N@100:
    - ML-20M: 0.389 / 0.518 / 0.420
    - Netflix: 0.362 / 0.446 / 0.394
    - MSD: 0.263 / 0.354 / 0.320
  - This is roughly EASE level on ML-20M and Netflix. Low-rank is below full-rank EASE on MSD.
  - The closed form is EASE with item-specific L2 `Λ = p/(1−p)·diag(XᵀX) + λ`. irspack implements exactly this in `edlae.py`.
- **DAN.** Park, Yoon, Kim, Lee, "Why is Normalization Necessary for Linear Recommenders?", SIGIR 2025. https://arxiv.org/pdf/2504.05805v2 (PDF read)
  - Setup:
    - Strong generalization: 8:1:1 user split, fold in 80%.
    - Six datasets, all models in one framework.
    - λ grid for plain LAEs: 10, 20, …, 500, 1000. Dropout p in 0.1–0.9. α in [0, 0.5], β in [0, 1].
    - Linear models run once (deterministic).
  - Average-over-all results (strong generalization), R@20 / N@20:

    | Dataset | EASE | EASE+DAN | EDLAE | RDLAE | GF-CF | Turbo-CF | SVD-AE |
    |---|---|---|---|---|---|---|---|
    | ML-20M | 0.3905 / 0.3390 | 0.3950 / 0.3430 | 0.3925 / 0.3421 | 0.3932 / 0.3422 | 0.3250 / 0.2736 | 0.3276 / 0.2728 | 0.3720 / 0.3205 |
    | Gowalla | 0.2414 / 0.1831 | 0.2527 / 0.1918 | 0.2469 / 0.1859 | 0.2499 / 0.1900 | 0.2252 / 0.1660 | 0.2281 / 0.1686 | 0.2292 / 0.1717 |
    | Yelp2018 | 0.1144 / 0.0933 | 0.1238 / 0.1011 | not quoted | not quoted | not quoted | 0.1144 / 0.0930 | not quoted |

    BSPM on Gowalla: 0.2373 / 0.1757. On Yelp2018, plain LAE (0.1160 / 0.0954) also edged EASE.
  - Tail and unbiased metrics improve much more than the average. ML-20M Tail N@20 goes from 0.0022 (EASE) to 0.0120 (EASE+DAN).
  - Takeaway: under strong generalization, **tuned LAEs beat every graph filter (GF-CF, BSPM, Turbo-CF) and SVD-AE**. DAN adds +1–8% on the average metric and multiples on tail items, at negligible cost.
- **DEQL.** Guo et al., "Generalizing Linear Autoencoder Recommenders with Decoupled Expected Quadratic Loss", ICLR 2026. https://arxiv.org/html/2603.07402v1
  - Strong-generalization R@20, EASE → EDLAE → DEQL(L2):

    | Dataset | EASE | EDLAE | DEQL(L2) |
    |---|---|---|---|
    | Games | 0.2733 | 0.2851 | 0.2998 |
    | Beauty | 0.1323 | 0.1324 | 0.1391 |
    | ML-20M | 0.3905 | 0.3925 | 0.3934 |
    | MSD | 0.3332 | 0.3336 | 0.3340 |

  - **Gains concentrate on the small Amazon datasets** (+5% R@20) and are negligible at scale.
  - Cost on Yelp2018 (CPU): about 8 min and 80 GB for DEQL vs about 2 min and 60 GB for EASE. The cost is O(n³).
  - The paper reports 5-split t-tests, but Games N@20 and Beauty R@20 are not significant vs EDLAE. The visible text does not say whether tuning used validation or test data.
- **RLAE/RDLAE.** Moon, Kim, Lee, "It's Enough: Relaxing Diagonal Constraints in Linear Autoencoders", SIGIR 2023. https://arxiv.org/abs/2305.12922
  - Reported "comparable or better", mainly on long-tail items.
  - In DAN's rerun, RDLAE is at most +0.7% over EASE on ML-20M R@20.
- **Turbo-CF.** SIGIR 2024. https://arxiv.org/html/2404.14243v1
  - Weak-generalization 70/20/10 split, baselines copied from earlier papers, **no EASE**.
  - Gowalla R@20: 0.1835 vs GF-CF 0.1849. Fast (0.3 s for the linear filter), but **beaten by EASE+DAN in DAN's weak-generalization table**.
- **ChebyCF.** SIGIR 2025. https://arxiv.org/html/2505.00552v1
  - **No EASE, iALS or RP3beta** in the comparison. Some baseline numbers are copied.
  - Margins over BSPM are 0.0003–0.0044. The paper does not say whether its grid search used validation data.
  - Reject: no evidence against tuned LAEs.
- **SVD-AE.** IJCAI 2024. https://arxiv.org/html/2405.04746v1
  - Beats EASE in its own weak-generalization table (Gowalla HR@10 14.40 vs 13.67).
  - But under DAN's strong-generalization rerun it is below EASE on all datasets (ML-20M R@20 0.3720 vs 0.3905).
  - Reject as a model. Truncated SVD is still a cheap speed fallback.
- **Propensity-weighted LAE.** Onishi et al., SIGIR-AP 2025. https://arxiv.org/abs/2512.20896
  - Uses a sigmoid(log frequency) propensity inside an LAE. Claims more diversity without accuracy loss. The abstract has no numbers.
  - A possible alternative to DAN's popularity knob. Not verified quantitatively.
- **Regularization landscape.** Li et al., arXiv Sept 2026. https://arxiv.org/abs/2609.11876
  - Theory paper: leading linear models reduce to nuclear-norm or Frobenius-norm regularizers. Proposes low-rank closed forms. No numbers checked.

**Alternatives considered and rejected:**
- GF-CF, Turbo-CF, ChebyCF, BSPM: they lose to tuned LAEs under strong generalization (DAN), and the papers proposing them omit EASE as a baseline.
- SVD-AE: below EASE under strong generalization.
- DEQL: marginal gains, about 4× EASE's time, unclear tuning hygiene. Revisit only if EDLAE+DAN plateaus on small catalogs.

**Why EDLAE + DAN wins for small stores:**
- Both are closed form with the same O(n³) solve as EASE.
- Together they add only 1–3 hyperparameters (p for EDLAE; α, β for DAN).
- The DEQL table shows that on small, sparse catalogs (Games, Beauty) regularization choice moves R@20 by around 5%, more than it does on ML-20M.
- DAN's tail gains matter in retail, where long-tail exposure has business value.

**Caveats:**
- All gains are from the authors' own groups (Lee lab for DAN, RLAE and L3AE), though DAN reran every baseline in one framework.
- None of these papers compares against tuned iALS or RP3beta.
- Validate on Mitate's own temporal split before turning any of them on by default.

---

## 4. L3AE (LLM-enhanced LAE): candidate for item text / cold-start

**Choice:** a phase-2 candidate when catalog text is available. Not in the core.

**Evidence:** **Moon, Park, Lee. "LLM-Enhanced Linear Autoencoders for Recommendation". CIKM 2025, DOI 10.1145/3746252.3760952.** https://arxiv.org/pdf/2508.13500v2 (PDF read)

Method:
1. Build an item-item semantic matrix with an EASE-style closed form over LLM item embeddings (NV-Embed-v2, LLaMA-3.2-3B or Qwen3-Embedding-8B).
2. Learn the collaborative weight matrix in closed form, with the semantic matrix as a distillation regularizer.

Data and tuning:
- Three Amazon-2023 datasets of small-store scale:
  - Games: 5,222 users / 2,676 items / 85,690 interactions
  - Toys: 14,750 / 13,358 / 250,509
  - Books: 25,300 / 30,966 / 640,901
- Ratings above 3, 10-core filtering, **8:1:1 split (weak-generalization style)**.
- Grid: λ, λ_X, λ_F in {0.1, …, 1000}; λ_KD in {10, …, 300}. λ is fixed first from interactions only, then λ = λ_KD + λ_X is enforced.

R@20 results:

| Dataset | EASE (interactions only) | L3AE | Best other linear model |
|---|---|---|---|
| Games | 0.2448 | 0.2737 | SGFCF 0.2651 |
| Toys | 0.1260 | 0.1573 | CEASE 0.1474 |
| Books | 0.2241 | 0.2409 | SGFCF 0.2302 |

- Average gain over EASE: +14.7% R@20 and +15.3% N@20.
- The headline +27.6% / +39.3% figures are versus LLM-enhanced neural baselines (AlphaRec and others), not EASE.

**Alternatives:**
- CEASE / Add-EASE (EASE with tag side-information) gain about 2–17% over EASE in the same table.
- Plain EASE on semantic embeddings alone is much worse (Games R@20 0.1536).

**Why it fits small stores:** the datasets are exactly Mitate's scale, the method is CPU-friendly apart from one-off embedding extraction, and side information helps most when interactions are sparse.

**Caveats:**
- Neural baselines used a fixed hidden dimension of 32 (weakly tuned).
- Significance was tested only against the non-linear models.
- No iALS, RP3beta or DAN in the comparison.
- The code repo (github.com/jaewan7599/L3AE_CIKM2025) has no license detected by the GitHub API, so treat it as reference only and reimplement.

---

## 5. SANSA (scalable sparse EASE)

**Choice:** not needed at Mitate's scale. Keep it as an escape hatch if a store's catalog goes beyond about 50k items.

**Evidence:**
- **Spišák, Bartyzal, Hoskovec, Peška, Tůma. "Scalable Approximate NonSymmetric Autoencoder for Collaborative Filtering". RecSys 2023, pp. 763–770, Best Short Paper Runner-Up.** Metadata via https://api.crossref.org/works/10.1145/3604915.3608827. README at https://github.com/glami/sansa.
- Method: a sparse approximate inverse of XᵀX (CHOLMOD or incomplete Cholesky, then UMR inverse). Training memory scales with nnz(XᵀX). Density is set by `weight_matrix_density`. Designed "for item sets with millions of items".
- API: `SANSA(SANSAConfig(l2, weight_matrix_density, gramian_factorizer_config=CHOLMODGramianFactorizerConfig() | ICFGramianFactorizerConfig(...), lower_triangle_inverter_config=UMRUnitLowerTriangleInverterConfig(scans, finetune_steps)))`, then `.fit(X)`, `.recommend(X, k, mask_input=True)`, `.forward(X)`, `.weights`.
- License and maintenance:
  - Apache-2.0.
  - PyPI `sansa` 1.1.0, released 2024-04-15. Python ≥3.10; classifiers 3.10–3.12. Pure-Python wheel.
  - Depends on `scikit-sparse>=0.4.8`. scikit-sparse 0.5.0 (2026-03-03, BSD-2) ships **no wheels**, so it needs SuiteSparse (`brew install suite-sparse` plus `SUITESPARSE_*` env vars).
  - Repo last pushed 2025-11-21.
  - Cornac also ships a `sansa` model directory.

**Why it does not win for small stores:** at n ≤ 30k a dense EASE solve fits in memory and is exact. SANSA adds a native build dependency and approximation error.

**Caveats:** I could not read the paper's accuracy tables (dl.acm.org returned 403, and the reproduce branch README has no numbers).

---

## 6. RP3beta and ItemKNN (cheap, strong neighbourhood baselines)

**Choice:**
- RP3beta (α, β, top-k) is the always-on, interpretable baseline and fallback.
- Cosine ItemKNN with shrinkage is the "related items" model.

**Evidence:**
- **Paudel, Christoffel, Newell, Bernstein. "Updatable, Accurate, Diverse, and Scalable Recommendations for Interactive Applications". ACM TiiS 7(1), issued 2016-12-19** (often cited as 2017). Metadata via https://api.crossref.org/works/10.1145/2955101. The abstract was not readable (ACM returned 403).
- Head-to-head results:
  - Quest (UMAP'22): RP3beta was **best** on Amazon Digital Music (nDCG@10 0.085 vs EASE 0.083), third on Epinions (0.163 vs 0.164), and fourth on ML-1M (0.315 vs 0.336).
  - Diffusion paper (TORS'26): RP3beta and ItemKNN beat all diffusion models on Amazon-Books. On Yelp, RP3beta R@10 is 0.0383 against iALS 0.0381 and ItemKNN 0.0360.
- Implementation: irspack `RP3betaRecommender(alpha=1, beta=0.6, top_k=None)` with tune ranges top_k 2–1000 and β 1e-5–0.5. Source: https://raw.githubusercontent.com/tohtsky/irspack/main/src/irspack/recommenders/rp3.py

**Alternatives:**
- UserKNN beat ItemKNN on all three Quest datasets, so it is worth including as a tuned candidate.
- implicit ItemItem with cosine, TF-IDF or BM25.

**Why it wins for small stores:**
- No training loop and sparse top-k storage.
- Popularity penalization (β) gives diversity.
- Explainable ("bought with X").
- Competitive with EASE on small, sparse retail-like data.

**Caveats:** quality depends on the top-k, β and shrink tuning, and it degrades on very dense data, where EASE and MF pull ahead (Quest on ML-1M).

---

## 7. Session-based linear models (pointer only)

- **SLIST.** Choi et al., "Session-aware Linear Item-Item Models for Session-based Recommendation", WWW 2021. https://arxiv.org/abs/2103.16104 — a closed-form EASE-style model for sessions.
- **LINK.** Choi et al., SIGIR 2025. https://arxiv.org/abs/2504.15057 — up to +14.78% R@20 over state-of-the-art linear session models, with up to 813× fewer inference FLOPs.
- DAN's appendix reports DAN also improves SLIST (not quantified here).
- Relevant if many small-store shoppers are anonymous. Belongs in the sequential/session review.

---

## 8. Tooling

Release data from the PyPI JSON API (https://pypi.org/pypi/<pkg>/json); repo activity from the GitHub API (https://api.github.com/repos/<repo>); both fetched 2026-10-19.

| Library | Latest (date) | License | Python | macOS arm64 | Relevant models | Verdict |
|---|---|---|---|---|---|---|
| **implicit** | 0.7.3 (2026-05-14); previous 0.7.2 (2023-09-29) | MIT | ≥3.9 (README: tested 3.9–3.14) | **Yes**: `macosx_11_0_arm64` wheels for cp39–cp314, plus universal2 | ALS, BPR, LMF, ItemKNN (cosine / TF-IDF / BM25) | Use for ItemKNN and fast ALS. ALS lacks ν/α₀; BPR is poor (Milogradskii 2024). |
| **irspack** | 0.5.2 (2026-07-25) | MIT | ≥3.9; 3.10–3.14 | **Yes**: cp310, cp311, cp312-abi3 arm64 wheels | iALS (α₀, ν per Rendle 2022), DenseSLIM (= EASE), EDLAE, RP3beta, P3α, Item/UserKNN, SLIM, TruncSVD, TopPop, Optuna tuning, evaluator | **Best fit.** One package covers EASE, EDLAE, iALS (Rendle), RP3beta and KNN with a shared evaluator. Small project (32 stars) but active. |
| **cornac** | 3.0.1 (2026-09-14) | Apache-2.0 | ≥3.10; 3.10–3.14 | Yes (universal2 cp310–cp314) | EASE, SANSA, BPR (best third-party BPR per Milogradskii 2024), WMF, KNN, RecVAE, LightGCN, SASRec | Good second framework for cross-checking. |
| **RecTools** | 0.19.0 (2026-06-13) | Apache-2.0 | 3.9–3.13 (<3.14) | Pure Python; depends on `pm-implicit` fork for py≥3.10 | EASE (dense), PureSVD, implicit ALS/BPR/ItemKNN wrappers, SASRec/BERT4Rec/HSTU | Usable. The extra implicit fork is a risk. |
| **sansa** | 1.1.0 (2024-04-15) | Apache-2.0 | ≥3.10 (3.10–3.12) | Pure Python, but scikit-sparse/SuiteSparse must be built (no wheels) | SANSA | Only for >50k items. |
| **LensKit (lkpy)** | 2026.4.0 (2026-09-30) | MIT (PyPI) | **≥3.12.5** (excludes 3.11) | Yes (cp312-abi3 arm64) | EASE, SLIM, Item/UserKNN, implicit MF, FlexMF; pulls in torch | Fine if Mitate uses 3.12+. Heavy deps. |
| **RecBole** | 1.2.1 (2025-02-24) | MIT | not declared | Pure Python + torch; pins `ray<=2.6.3` | Many models | Avoid as a baseline source: BPR collapses on MSD (N@100 0.0117). Stale. |
| **Elliot** | not on PyPI; repo pushed 2026-10-08 | Apache-2.0 | README badge 3.6–3.8; requirements pin TF 2.3.2 and numpy 1.18 | No | Most faithful BPR per Milogradskii 2024 | Not installable on 3.11/3.12 as pinned. |
| **RecPack** | 0.3.6 (2023-12-11) | **AGPL-3.0** (GitLab API) | ≥3.8; numpy 1.x | Pure Python | EASE, KNN, etc. | Avoid: copyleft and stale. |
| **daisyRec** | 2.3.0 (2022-08-14) | MIT | not declared | — | — | Stale. |
| Ferrari Dacrema RecSys2019 framework | repo pushed 2023-05-25 | **AGPL-3.0** | — | — | Reference RP3beta/EASE/SLIM code | Reference only. |
| Rendle iALS reference (google-research/ials) | monorepo active | Apache-2.0 | C++ | — | Block iALS | Reference for numbers. |

---

## Recommendations for Mitate

1. **Core model set:**
   - EASE (own ~30-line NumPy/SciPy implementation, float32, λ on a log grid, roughly 10–10⁴)
   - RP3beta
   - cosine ItemKNN
   - iALS (Rendle recipe: ν = 1, log grid over λ and α₀, double d until plateau)
   - popularity

   Use irspack (MIT, arm64 wheels, py3.11/3.12) as the reference implementation and cross-check EASE against Cornac.
2. **Add two cheap EASE variants as tunables, not new models:** EDLAE (one dropout parameter p) and DAN normalization (α ∈ [0, 0.5], β ∈ [0, 1]). Both are closed form, both beat plain EASE under strong generalization in a single-framework rerun (DAN, SIGIR'25), and DAN greatly improves long-tail recall.
3. **Do not add** GF-CF, Turbo-CF, ChebyCF, BSPM, SVD-AE or diffusion/GNN models. Under strong generalization they lose to tuned LAEs, and their source papers omit EASE, iALS or RP3beta.
4. **Do not use BPR** from implicit or RecBole. If a pairwise baseline is ever needed, use Cornac's BPR.
5. **Pipeline validation:** reproduce EASE on ML-20M with Liang's exact split (seed 98765, 10k/10k held-out users, `rating > 3.5`, 80/20 fold-in, truncated Recall). Targets are R@20 0.391, R@50 0.521, N@100 0.420 (SE about 0.002). Optionally check iALS against 0.395 / 0.532 / 0.425, which requires d = 2048.
6. **Tuning protocol:**
   - Tune on a separate validation split (TPE or a grid, at least 40 trials per model, matching the Quest and KDD'24 practice).
   - Report on a temporal hold-out.
   - Give every baseline the same budget, since that is the central lesson of Dacrema 2019, Rendle 2020/2022 and Milogradskii 2024.
7. **Phase 2:** L3AE-style semantic regularization (or CEASE / Add-EASE with tags) for item cold-start and very sparse stores, on our own reimplementation.
8. **Scale guard:** dense EASE for n ≤ ~30k items. Above that, use SANSA (requires a SuiteSparse build) or iALS.

## Could not verify

- **"BISM"**: no paper by this name was found in searches. It may be a mis-remembered name. Not cited.
- **"Linear models are all you need"-style paper**: no paper with this title was located. Not cited.
- **SANSA accuracy and runtime numbers**: dl.acm.org returned 403, no arXiv version was found, and the repo READMEs have no tables.
- **RP3beta original abstract and results** (TiiS 2016/17): only Crossref metadata was retrieved. The method is taken from the irspack source and the benchmark tables above.
- **Netflix held-out user count in Liang et al. (believed 40k)**: Liang's public notebook covers ML-20M only. Not confirmed.
- **Steck & Liang, "Negative Interactions… Don't go Deeper, go Higher"** (RecSys 2021, higher-order EASE): seen only in search snippets. Numbers not read.
- **GF-CF original paper** (Shen et al., CIKM 2021): numbers taken only as reproduced in DAN, Turbo-CF and ChebyCF.
- **DEQL tuning hygiene and code license**: not visible in the HTML read.
- **Mult-VAE paper itself** (Liang et al., WWW 2018): the protocol was verified from the authors' code, not the paper.
- **The brief's "Paudel et al." author list**: Crossref gives Paudel, Christoffel, Newell, Bernstein. No other author list was checked.
