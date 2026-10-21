# 04: Catalog-only cold start, text embeddings, LLM components, complements, LTR blending, ANN

Scope: components that let Mitate recommend from **catalog text alone** and blend several candidate sources. Constraints:
- serving is CPU-only;
- all offline work runs on an 8 GB Apple M1;
- no paid LLM API (local Ollama / llama.cpp / transformers only).

Review date: 2026-10-19. Every number below was read from the URL listed with it. Model licences and parameter counts come from the Hugging Face model API (`https://huggingface.co/api/models/<id>`) and the model-card README. Anything I could not open or confirm is listed under "Could not verify". Note 01 (`01_cf_linear.md`) already covers L3AE in depth; §2 here only adds the cold-start angle.

Notation: R@k = Recall@k, N@k = NDCG@k, HR@k = HitRate@k. "Item cold start" means a new SKU with zero interactions. "User cold start" means a new or very short-history visitor.

---

## 1. An "LLM-based cold-start" baseline: what is credible and runnable on 8 GB

### 1a. Verification: "Small LLMs can be good coldstart recommenders" (PMC13047167)

**Source:** Noel, Monterola, Tan. *Frontiers in Artificial Intelligence* 9:1705245, 2026. https://pmc.ncbi.nlm.nih.gov/articles/PMC13047167/ (fetched)

**Models:**
- Danube-1.8B (Llama-2-style, 1T training tokens).
- Gemma-2B (3T tokens).

**LoRA setup:**
- Danube: 8,650,752 trainable parameters (0.47%). Gemma: 9,805,824 (0.39%).
- **Rank, alpha, learning rate and epochs are not reported.**
- Trained on GTX 1080 Ti GPUs; 5 runs per model/dataset pair, averaged.

**Task:**
- Item IDs are written as text, and the LLM generates the next item ID **digit by digit**.
- Input is the last 5 items; target is the next item.
- Datasets: MovieLens-10M (55,902 train / 13,976 test sequences) and a Yoochoose-clicks sample (53,840 / 13,560).
- Only metric is HR@1.

| HR@1 | Danube-1.8B | Gemma-2B | GRU4Rec | SASRec | BERT4Rec |
|---|---|---|---|---|---|
| Yoochoose | **0.0555** | 0.0540 | 0.0440 | 0.0354 | 0.0499 |
| MovieLens | 0.0995 | **0.1019** | 0.0934 | 0.0898 | 0.0854 |

**History-length experiment** (MovieLens, Danube vs GRU4Rec only):

| History length | 5 | 10 | 20 | 50 |
|---|---|---|---|---|
| Danube | 0.0995 | 0.1044 | 0.0973 | 0.0917 |
| GRU4Rec | 0.0934 | 0.0961 | 0.1037 | 0.1080 |

- GRU4Rec overtakes the LLM between 10 and 20 items of history.
- **SASRec was not part of this experiment, so the paper does not say at what history length the LLM loses to SASRec.**

**Cost:**
- 34.3 ms/token (Danube) and 59.5 ms/token (Gemma) on an RTX 2080 Ti, with about 5.2 GB of GPU memory.
- No CPU numbers. No baseline latency numbers.

**Credibility: low to moderate.**
- "Cold start" here means a 5-item history. It is **not** a new item or a zero-history user, so it does not address Mitate's catalog-only case.
- Margins are under 0.01 absolute HR@1. ANOVA/Tukey tests are mentioned without p-values.
- Only one metric (HR@1), two datasets, and no LoRA hyperparameters.
- Generating item IDs as digits ties the model to a fixed ID space. A new SKU still needs retraining.

**Affordability on Mitate hardware:**
- Inference with a 2B model quantised in llama.cpp is plausible on an M1.
- LoRA fine-tuning a 2B model in 8 GB is marginal and was not tested here.
- Gemma-2B also comes under the Gemma Terms of Use (see §5).

**Verdict:** do not use it as Mitate's LLM cold-start baseline. At most, cite it as related work for short-history users.

### 1b. 2024–2026 alternatives (fetched)

| Paper | Venue | What the LLM does | Key evidence | Runs on 8 GB M1? |
|---|---|---|---|---|
| Sheng et al., "Language Representations Can be What Recommenders Need" (AlphaRec; v1 title "Language Models Encode Collaborative Signals in Recommendation") https://arxiv.org/html/2407.05441v4 | ICLR 2025 (oral) | Frozen LM **title embeddings**, then MLP + graph convolution + contrastive loss | See note 1 below the table | Yes with a ≤0.6B local embedder. The paper used OpenAI text-embedding-3-large, so a local embedder is a substitution that needs a test |
| He et al., "LLM2Rec" https://arxiv.org/html/2506.21579v1 | KDD 2025 | Qwen2-0.5B fully fine-tuned on next-item prediction (CSFT), then turned into a bidirectional embedder (MNTP + item-level contrastive learning) | Embeddings plugged into SASRec. R@10: Games 0.0865 vs LLMEmb 0.0813; out-of-domain Baby 0.0708 vs 0.0659. Beats BERT/GTE/BGE/LLM2Vec embeddings | Training used 4×A40. Full fine-tuning of 0.5B (10k steps, batch 128) is **not practical on 8 GB**. Inference is fine |
| Zhang et al., "Are ID Embeddings Necessary? Whitening Pre-trained Text Embeddings…" https://arxiv.org/abs/2402.10602 | arXiv 2024 (abstract page lists no venue). AAAI 2024 sibling paper "Dual-View Whitening" from the same group (search result only) | **Whitening** of frozen text embeddings for sequential rec | Pre-trained text embeddings are anisotropic: average item cosine above 0.8. Whitening fixes it | Yes. Whitening is a d×d matrix |
| Li et al., "Text Is All You Need" (Recformer) https://arxiv.org/abs/2305.13731 | KDD 2023 | Longformer-style model over item key-value text | Abstract claims gains "especially in low-resource and cold-start settings"; no numbers on the abstract page | Pre-training is GPU work |
| Wei et al., "LLMRec" https://arxiv.org/abs/2311.00423 | WSDM 2024 | LLM augments the graph: interaction edges, item attributes, user profiles | Abstract only, no numbers on the page | Augmentation is offline. Feasibility with a local 3–8B model is unknown |
| Wang et al., "LLMs as Data Augmenters for Cold-Start Item Recommendation" https://arxiv.org/abs/2402.11724 | arXiv 2024 (Google authors) | LLM infers user preference for cold items, added through an auxiliary pairwise loss | "Significant improvements" on Amazon data, no numbers on the page | Offline only. Cost scales with users × cold items |
| Huang et al., "LLM Simulator for Cold-Start Recommendation" (ColdLLM) https://arxiv.org/abs/2402.09176 | WSDM 2025 | LLM simulates interactions for cold items behind a filter funnel | 2-week A/B test showed a GMV gain (no figure on the page) | The funnel design matters at scale. A small store could run it offline |
| Hou et al., "LLMs are Zero-Shot Rankers for Recommender Systems" https://arxiv.org/abs/2305.08845 | ECIR 2024 | Zero-shot listwise rerank | LLMs ignore history order and show **position and popularity bias**; prompting and bootstrapping mitigate this | Yes for small models, but see the next two rows |
| **Kusano, Abe, Takeoka, "Are LLMs Really Effective for Training-Free Cold-Start Recommendation?"** https://arxiv.org/html/2512.13001v1 | arXiv Dec 2025 | Compares zero-shot LLM rerankers with text embedding models (TEMs), user cold start | See note 2 below the table | **Directly relevant.** Qwen3-Embedding-0.6B is CPU-feasible |
| Dong et al., "Diagnosing and Mitigating Retrieval Bottlenecks in LLM-Based Cold-Start Recommendation" https://arxiv.org/abs/2606.29947 | arXiv Jun 2026 | Qwen3-8B / 32B prompt reranking | See note 3 below the table | Supports spending effort on retrieval and fusion, not on an LLM reranker |
| Zhang et al., "Cold-Start Recommendation towards the Era of LLMs" (survey) https://arxiv.org/abs/2501.01945 | arXiv 2025 | Taxonomy: content features, then graph relations, then domain information, then LLM world knowledge | n/a | n/a |

Notes for the table:
1. **AlphaRec, linear map from LM embeddings to item space, Recall** (Movies / Games / Books):
   - BERT 0.0415 / 0.0524 / 0.0226
   - Llama2-7B 0.1027 / 0.1249 / 0.0662
   - SFR-Embedding-Mistral 0.1152 / 0.1370 / 0.0738
   - Full AlphaRec beats LightGCN and XSimGCL (Games 0.1519 vs 0.1138).
   - Zero-shot on ML-1M it trails a fully trained LightGCN (0.1486 vs 0.2019).
2. **Kusano et al., narrow cold start, average over datasets, R@10 / N@10** (50-item candidate pools):
   - Qwen3-Embedding-0.6B 0.362 / 0.401
   - multilingual-e5-large 0.300 / 0.339
   - bge-m3 0.293 / 0.331
   - BM25 0.315 / 0.362
   - gpt-4.1 rerank 0.311 / 0.283
   - **Qwen3-8B rerank 0.114 / 0.101**
   - Authors' conclusion: TEMs beat LLM rerankers in both cold and warm start.
3. **Dong et al.:**
   - With the gold item guaranteed in the pool, calibrated LLM rerankers do **not** consistently beat strong CF and content baselines.
   - Without injection, single retrievers put the gold item in a 200-item pool only **4.6–22.9%** of the time.
   - Prompt-based LLM reranking often *degrades* a learned fusion pool.

**Choice: the "LLM-based cold-start" baseline Mitate reports.**
1. **LLM-as-embedder (primary):** encode each item's catalog text with a local, licence-clean LLM-derived embedder (Qwen3-Embedding-0.6B, Apache-2.0).
   - Text template like L3AE's: `Title; Category; Brand; Description`.
   - Optionally whiten (WhitenRec), then
   - (a) do item-to-item kNN for new items, and
   - (b) feed the item-item semantic matrix into the closed-form L3AE / EASE regulariser (note 01 §4).
   - This is the setting where the 2025–26 evidence (Kusano; AlphaRec Table 1; L3AE's backbone ablation) consistently favours LLM embeddings over BERT-era encoders.
2. **LLM-as-profiler (secondary ablation):**
   - A local 3–4B instruct model (Ollama) rewrites sparse or messy catalog rows into a normalised attribute profile (type, use case, audience, material), which is then embedded as in step 1.
   - This is LLMRec / RLMRec-style augmentation scaled down. It runs once per item, so it is affordable offline on CPU. A few thousand items at a few seconds each is plausible but **not measured**.
   - It must be reported as an ablation (raw text vs profiled text) because no fetched paper measures this exact variant at small scale.
3. **Small-LLM zero-shot reranking: a negative-control ablation only.**
   - Kusano and Dong both find it below embedding baselines, with Qwen3-8B especially poor.
   - It is also the most expensive option per request.

**Alternatives:**
- LLM2Rec: best accuracy, but needs GPU fine-tuning.
- AlphaRec: needs a GNN and contrastive training. Its zero-shot transfer is interesting but weaker than in-domain CF on ML-1M.
- Recformer: a GPU pre-training budget.

**Caveats:**
- Most evidence is from Amazon/MovieLens English text. Kusano covers user cold start, not item cold start.
- None of these papers test Indic or Hinglish catalogs.

---

## 2. L3AE (arXiv 2508.13500) and cold start

**Source:** Moon, Park, Lee, CIKM 2025. https://arxiv.org/html/2508.13500v2 (fetched). Details are in note 01 §4.

What matters for cold start:
- **The paper has no cold-start experiment.** The word "cold" does not appear (verified on the HTML), and 10-core filtering removes rare items.
- What it does report is that gains are "particularly pronounced" on **long-tail** items: Games, tail = bottom 80% by popularity (Fig. 1, values shown only in the figure).
- Backbones: NV-Embed-v2 (7B, main), LLaMA-3.2-3B and Qwen3-Embedding-8B.
  - Qwen3-Emb-8B L3AE R@20 is 0.2625 / 0.1441 / 0.2356 vs 0.2737 / 0.1573 / 0.2409 for NV-Embed-v2 (Games / Toys / Books).
  - **No sub-1B embedder was tested.**
- NV-Embed-v2's licence is non-commercial (HF card, not re-checked here; see "Could not verify"). Mitate should use Qwen3-Embedding-0.6B and measure the drop.
- **My reading, not the paper's claim:** the semantic matrix F covers every item. A closed-form solution regularised towards F should therefore give a zero-interaction item a non-zero column through F. Whether that ranks new items well is untested, so it is an experiment for Mitate, not evidence.

---

## 3. LLM reranking for complementary items (arXiv 2507.16237)

**Source:** Xu, Zhang, "LLM-Enhanced Reranking for Complementary Product Recommendation", arXiv July 2025 (no venue). https://arxiv.org/html/2507.16237v1 (fetched)

**Model and pipeline:**
- **Llama-3.3-70B** powers two sequential prompt "agents" (diversity, then accuracy). The paper does not say whether it ran locally or through an API.
- Inputs: anchor title plus candidate IDs and titles.
- Candidates: the top 50 from a GNN retriever (GraphSAGE, GAT or SComGNN). The diversity agent reranks all 50 and the accuracy agent keeps 25.

**Data:** Amazon Electronics, Cell Phones, Grocery and Home.

**Gains:**
- GraphSAGE Hit@1: Cell Phones 0.154→0.351, Electronics 0.323→0.487, Grocery 0.320→0.400, Home 0.383→0.551.
- Abstract: "at least 50% lift in accuracy … 2% lift in diversity" on average.
- Gains shrink as K grows. The accuracy agent reduces diversity: Vocabulary@10 on Cell Phones fell 122.9→109.3.

**Cost:** not reported at all. No tokens, no latency.

**Assessment for Mitate:**
- **Not reproducible on our hardware.** A 70B model does not fit in 8 GB at any useful quantisation.
- The GNN baselines use only category and price as node features, which inflates the headroom an LLM with titles can claim.
- No small-model ablation, so the claim cannot be transferred to a 3–8B local model without our own test.
- Useful as design evidence for "LLM as a complementarity judge over a small candidate list", which Mitate can do **offline at category level** instead of per request (§4).

---

## 4. Complementary-item recommendation: methods and evaluation

### 4a. Methods

- **McAuley, Pandey, Leskovec, "Inferring Networks of Substitutable and Complementary Products"** (KDD 2015; the arXiv page lists no venue). https://arxiv.org/abs/1506.08839 (fetched)
  - Supervised link prediction from review-text topics, ratings, specs, price and brand.
  - Run on about 9M Amazon products and 237M links.
  - Substitutes are bought *instead of* each other; complements are bought *together*.
  - The abstract page reports no metric values.
- **Hao et al., "P-Companion"** (CIKM 2020, Amazon). https://www.amazon.science/publications/p-companion-a-principled-framework-for-diversified-complementary-product-recommendation (fetched)
  - States explicitly that **co-purchased ≠ complementary**, so it builds cleaner distant-supervision labels.
  - Two stages: (1) an encoder-decoder predicts several **complementary product types** for the query type; (2) the query embedding is projected into each type's subspace to pick items.
  - Cold-start items are handled through a pretrained GAT "Product2vec".
  - +7.1% Hit@10 offline. The online deployment claim (200M+ products, "significant gains") comes without numbers.
  - **This is the type-then-item structure Mitate should copy.**
- **Li and Du, survey** (arXiv 2403.16135, 2024). https://arxiv.org/html/2403.16135v1 (fetched)
  - Gives three definitions: buy-together, use-together, and collocation (fashion).
  - Co-purchase labels are noisy (bananas + bottled water).
  - Metrics across the 34 surveyed studies are inconsistent: AUC, HR@k, NDCG@k, MRR.
  - Category-level complements are a recurring cold-start device: new products inherit their category's complement vectors.
- **Walmart, "Grocery to General Merchandise"** (RecSys 2025). https://arxiv.org/html/2509.02890v2 (fetched)
  - gpt-4o runs a three-agent pipeline:
    1. usage contexts for the anchor;
    2. 10 complementary product types per context;
    3. an LLM judge that drops scores below 0.4.
  - Outputs are mapped to catalog items with E5 embeddings and filtered by market-basket lift, support and confidence.
  - Results:
    - unique recommendations 16.5k→78.2k;
    - add-to-cart 1.1%→1.5% (+36% relative, about +0.4 pp absolute);
    - 200 anchors manually reviewed at 94% relevancy, agreeing with the combined score 95% of the time.
  - **This is the closest published analogue to an LLM-generated category complement graph.**
- **Chen et al. (Walmart), "KG Completion Models are Few-shot Learners"** (arXiv 2305.09858, 2023). https://arxiv.org/abs/2305.09858 (fetched)
  - PaLM and GPT-3.5 label complementary and substitute relations **between product types** with 1–5 examples.
  - The authors claim performance "competitive with humans". No numbers on the abstract page.
- **CCGen** (Amazon, arXiv 2305.11480). https://arxiv.org/abs/2305.11480 (fetched)
  - Fine-tuned LMs generate ranked complementary *concepts* with explanations distilled from a teacher. No numbers on the abstract page.
- **COSMO** (Amazon, SIGMOD Companion 2024). https://www.amazon.science/publications/cosmo-a-large-scale-e-commerce-common-sense-knowledge-generation-and-serving-system-at-amazon (fetched)
  - LLM seed assertions are filtered by critic classifiers trained on human annotations, then distilled into a smaller COSMO-LM.
  - Used about 30k annotated instructions, covering 18 major categories with millions of assertions.
  - Deployed in Amazon search navigation. The abstract reports A/B improvements without figures.
- **Yamasaki et al., "Function-based Labels for Complementary Recommendation"** (*Pattern Recognition Letters*, arXiv 2507.03945). https://arxiv.org/html/2507.03945v3 (fetched). This is the most useful source for a rubric:
  - Labels:
    - A = same function (substitute);
    - B = replenishment;
    - C-1 = must be combined to be usable; C-2/C-3 = one is more useful with the other; C-4 = both more useful;
    - D = unrelated;
    - E = seems related but only via a third item.
  - Human agreement: Fleiss κ = 0.63 overall, about 0.4 on C-3/C-4. 2,625 usable pairs. About 50 s and 16 JPY per pair.
  - LLM-as-judge with 7 samples at T = 0.6 and majority vote:
    - GPT-4o-mini: macro-F1 0.849 (9-class), 0.729 (3-class), consistency 0.989.
    - **Llama-3.1-70B (8-bit): 0.822 / 0.703.**
  - LLMs over-reject ambiguous pairs as D.
  - A fine-tuned ModernBERT-Ja-70m reached macro-F1 0.911. **A small supervised classifier beat the LLM judges.**
  - The dataset is public (github.com/okamoto-lab/fbl_dataset).

### 4b. Evaluating an LLM-generated category complement graph

Proposed protocol. My synthesis: each element is tied to the source it comes from; the combination itself is not from any one paper.

1. **Label scheme: Yamasaki FBL, collapsed for the graph.**
   - Accept B and C-1..C-4 as complement.
   - Reject A (substitute leakage) and D (nonsense).
   - Flag E (indirect) as weak, and keep it only if basket lift supports it.
   - Record **direction** (phone → case ≠ case → phone). C-2/C-3 are directional, and P-Companion and the survey both treat complementarity as asymmetric.
2. **Nonsense rubric** (each edge gets a label plus a 0–1 score, as in Walmart's 4-dimension judge):
   - **(i) Functional link:** can you state how A is used with, or replenishes, B? (FBL B/C)
   - **(ii) Not a substitute:** different function, or at least a different product type. Reject if the types share a parent and a function (FBL A).
   - **(iii) Shared scenario plausibility:** the same occasion or activity (Walmart "relevance & coherence" and the "usage context" agent).
   - **(iv) Practical utility for this store:** the target type exists in the catalog with stock (Walmart "matching accuracy" and CE×LLM gating).
   - **(v) Audience and attribute compatibility:** gender, age, size and platform. For example, an iPhone case for an Android phone is nonsensical even though "phone → case" is valid. This is the COSMO `used_for_audience` relation.
3. **Raters and agreement:**
   - Two humans label a stratified sample: per parent category, plus all high-degree nodes. Report Fleiss or Cohen κ.
   - Yamasaki's 0.63 is the reference: κ well below about 0.6 means the rubric is ambiguous, not that the graph is bad.
4. **LLM judge:**
   - A different model from the generator, with k = 7 samples and majority vote (Yamasaki).
   - Validate it against the human sample (macro-F1, plus the confusion between A, D and E).
   - Expect over-rejection of borderline pairs.
5. **Behavioural validation where data exists:** for stores with baskets, compute type-level co-purchase lift.
   - Report precision of LLM edges with lift > 1 and recall of high-lift type pairs.
   - Remember that co-purchase is a noisy proxy (P-Companion, survey). Disagreement is a signal to inspect, not ground truth.
6. **Graph-level sanity checks (cheap, automatic):**
   - out-degree distribution, so no type has 0 or more than about 20 complements;
   - self-loops;
   - edges whose target type has no SKUs in the store;
   - symmetric-edge rate, since a high rate suggests "related" rather than "complement";
   - share of edges inside the same parent category, a proxy for substitute leakage.
7. **Downstream:** offline HR@k / NDCG@k against basket pairs for items mapped through the graph, plus diversity (Entropy@k as in 2507.16237).

**Choice:**
- Generate the **type-level** graph offline with a local instruct model. Ollama Qwen3 4–8B Q4 fits 8 GB; quality is unmeasured for this task.
- Judge it with the rubric above.
- Map types to SKUs with the §5 embedder plus attribute filters.
- Optionally train a tiny supervised edge classifier on the human-labelled sample, since Yamasaki's 70M model beat the LLM judges.

**Caveat:** every published LLM-judge number above uses GPT-4o-mini, gpt-4o or a 70B model. Quality with small local models is unverified.

---

## 5. Text embedding models: quality, CPU cost and licences

### 5a. Model table

Licence = `cardData.license` from the HF API. Params = HF API `safetensors.total` unless noted. "Commercial" is my reading of the licence text, not legal advice.

| Model (HF URL) | Params | Dims | Max tokens | Licence | Commercial? | Multilingual / Indic | Reported scores (source) |
|---|---|---|---|---|---|---|---|
| [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 595.8M | 1024 (MRL 32–1024) | 32K | Apache-2.0 | Yes | "100+ languages"; Indic not itemised | MTEB-Multilingual mean (task) 64.33, retrieval 64.64 (card). Cold start R@10 0.362 (Kusano) |
| [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | 117.7M | 384 | 512 | MIT | Yes | 100 langs (XLM-R); HF tags hi/ta/te/bn/mr | **MTEB(Indic) 64.7**, MTEB(Multilingual) 55.5 (MMTEB Table 2) |
| [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) | 278.0M | 768 | 512 | MIT | Yes | same | MTEB(Indic) 64.6, Multilingual 57.0 (MMTEB) |
| [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) | 559.9M | 1024 | 512 | MIT | Yes | same | MTEB(Indic) 66.4, Multilingual 58.6 (MMTEB). Cold start R@10 0.300 (Kusano) |
| [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) | 559.9M | 1024 | 512 | MIT | Yes | same | **MTEB(Indic) 70.2 (rank 1)**, Multilingual 63.2 (MMTEB) |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | 568M (from Arctic/Nomic card tables; API gives none) | 1024 (+ sparse + ColBERT) | 8192 | MIT | Yes | "100+ working languages" | MTEB-Multilingual 59.56 (Qwen card). MIRACL 69.2 (Nomic card). Cold start R@10 0.293 (Kusano) |
| [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) | 302.9M | 768 (MRL 512/256/128) | 2048 | **Gemma Terms of Use** (`gemma`, gated) | Yes, with conditions (see 5b) | "100+ spoken languages" | MTEB Multilingual v2 61.15, English v2 69.67 (768d). Q8_0 60.93 / 69.49. **No float16** |
| [jinaai/jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3) | 572.3M | 1024 (MRL 32–1024) | 8192 | **CC-BY-NC-4.0** | **No** (card: commercial use needs a Jina licence) | Tuned on 30 langs incl. Hindi | FLORES retrieval: hi 98.8, ta 95.8 (Vyakyarth card) |
| [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) | 335.1M | 1024 (MRL, binary) | 512 | Apache-2.0 | Yes | English | MTEB-en (56) 64.68, retrieval 54.39 (card) |
| [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | 136.7M | 768 (MRL 64–768) | 8192 | Apache-2.0 | Yes | English | MTEB 62.28 @768, 61.04 @256 (card) |
| [nomic-ai/nomic-embed-text-v2-moe](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe) | 475M total / 305M active | 768 (MRL→256) | 512 | Apache-2.0 | Yes | ~100 langs; HF tags hi/ta/te/bn/mr | BEIR 52.86, MIRACL 65.80 (card). Needs `trust_remote_code` |
| [Snowflake/snowflake-arctic-embed-m-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0) | 305.4M (113M non-embedding) | 768 (MRL 256) | 8192 | Apache-2.0 | Yes | Multilingual; HF tags hi/ta/te/bn/mr | BEIR 55.4, MIRACL(4) 55.2, CLEF 51.7. At 256d: 54.4 / 54.0 (card) |
| [ibm-granite/granite-embedding-278m-multilingual](https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual) | 278.0M | 768 | 512 | Apache-2.0 | Yes | **12 languages; Hindi/Tamil not listed** | MIRACL(18) 58.3 (card) |
| [ibm-granite/granite-embedding-107m-multilingual](https://huggingface.co/ibm-granite/granite-embedding-107m-multilingual) | 107.0M | 384 | 512 | Apache-2.0 | Yes | same 12 | (card table not extracted) |
| [ibm-granite/granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2) | 47.7M | 384 | 8192 | Apache-2.0 | Yes | English | BEIR 50.9, MTEB-v2 61.1 (card) |
| [minishlab/potion-base-8M](https://huggingface.co/minishlab/potion-base-8M) (static) | 7.6M | 256 | unbounded | MIT | Yes | English (bge-base tokenizer) | MTEB avg 51.08 vs all-MiniLM-L6-v2 55.93 (card) |
| [minishlab/potion-retrieval-32M](https://huggingface.co/minishlab/potion-retrieval-32M) (static) | 32.3M | 512 | unbounded | MIT | Yes | English | (not extracted) |
| [minishlab/potion-multilingual-128M](https://huggingface.co/minishlab/potion-multilingual-128M) (static, distilled from bge-m3) | 128.1M | 256 | unbounded | MIT | Yes | 101 langs; tags hi/ta/te/bn/mr | MMTEB mean (task) 47.31 vs LaBSE 52.07; retrieval 37.86 vs 33.17 (card) |
| [sentence-transformers/LaBSE](https://huggingface.co/sentence-transformers/LaBSE) | 470.9M | 768 | 256 (ST config) | Apache-2.0 | Yes | 109 langs | MTEB(Indic) 61.9, Multilingual 52.1 (MMTEB) |
| [l3cube-pune/indic-sentence-similarity-sbert](https://huggingface.co/l3cube-pune/indic-sentence-similarity-sbert) | ≈237M (computed from config: BERT-base, 197,285 vocab, the same vocab as MuRIL) | 768 | 512 | CC-BY-4.0 | Yes, with attribution | en + 10 Indic | Paper (arXiv 2304.11434) claims > LaBSE on Indic STS; no numbers on the page |
| [krutrim-ai-labs/Vyakyarth](https://huggingface.co/krutrim-ai-labs/Vyakyarth) | 278.0M | 768 | **128** | Krutrim Community License v1.0 | **No for SaaS** without a separate agreement (LICENSE §3, §7, §11, §15) | Indic + English | FLORES retrieval: hi 99.9, ta 97.9, te 97.5 (card) |
| [google/muril-base-cased](https://huggingface.co/google/muril-base-cased) | ≈237M (computed) | 768 | 512 | Apache-2.0 | Yes | 17 Indian langs, **trained with transliterated pairs** | MLM encoder, **not a sentence embedder** |
| [ai4bharat/IndicBERTv2-MLM-only](https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only) | 278M (card) | 768 | 512 | MIT | Yes | 23 Indic + en | MLM encoder, not a sentence embedder |
| Sarvam AI | — | — | — | — | — | — | **No embedding model in the `sarvamai` HF org** (listed via the API: only LLM, translate and speech models) |

**Benchmark sources:**
- MMTEB (Enevoldsen et al., ICLR 2025). https://arxiv.org/html/2502.13595v2
  - **MTEB(Indic) exists:** 23 tasks (4 bitext mining, 13 classification, 2 retrieval, 1 each of STS, pair classification, clustering and reranking).
  - Only 2 of the 23 tasks are retrieval, so MTEB(Indic) is a weak proxy for product search.
  - The Table 2 results above cover only the models MMTEB evaluated: no Qwen3, EmbeddingGemma, bge-m3 or Arctic.
- Qwen3 card table. https://huggingface.co/Qwen/Qwen3-Embedding-0.6B (comparison rows taken from the leaderboard on 2025-05-24)

**My CPU throughput measurement** (M1 8 GB, 4 torch threads, sentence-transformers, fp32, batch 32): see §5c.

### 5b. Licence notes that change the shortlist

- **jina-embeddings-v3: CC-BY-NC-4.0.** The card says use beyond AWS/Azure listings needs a commercial licence. **Exclude** it from anything a store runs commercially.
- **Vyakyarth (Krutrim Community License v1.0):**
  - It defines "Commercial Use" as use where more than 1M MAU use the software.
  - But §3 says commercial use is permitted "only through a separate commercial license agreement".
  - §7 prohibits commercial distribution, including "integration into proprietary products".
  - §11 (cloud deployment) restricts commercial SaaS, and §15 restricts commercial use of derivative models, without such an agreement.
  - It also bars use to compete with Krutrim (§5).
  - Too ambiguous for a SaaS product: **exclude.**
- **EmbeddingGemma: Gemma Terms of Use** (https://ai.google.dev/gemma/terms, last modified 2026-04-01, fetched).
  - It does not restrict commercial use as such, and Google claims no rights in outputs.
  - But it requires: passing the Prohibited Use Policy through to downstream users as an enforceable term; shipping a NOTICE file when distributing weights; marking modified files.
  - Google "reserves the right to restrict (remotely or otherwise) usage".
  - The HF repo is **gated** (manual acceptance), which complicates automated installs.
  - Usable, but it is the only shortlisted model with custom terms, so prefer an Apache-2.0 or MIT equivalent unless it wins clearly.
- **L3Cube IndicSBERT: CC-BY-4.0.** Commercial use is fine with attribution. Note that CC licences are not designed for model weights.
- **Everything else in the table is Apache-2.0 or MIT.**

### 5c. CPU cost

**Measured on the target laptop** (Apple M1, 8 GB, torch CPU, 4 threads, sentence-transformers fp32, batch 32, `max_seq_length = 256`).
- Synthetic product strings: "short" ≈ 15 tokens (titles), "long" ≈ 100+ tokens (title + description).
- Indicative only: one run, other processes active.

| Model | Short texts/s | Long texts/s | Peak RSS |
|---|---|---|---|
| BAAI/bge-small-en-v1.5 (33M, 384d; cached locally) | 111 | 23.5 | 0.32 GiB |

- This is the only model measured. Weight downloads for multilingual-e5-small and Qwen3-Embedding-0.6B stalled in this session (only README/tokenizer blobs arrived), so their CPU throughput is **not measured**. Script: `scratchpad/bench/emb_bench.py`, re-runnable.
- The machine was under other load, so treat the value as a lower bound.
- Extrapolating by parameter count, a 600M model would be roughly 10–20× slower than bge-small. That is still about 10K catalog items in well under an hour, offline. This is an estimate, not a measurement.

Static models: the Model2Vec cards claim speed "orders of magnitude faster". Not measured here because `model2vec` is not installed.

**Rule of thumb:** a 5–30k-item catalog is embedded once offline, so even the slowest shortlisted model is a one-off cost of minutes. Serving cost only matters for **query** embedding in search, which is single short strings.

### 5d. Indic, Hinglish and mixed script

- **Best published Indic signal:** MTEB(Indic) is led by the multilingual-e5 family. e5-large-instruct scores 70.2; **e5-small 64.7 roughly equals e5-base 64.6**, both above LaBSE at 61.9.
  - That makes e5-small the best quality-per-FLOP Indic option among *benchmarked* models.
  - Qwen3-Embedding-0.6B and EmbeddingGemma claim 100+ languages, but I found no MTEB(Indic) scores for them.
- **Romanised Hinglish:**
  - No e-commerce Hinglish retrieval benchmark found.
  - Mazumder et al., "Neither Here Nor There" (EMNLP Findings 2026, https://arxiv.org/abs/2603.19771) shows that standard multilingual encoders keep code-mixed text "loosely connected to either language" and route it through an English-dominant subspace. Devanagari input adds complementary signal.
  - MuRIL was trained with transliterated pairs, but it is an MLM encoder, not a sentence embedder. L3Cube IndicSBERT appears to be built on the MuRIL vocabulary (inferred from config).
  - **Practical implication:** for Hinglish queries, pair the dense embedder with a lexical or sparse channel (BM25 on transliterated tokens, or bge-m3 sparse), and build a small in-house Hinglish query → product test set before choosing.

**Choice:**
- **Default:** `Qwen3-Embedding-0.6B` (Apache-2.0) for catalog embeddings. It has the strongest published cold-start evidence (Kusano) and MTEB-Multilingual 64.33. Truncate via MRL to 256–512d if memory matters.
- **Indic-first or low-CPU stores:** `multilingual-e5-small` (MIT, 118M, 384d). It has the best published MTEB(Indic) per parameter.
- **Ultra-cheap fallback and per-keystroke query encoding:** `potion-multilingual-128M` (MIT, static, 256d). Expect a large quality drop: MMTEB 47.3 vs e5-small 55.5.

**Ablation set:**
- bge-m3 (adds sparse vectors for Hinglish and lexical match);
- snowflake-arctic-embed-m-v2.0;
- e5-large-instruct as a quality ceiling.

**Excluded:** jina-v3 (NC), Vyakyarth (licence), granite multilingual (no Hindi/Tamil), MuRIL and IndicBERT (not sentence embedders), English-only models for Indic stores.

---

## 6. Learning-to-rank for blending candidate sources

**Evidence:**
- **Qin et al., "Are Neural Rankers still Outperformed by Gradient Boosted Decision Trees?"** ICLR 2021 (Google). PDF fetched: https://storage.googleapis.com/gweb-research2023-media/pubtools/5954.pdf. Abstract page: https://research.google/pubs/are-neural-rankers-still-outperformed-by-gradient-boosted-decision-trees/
  - Recent neural LTR models trailed LightGBM λMART "by a large margin".
  - Their DASALC needs log1p transforms, Gaussian-noise augmentation, self-attention, latent cross and ensembling to match or slightly beat it.

  | Table 3, NDCG@5 | Web30K | Yahoo | Istella |
  |---|---|---|---|
  | λMART (LightGBM) | 49.66 | **74.21** | 71.24 |
  | SetRank | 45.15 | 70.29 | 63.45 |
  | DASALC | 50.92 | 73.76 | 70.06 |
  | DASALC-ens | **51.72** | 74.07 | **71.32** |

  - Ablation on Istella NDCG@5: plain DNN 64.72, then 67.09 with log1p, then 71.32 with all components.
  - **Takeaway:** on tabular ranking features, a tuned GBDT is the bar. Matching it with a neural ranker takes several non-trivial tricks plus an ensemble.
- **Jeunen et al. (ShareChat), "On GBDTs and Neural Rankers"** FIRE 2023. https://arxiv.org/html/2312.01760v1
  - About 2B training rows and around 500 features. MMoE beat CatBoost only slightly (Like ROC-AUC 0.9420 vs 0.9388).
  - Gains came from embedding-history features and more data.
  - The authors say GBDTs "stay competitive on smaller datasets and at lower cost". **Small stores are the GBDT regime.**
- **Kaggle winners** (first-party write-ups fetched through Kaggle's discussion API):
  - **OTTO 1st place (mrkmakr, 2023-02-06).** https://www.kaggle.com/competitions/otto-recommender-system/discussion/384022. The Kaggle Solution Report docx is in github.com/mrkmakr/OTTO-Multi-Objective-Recommender-System/document.
    - Candidates: about 1,200 per session from visited items, several co-visitation matrices (applied iteratively "like beam search") and MLP/transformer two-tower NNs.
    - Ranker: **LGBMRanker**. A single model scored LB 0.604; an average of 9 rankers with different hyperparameters scored 0.605. NN scores are stacked as features.
    - Features: about 200, pruned to about 100 per target by gain importance.
    - Negative sampling: 5% clicks, 25% carts, 40% orders.
    - Ablation (weighted R@20): full 0.5884; without NN 0.5831; without co-visitation 0.5831; **single NN only 0.5151**.
    - A "simple" version (1 NN, 4 co-visitation matrices, 1 LGBM) scored 0.5940 vs 0.6050 final, taking 2 days vs 2 weeks.
  - **OTTO 3rd-place team** (Chris Deotte's part). https://github.com/cdeotte/Kaggle-OTTO-Comp
    - 100 candidates per target from co-visitation + rules, then a GBT reranker picks 20.
    - XGBoost on GPU (RAPIDS). The README notes CatBoost as a stronger later alternative.
  - **H&M 1st place (senkin13 & h4211819, 2022-05-10).** https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324070
    - "Candidate generation strategy is the key", with 100 candidates per user from popularity, repurchase, item-CF, word2vec and ProNE graph embeddings.
    - **Best single model was a LightGBM *classifier*** (CV 0.0441, LB 0.0367). Final ensemble: 5 LGBM + 7 CatBoost classifiers (LB 0.0371). "CatBoost's LB score is much worse than LightGBM".
    - Negatives downsampled to 1–2M per week.
    - **"tried to add cold start items but they will never be ranked to top12 due to lacking of interaction information … image and text features didn't help but should be useful for cold start."**

**Library docs** (raw docs fetched from GitHub master):
- LightGBM `lambdarank`: label_gain defaults to 2^i−1; `lambdarank_truncation_level` default 30; `lambdarank_norm`; position-bias regularisation. `rank_xendcg` is "faster than and achieves the similar performance as lambdarank". https://github.com/microsoft/LightGBM/blob/master/docs/Parameters.rst
- XGBoost `rank:ndcg` (LambdaMART, the default), `rank:pairwise` and `rank:map`:
  - `lambdarank_pair_method` = topk (default since 2.0) or mean;
  - `lambdarank_num_pair_per_sample`;
  - unbiased LTR since 2.0.0.
  - https://github.com/dmlc/xgboost/blob/master/doc/tutorials/learning_to_rank.rst
- CatBoost: YetiRank / YetiRankPairwise (GPU only in Classic mode); LambdaMart and StochasticRank are CPU-only and optimise NDCG/MAP/MRR directly. https://catboost.ai/docs/en/concepts/loss-functions-ranking
- macOS arm64 wheels on PyPI (checked today): lightgbm 4.7.0, xgboost 3.4.1 (requires Python ≥ 3.12), catboost 1.2.10.

**Choice:**
- **LightGBM `lambdarank`** (with `rank_xendcg` as a speed ablation), grouped by request (user/session or anchor item).
- Candidate sources become features: per-source score **and rank**, and a source-membership one-hot; this is the OTTO pattern. Add item popularity over several windows, recency, price and category match, embedding cosine, and complement-graph edge score.
- Also train a **binary LightGBM classifier** baseline, since H&M's winner found the classifier best. Report both.

**Alternatives:**
- XGBoost `rank:ndcg` (topk) and CatBoost YetiRank, as a one-off robustness check. I found no head-to-head public benchmark of the three, so do not claim one is better.

**Caveats:**
- Cold items get no help from interaction features (H&M quote).
- Give the ranker content features (embedding similarity to the user's recent items, a "days since launch" feature) and consider **slot reservation or exploration** for new items. Otherwise the ranker learns to bury them.
- Small stores have few positives per day, so pool weeks and use time-based validation (the OTTO winner followed Radek's local-CV setup).

---

## 7. ANN libraries on CPU and when brute force is enough

**Library status** (GitHub API and PyPI JSON, checked 2026-10-19):

| Library | Licence | Latest on PyPI (date) | macOS arm64 wheel on PyPI | Activity |
|---|---|---|---|---|
| FAISS (`faiss-cpu`) | MIT | 1.15.1 (2026-09-16) | Yes: `cp310-abi3-macosx_14_0_arm64` (needs macOS ≥ 14) | Pushed 2026-10-18 |
| USearch (`usearch`) | Apache-2.0 | 2.26.5 (2026-10-18) | Yes (12 wheels) | Pushed 2026-10-18 |
| hnswlib | Apache-2.0 | **0.8.0 (2023-12-03), sdist only** | **No.** Compiles from source | GitHub v0.9.0 on 2026-03-28 and v0.10.0-rc.2 on 2026-09-14, but not on PyPI |
| Spotify Voyager (hnswlib-based) | Apache-2.0 | 2.2.2 (2026-10-16) | Yes | Pushed 2026-10-17 |

**Brute force, measured** on the target M1 (numpy 2.5.2, float32, L2-normalised, `X @ q` + `argpartition` top-20; one run, indicative):

| Items × dims | Matrix size | Single query (ms) | Batch of 256 (ms per query) |
|---|---|---|---|
| 10K × 384 | 14.6 MiB | 0.74 | 0.16 |
| 50K × 384 | 73 MiB | 3.2 | 0.98 |
| 50K × 768 | 147 MiB | 11.3 | 2.2 |
| 50K × 1024 | 195 MiB | 8.0 | 1.4 |
| 200K × 384 | 293 MiB | 13.0 | 8.2 |

The 768 vs 1024 inversion is run-to-run noise or BLAS blocking. Treat all values as ±2×.

**Choice:**
- **Exact numpy or BLAS brute force for every catalog up to about 50K items** at ≤ 1024d. It costs ≤ about 10 ms per query on an M1, returns exact results, has no index to build or keep in sync with catalog updates, and needs no extra dependency.
- Item-item neighbour tables (offline) are one batched matmul.
- Add an ANN only when the catalog exceeds about 100–200K or p99 latency demands it:
  - **USearch** first: Apache-2.0, arm64 wheels, active, small footprint.
  - **FAISS** if PQ compression or IVF is needed (Arctic's card suggests `pq256x4fs` for 128-byte vectors).
- **Avoid hnswlib from PyPI:** stale wheels and an sdist-only install on Apple Silicon.

**Caveats:**
- If MRL-truncated 256d vectors are used, brute force gets about 3–4× cheaper again.
- int8 or binary quantisation (mxbai and Arctic cards) is a further option, but it costs recall that should be measured.

---

## Recommendations for Mitate

1. **Cold-start content channel = LLM-derived text embeddings, not an LLM recommender.**
   - Qwen3-Embedding-0.6B (Apache-2.0) over `Title; Category; Brand; Attributes; Description`, MRL-truncated to 512d, optionally whitened.
   - It drives: (a) item-item kNN for new SKUs; (b) the semantic regulariser in L3AE-style EASE; (c) cosine features for the ranker.
   - Indic-heavy or very low-CPU stores swap in multilingual-e5-small (MIT). potion-multilingual-128M is the zero-cost fallback.
2. **The paper's "LLM-based cold-start baseline"** should be that embedding pipeline, plus two ablations:
   - an LLM-profiled catalog text (local 3–4B model via Ollama, offline, once per item);
   - small-LLM zero-shot reranking as a negative control, given Kusano 2025 and Dong 2026.
   - Do **not** adopt the Frontiers small-LLM method: wrong notion of cold start, weak evidence.
3. **Complements:**
   - Generate a **category/type-level** complement graph offline with a local LLM (P-Companion structure, Walmart pipeline shape).
   - Evaluate it with the FBL-based rubric in §4b: human κ on a sample, a k = 7 majority-vote LLM judge validated against humans, lift-based behavioural check, and graph sanity checks.
   - Map types to SKUs with embeddings plus attribute-compatibility filters.
   - No per-request LLM. 2507.16237's 70B reranker is out of budget and has no cost data.
4. **Blending:**
   - LightGBM `lambdarank`, grouped per request.
   - Features: per-source score and rank, popularity windows, recency, content cosine, complement edge score.
   - Baselines: a LightGBM binary classifier and a fixed-weight blend.
   - Reserve slots or explore for new items, because interaction-driven rankers bury cold items.
5. **Retrieval index:** exact numpy matmul up to about 50K items. USearch beyond that. Skip hnswlib.
6. **Hinglish:** build a 200–500-query in-house test set before final embedder selection, and keep a lexical or sparse channel (BM25 or bge-m3 sparse) alongside dense.

---

## Could not verify

- **Small-LLMs paper:** LoRA rank, alpha, learning rate and epochs (not in the article). The history length at which **SASRec** overtakes the LLM (only GRU4Rec was tested). Any CPU latency.
- **2507.16237:** whether Llama-3.3-70B ran locally or via API; token counts, latency and cost (none reported).
- **L3AE:** any cold-start (zero-interaction item) results (none in the paper). Performance with sub-1B embedders.
- **WhitenRec (2402.10602):** text encoder, datasets and numbers, since the abstract page has none. ICDE 2024 venue appeared only in a search snippet, not on the fetched page.
- **Recformer, LLMRec, Wang et al. 2402.11724, ColdLLM, Hou et al.:** numeric results (only abstract pages fetched). The LLM used by LLMRec and ColdLLM is not named on the abstract pages.
- **McAuley 2015:** the KDD venue and result numbers (the arXiv page lists no venue or metrics).
- **COSMO:** the "up to 60%" improvement figure from a third-party summary (unverified; not used).
- **CCGen, KG-completion (2305.09858):** numeric results.
- **Kusano et al.:** the paper evaluates **user** cold start only. I used only its narrow cold-start table.
- **MTEB(Indic) scores for Qwen3-Embedding, EmbeddingGemma, bge-m3, Arctic v2 and nomic v2:** not in the MMTEB tables I read. The live leaderboard was not fetched.
- **bge-m3 parameter count:** not in its HF card or API. 568M is taken from the Arctic and Nomic cards' comparison tables.
- **MuRIL and IndicSBERT parameter counts:** computed from `config.json` (vocab 197,285 × 768 + BERT-base encoder ≈ 237M), not stated by the authors. IndicSBERT's MuRIL base is inferred from the identical vocabulary size.
- **NV-Embed-v2 licence:** not re-fetched in this review.
- **Sarvam:** no open embedding model found. Whether Sarvam offers a hosted embedding API was not checked (and is out of scope: no paid API).
- **An e-commerce Hinglish retrieval benchmark:** none found.
- **A head-to-head public benchmark of LightGBM lambdarank vs XGBoost rank:ndcg vs CatBoost YetiRank:** none found.
- **CPU throughput of Qwen3-Embedding-0.6B, multilingual-e5-small and potion models on the M1:** downloads stalled, so they were not measured. Only bge-small was measured.
- **Kaggle pages:** they render only a title through WebFetch. Write-up text was read through Kaggle's `GetForumTopicById` endpoint and the winner's GitHub docx.
- **OpenReview:** blocked by a browser check. Qin et al. was read from the Google-hosted PDF instead.
