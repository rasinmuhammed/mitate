# 05 — Datasets, library and model licenses

Review date: **2026-10-19**. Scope: the Mitate offline benchmark, to be published (arXiv preprint and an open-source repo), and the Mitate product, which may be used commercially.
Machine budget: 8 GB RAM, ~20 GB free disk.

Method: every license line below was read from a primary source: the dataset page, the Kaggle competition rules (pulled as raw text from Kaggle's page API), HF cards, GitHub LICENSE files, or PyPI metadata. "UNCLEAR" means no primary statement could be found, and the reason is given. Third-party re-uploads (Kaggle mirrors) are **never** treated as the licensor: a re-uploader's "CC0" tag on someone else's data has no legal effect.

This is engineering due diligence, not legal advice. Before any commercial launch, have counsel review the NC (non-commercial) datasets and the DPDP items.

---

## 0. TL;DR

| Asset | License (exact) | Benchmark / publication | Commercial use | Redistribute derived samples (IDs/seeds) | Source |
|---|---|---|---|---|---|
| **OTTO** | CC BY 4.0 (data); MIT (code) | Yes | **Yes** | **Yes** (attribution) | github.com/otto-de/recsys-dataset ; kaggle.com/datasets/otto/recsys-dataset |
| **H&M** (Kaggle) | Kaggle rules: "Non-Commercial Purposes & Academic Research" | Yes (academic) | **No** | **No**: seeds and code only | kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules |
| RelBench rel-hm | Inherits the Kaggle H&M rules (RelBench's own layer is CDLA-Permissive-2.0) | Yes (academic) | No | No | huggingface.co/datasets/stanford-star/relbench-v1 |
| **Amazon Reviews 2023** | No license ("not in a position to assign a license") | Yes (research) | UNCLEAR, so treat as No | UNCLEAR: seeds only | amazon-reviews-2023.github.io ; HF discussion #1 |
| ↳ `bought_together` | **Empty in 100 % of items** (verified) | n/a | n/a | n/a | GH issue hyp1231/AmazonReviews2023#48 |
| Amazon 2018 (Ni et al.) `also_buy` | No license; "cite … if you use the data in any way" | Yes (research) | UNCLEAR, so treat as No | UNCLEAR: seeds only | nijianmo.github.io/amazon |
| **Instacart 2017** | "provided as-is for non-commercial use … subject to our Terms and Conditions" | Yes (research) | **No** | UNCLEAR: seeds only | tech.instacart.com (2017 post) |
| Synerise RecSys'25 | CC BY-NC 4.0 | Yes | No | Yes, NC + attribution | recsys.synerise.com/summary |
| **Amazon-M2** (KDD Cup '23) | Apache 2.0 (per the paper's own License section) | Yes | **Yes** (as stated by authors) | Yes (license and notice) | arXiv 2307.09688 §B.3 |
| RetailRocket | CC BY-NC-SA 4.0 (posted by the Retailrocket owner) | Yes | No | Yes, NC + ShareAlike | kaggle.com/datasets/retailrocket/ecommerce-dataset |
| Coveo SIGIR eCom'21 | Coveo T&C: "only for non-commercial research and educational purposes" | Yes (attribution) | No | **No** (not even to colleagues) | github.com/coveooss/SIGIR-ecom-data-challenge |
| Dunnhumby Complete Journey | UNCLEAR (general site T&C: "research, personal or non-commercial") | Probably yes | Treat as No | UNCLEAR | dunnhumby.com/source-files |
| Ta-Feng | UNCLEAR (no licensor statement found) | Risky | No | No | — |
| Tmall / Ali-CCP (Tianchi) | UNCLEAR (Tianchi pages not readable without login) | Risky | No | No | tianchi.aliyun.com |
| Diginetica (CIKM Cup '16) | UNCLEAR | Risky | No | No | competitions.codalab.org/competitions/11161 |
| Yoochoose (RecSys '15) | UNCLEAR | Risky | No | No | recsys.acm.org/recsys15/challenge |
| MovieLens-20M | GroupLens usage license | Yes (must cite) | **No** without permission | **No** without permission | files.grouplens.org/datasets/movielens/ml-20m-README.html |
| Million Song Dataset | Echo Nest API terms + MusicBrainz tags CC BY-NC-SA 2.0 | Yes | UNCLEAR | UNCLEAR | millionsongdataset.com/faq |
| Indian Kaggle sets (Flipkart / Myntra / BigBasket) | Uploader-claimed CC0 / CC BY-SA / CC BY-NC-SA on **scraped catalogs** | Catalog only | No (claims invalid) | No | see §1.7 |
| GRU4Rec official (PyTorch) | Original GRU4Rec license: "Non-commercial use only" | Yes (cite 2 papers) | **No** (paid license) | Code redistribution is research/edu only | github.com/hidasib/GRU4Rec_PyTorch_Official |
| sansa | Apache-2.0, **but it needs scikit-sparse → CHOLMOD (GPL-2.0+ modules)** | Yes | Yes, with GPL caveat if distributed | n/a | github.com/glami/sansa |
| Misata | MIT | Yes | Yes | Yes | github.com/rasinmuhammed/misata |
| Other libraries (numpy … statsmodels) | BSD / MIT / Apache-2.0 / PSF-style | Yes | Yes | Yes | §2 |
| Qwen3 / Qwen3.5 ≤ 9B | Apache-2.0 | Yes | Yes | Yes | HF cards |
| Phi-4-mini | MIT | Yes | Yes | Yes | HF card |
| Granite 4.0 / 3.3 | Apache-2.0 | Yes | Yes | Yes | HF cards |
| SmolLM3-3B | Apache-2.0 | Yes | Yes | Yes | HF card |
| Ministral 3 (2512) 3B/8B | Apache-2.0 | Yes | Yes | Yes | HF cards |
| Ministral-8B-2410 | Mistral Research License (MRL-0.1) | Research only | **No** | No | mistral.ai/licenses/MRL-0.1.md |
| Llama 3.1 / 3.2 | Llama Community License | Yes | Yes (with conditions) | Yes (with conditions) | llama-models repo |
| Gemma 3 | Gemma Terms of Use | Yes | Yes (with conditions) | Yes (with conditions) | ai.google.dev/gemma/terms |
| Gemma 4 | Apache-2.0 (HF card) | Yes | Yes | Yes | huggingface.co/google/gemma-4-E4B-it |

### Recommended benchmark set

| Role | Dataset | Why | Constraint |
|---|---|---|---|
| Session-heavy, general e-commerce (**primary**) | **OTTO** | 12.9M sessions, clicks/carts/orders, CC BY 4.0: the only large real dataset that is cleanly usable commercially and redistributable | 11 GB JSONL: stream-convert to Parquet (see §1.1) |
| Fashion | **H&M** | Real fashion purchases with rich article text and metadata, and a strong community baseline | NC/academic only, no redistribution. Keep a **firewall**: no product defaults or weights tuned on H&M ship in Mitate |
| Grocery / repeat purchase | **Instacart 2017** | Canonical repeat-basket dataset, 3M orders, aisle/department taxonomy | NC only. The Kaggle competition page now returns **404**, and the original S3 returns 403. Pin a mirror by SHA-256 and cite Instacart |
| Item text, multilingual, cold-start, LLM complement graph | **Amazon-M2** | Sessions plus product titles/brand/desc in 6 locales; Apache 2.0 per the authors | No timestamps; AIcrowd login needed |
| Complement ground truth (eval only) | **Amazon 2018 `also_buy`** (Grocery, Home & Kitchen, Clothing) | The 2023 `bought_together` field is empty | Research use only; license UNCLEAR |
| Classic EASE reproduction | **MovieLens-20M** | Standard EASE / SANSA reference numbers | No redistribution; no commercial use without permission |

**Exclude:** Amazon 2023 `bought_together` as ground truth (empty field). Synerise as a core dataset (NC, and its download URL returns 404). Coveo (no redistribution to anyone, NC). Ta-Feng, Tmall, Ali-CCP, Diginetica, Yoochoose (license UNCLEAR). All "Indian" Kaggle sets (scraped catalogs, no interactions, invalid license claims). The GRU4Rec official code and Ministral-8B-2410 must stay out of the **product**; GRU4Rec is fine as a research baseline in the benchmark.

**Release policy for the benchmark repo (all datasets):** publish download scripts, dataset version and SHA-256, deterministic sampling code, and seeds. Publish actual sampled IDs (e.g. "store" partitions as session/aid lists) **only for OTTO (CC BY 4.0) and Amazon-M2 (Apache 2.0)**, with attribution and license files.

---

## 1. Datasets

### 1.1 OTTO Recommender Systems Dataset

- **URLs:** https://github.com/otto-de/recsys-dataset · dataset https://www.kaggle.com/datasets/otto/recsys-dataset · competition https://www.kaggle.com/competitions/otto-recommender-system
- **License (GitHub README):** "The OTTO dataset is released under the [CC-BY 4.0 License], while the code is licensed under the [MIT License]." Kaggle dataset metadata (API): `licenseName: "Attribution 4.0 International (CC BY 4.0)"`. The repo `LICENSE` file is MIT, © 2022 Otto (GmbH & Co KG), and covers code only.
- **Competition rules (Kaggle, §7.A):** "*Competition Use and Commercial*: You may access and use the Competition Data for any purpose, whether commercial or non-commercial, including for participating in the Competition and on Kaggle.com forums, and for academic research and education." The rules also say: "The Competition Data is also subject to the following terms and conditions: CC-By 4.0." Note that §7.B (data security: "not to … redistribute … the Competition Data to any party not participating") binds the **competition** copy. The **dataset** copy (`otto/recsys-dataset`) is plain CC BY 4.0. **Use the dataset copy.**
- **Benchmark / publication:** yes. **Commercial:** yes. **Redistribution of derived samples:** yes, under CC BY 4.0 (attribution, link to the license, indicate changes). Cite DOI 10.34740/KAGGLE/DSV/4991874.
- **Files and sizes (Kaggle API, `kaggle datasets files otto/recsys-dataset`):** `otto-recsys-train.jsonl` **11 GB**, `otto-recsys-test.jsonl` **716 MB**. Kaggle `totalBytes` = 12,057,962,667 (~12.06 GB, uncompressed). The zip download size was not verified.
- **Kaggle dataset vs competition download:** the competition shipped `train.jsonl` + **truncated** `test.jsonl` (labels hidden) + `sample_submission.csv`. KAGGLE.md: "The final test set has been published following the conclusion of the Kaggle competition. You can access the complete dataset here [otto/recsys-dataset]." The repo's `testset.py` regenerates truncated test sets and labels, and `evaluate.py` computes Recall@20. The README now recommends "Evaluating on the **entire test sequences** without truncation". The statistics are identical (Train 12,899,779 sessions / 1,855,603 items / 216,716,096 events; Test 1,671,803 / 1,019,357 / 13,851,293).
- **Schema:** `{session, events:[{aid, ts, type}]}` with `type ∈ {clicks, carts, orders}`. `ts` is documented as "the Unix timestamp". The README example `1661200010000` is 13 digits, i.e. **milliseconds** (inferred from the example). There is **no item text or metadata**: "The dataset contains only anonymized IDs, with no item or user metadata" (KAGGLE.md FAQ).
- **Date range:** "4 weeks" train + "the following week" test. The official pages give no calendar dates; community analyses place it at roughly 2022-07-31 → 2022-09-04 UTC (UNVERIFIED; derive from `ts` at load time).
- **Access:** `kaggle datasets download -d otto/recsys-dataset -f otto-recsys-test.jsonl` (no rule acceptance needed for the dataset copy).
- **8 GB / 20 GB practicality:** never materialize the 11 GB JSONL. Stream it (`unzip -p … | python -I convert.py`) into Parquet with `session:int32, aid:int32, ts:int32 (seconds), type:uint8`. 216.7M rows × 13 B ≈ 2.8 GB on disk, less with zstd. Sample sessions or "stores" in DuckDB or Polars-lazy.

### 1.2 H&M Personalized Fashion Recommendations (+ RelBench rel-hm)

- **URL:** https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations (rules page pulled verbatim via Kaggle's page API; competition id 31254; `rulesRequired: true`)
- **License header:** "**DATA ACCESS AND USE**: Non-Commercial Purposes & Academic Research". Kaggle metadata: `license: "Subject to Competition Rules"`.
- **Operative clause (§7.A):** "*Competition Use and Non-Commercial & Academic Research*: You may access and use the Competition Data for non-commercial purposes only, including for participating in the Competition and on Kaggle.com forums, and for academic research and education."
- **Redistribution (§7.B):** "You agree not to transmit, duplicate, publish, redistribute or otherwise provide or make available the Competition Data to any party not participating in the Competition."
- **Benchmark / publication:** yes, as academic research (results, plots, aggregate statistics). **Commercial:** **no**. **Derived samples:** **no**. Customer or article ID lists *are* Competition Data, so release only the seeds and the sampling code; users must download from Kaggle after accepting the rules.
- **Commercial-firewall note:** Mitate is commercial. Publishing a benchmark is defensible as academic research. Shipping hyper-parameters, priors or weights *fitted on H&M* inside the paid product is not. Keep H&M results in `benchmarks/` only.
- **Schema:** `transactions_train.csv` (`t_dat` date, no time-of-day; `customer_id`; `article_id`; `price`; `sales_channel_id`). `articles.csv` (rich product metadata including product/colour/section names and a text description). `customers.csv` (age, club/fashion-news fields). `images/` (not all articles have one). Purchases only, with no views or carts.
- **Date range / size:** ~Sep 2018 → Sep 2020; ~31M transactions, ~1.3M customers, ~105k articles (secondary: arXiv 2506.21368). `transactions_train.csv` ≈ **3.49 GB** (secondary: prefgraph loader docs). The image folder makes the full zip tens of GB (UNVERIFIED), so download only the CSVs: `kaggle competitions download -c h-and-m-personalized-fashion-recommendations -f transactions_train.csv` (and `-f articles.csv`, `-f customers.csv`). This requires accepting the rules in the browser first.
- **RelBench `rel-hm`:** the RelBench card (https://huggingface.co/datasets/stanford-star/relbench-v1) lists rel-hm as "Kaggle competition rules: non-commercial and academic research only, no redistribution", and states that each database "keeps the license of its upstream source". RelBench's own layer ("the conversion to this format, schemas, task definitions, splits and statistics") is "CDLA-Permissive-2.0", and the RelBench code is Apache-2.0 (GitHub). So rel-hm grants **no additional rights**. (A secondary search snippet mentions a CC BY-SA 4.0 tag on an older HF listing. That would not override the upstream Kaggle terms. Ignore it.)

### 1.3 Amazon Reviews 2023 (McAuley Lab), and why its `bought_together` is unusable

- **URLs:** https://amazon-reviews-2023.github.io/ · https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- **License:** **none**. The site shows only "Copyright © 2024, McAuley Lab", and the HF card has no `license` field. The maintainer (hyp1231) wrote in HF discussion #1 (2024-03-08): "we are not in a position to assign a license to this dataset or dictate the terms of its usage. We have made the dataset available primarily for research purposes." (The data is scraped Amazon content, so Amazon's rights also apply.)
- **Benchmark:** yes, research use with citation (Hou et al., 2024). **Commercial:** UNCLEAR, so treat as **no**. **Redistribution of derived samples:** UNCLEAR, so release seeds and code only. Do not re-host review text.
- **Size / schema / dates:** 571.54M reviews, 54.51M users, 48.19M items, 33 categories, May 1996 → Sep 2023. Total **750 GB** (HF card). Per category examples (HEAD on mcauleylab.ucsd.edu): `meta_Grocery_and_Gourmet_Food.jsonl.gz` 324 MB, `meta_Clothing_Shoes_and_Jewelry.jsonl.gz` 4.05 GB, `Grocery_and_Gourmet_Food.jsonl.gz` (reviews) 1.63 GB. Reviews: `rating, title, text, images, asin, parent_asin, user_id, timestamp (ms), verified_purchase, helpful_vote`. Metadata: `main_category, title, average_rating, rating_number, features, description, price, images, videos, store, categories, details, parent_asin, bought_together`.
- **`bought_together` IS EMPTY.** The site documents it as "Recommended bundles from the websites", but the sample record shows `"bought_together": null`.
  - GitHub issue **hyp1231/AmazonReviews2023#48** ("bought_together is always None", 2026-05-07) scanned all 34 metadata files and found **0** non-null values in every category. It is still open with no maintainer reply.
  - HF discussion **#4** ("Missing 'bought_together' information", 2024-06-28): "the 'bought_together' information is missing for all of them". It references an earlier GitHub report from March 2024 and is unresolved.
  - **Independently verified today** by streaming the files: `Magazine_Subscriptions` 0 / 3,391; `All_Beauty` 0 / 112,590; `Grocery_and_Gourmet_Food` 0 / 603,274 non-empty.
  - **Conclusion:** there is no complementary ground truth in the 2023 release.
- **Best alternative complement ground truth: Amazon 2018 (Ni, Li, McAuley, EMNLP 2019)**, https://nijianmo.github.io/amazon/ (mirror: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
  - License: **none stated**. The only condition is "Please cite the following paper if you use the data in any way". Same research-only posture as above. The 2018 page says "Please see the 2023 version of this dataset".
  - Fields: `also_buy`, `also_view` (the page's prose lists "also bought, also viewed, bought together, buy after viewing"), plus `title, feature, description, price, brand, categories, rank, imageURL(HighRes)`. 15.5M products, reviews May 1996 → Oct 2018.
  - **Availability verified:** per-category metadata downloads directly (HTTP 200, no form), e.g. `metaFiles2/meta_Grocery_and_Gourmet_Food.json.gz` 110 MB, `meta_Clothing_Shoes_and_Jewelry.json.gz` 1.57 GB, `meta_Home_and_Kitchen.json.gz` 1.77 GB. (The complete files go through a form.)
  - **Population measured today** (Grocery): 287,051 items; `also_buy` non-empty for **83,294 (29.0 %)**, mean 32.5 ASINs per populated item; `also_view` non-empty for 120,938.
  - Caveat: `also_buy` is "customers who bought X also bought Y". That is co-purchase, which mixes complements with same-need substitutes. Following McAuley et al. (KDD 2015), treat `also_buy`/`bought_together` as complement-ish and `also_view` as substitute-ish. Filter same-leaf-category pairs if you need "true" complements.
  - **License-clean complement proxy:** co-`orders` / co-`carts` within OTTO sessions (CC BY 4.0, commercially usable) and co-basket pairs in Amazon-M2 sessions. Neither has text in OTTO's case, so evaluate the LLM-generated category-complement graph on Amazon 2018 (research only) and on Amazon-M2 / H&M / Instacart via category-level co-purchase lift.

### 1.4 Instacart Online Grocery Shopping Dataset 2017

- **Original terms** (Jeremy Stanley, "3 Million Instacart Orders, Open Sourced", tech.instacart.com, 2017-05-03; read via the Wayback Machine because the live site returns 403): "The dataset is provided as-is for non-commercial use, and can be downloaded from S3 at: https://www.instacart.com/datasets/grocery-shopping-2017 and is subject to our Terms and Conditions." Requested citation: "The Instacart Online Grocery Shopping Dataset 2017", Accessed from https://www.instacart.com/datasets/grocery-shopping-2017 on <date>.
- **Benchmark:** yes (non-commercial research). **Commercial:** **no**. **Derived samples:** UNCLEAR (the T&C text was not retrievable), so release seeds only.
- **Access status (checked 2026-10-19):** `instacart.com/datasets/grocery-shopping-2017` → 404. The S3 tarball returns 403. **The Kaggle competition `instacart-market-basket-analysis` returns 404** (both the web page and the Kaggle API). Mirrors exist, e.g. `psparks/instacart-market-basket-analysis`, which Kaggle tags "CC0: Public Domain". **That tag is the uploader's claim and is invalid**; Instacart's non-commercial terms still apply. Pin a mirror by SHA-256 in the benchmark and state the provenance.
- **Schema / size (mirror listing):** `orders.csv` 104 MB (`order_id, user_id, eval_set, order_number, order_dow, order_hour_of_day, days_since_prior_order`), `order_products__prior.csv` 551 MB and `__train.csv` 24 MB (`order_id, product_id, add_to_cart_order, reordered`), `products.csv` 2 MB (`product_name, aisle_id, department_id`), `aisles.csv`, `departments.csv`. Total ≈ 713 MB uncompressed. More than 3M orders from more than 200k users, with 4–100 orders per user. **No absolute timestamps**: only day-of-week, hour, and days since the prior order. Product names are available as text.

### 1.5 Synerise RecSys Challenge 2025 ("Universal Behavioral Modeling Dataset")

- **License (https://recsys.synerise.com/summary):** "Universal Behavioral Modeling Dataset © 2025 by Synerise SA is licensed under Creative Commons Attribution-NonCommercial 4.0 International." The challenge code (github.com/Synerise/recsys2025) is MIT.
- **Benchmark:** yes. **Commercial:** no. **Derived samples:** yes, under CC BY-NC 4.0 (NC + attribution).
- **Schema:** Parquet files `product_buy` (1,682,296 events), `add_to_cart` (5,235,882), `remove_from_cart` (1,697,891), `page_visit` (150,713,186), `search_query` (9,571,258), plus `product_properties` (`sku, category, price` bucket, `name` = product-quantized text embedding, **no raw text**). Target windows are 14 days. Calendar range: not stated.
- **Access:** `https://data.recsys.synerise.com/dataset/ubc_data/ubc_data.tar.gz` returns **HTTP 404 as of 2026-10-19**. Availability is UNCLEAR.

### 1.6 Alternatives

| Dataset | License evidence (primary) | Commercial | Notes |
|---|---|---|---|
| **Amazon-M2** (KDD Cup 2023) | Paper §B.3 "License": "The Amazon-M2 dataset can be freely downloaded at https://www.aicrowd.com/challenges/amazon-kdd-cup-23-multilingual-recommendation-challenge/… and used under the license of Apache 2.0." The AIcrowd page itself names no license (it links to its Terms of Use). | Yes (author statement) | 3,606,249 sessions; DE 1.11M, JP 0.98M, UK 1.18M, ES/FR/IT ~0.1M each. ~1.53M products with `title, price, brand, color, size, model, material, author, desc`. Sessions are **ordered lists without timestamps**. AIcrowd login required. Size not verified. |
| **RetailRocket** | Kaggle `retailrocket/ecommerce-dataset`, owner "Roman Zykov" (Retailrocket): **CC BY-NC-SA 4.0** | No | 2,756,101 events (2,664,312 view / 69,332 addtocart / 22,457 transaction), 1,407,580 visitors, 4.5 months (ms timestamps ≈ mid-2015). All property values hashed except `categoryid` and `available`, so **no usable text**. events.csv 90 MB; item_properties 462 + 390 MB. ShareAlike would bind any released derived data. |
| **Coveo SIGIR eCom 2021** | `Terms & Conditions.txt`: "1. Permitted Use. You will use the Dataset only for non-commercial research and educational purposes." "5. Redistribution. You will not distribute copies of the Dataset and/or data contained therein to any third party, which includes but is not limited to your research associates and colleagues." "6. Attribution…" | No | ~5M sessions, 36M events (pageview/detail/add/remove/purchase) + 800k search events. `sku_to_content` has hashed category, price decile, dense `description_vector`/`image_vector` (no raw text). Sign-up form. Timestamps shifted. Quebec law. **Exclude** (others cannot reproduce without signing; no redistribution). |
| **Dunnhumby "The Complete Journey"** | https://www.dunnhumby.com/source-files/: direct downloads, no license on the page. Site T&C (general "material"): "the material is used solely for research, personal or non-commercial purposes;" and "the material is not modified or altered in anyway;". Whether these apply to datasets is **UNCLEAR**. The Kaggle mirror's "ODbL" tag is the uploader's claim. The `completejourney` R package is CC0 but covers package code (and 84.51° data). | Treat as No | Grocery baskets + demographics + campaigns/coupons. The 84.51° "Complete Journey 2.0" version (via the R package) has 2,469 households, "over one year", 1,469,307 transaction rows; the dunnhumby original's stats were not verified. The "not modified" clause conflicts with derived samples. |
| **Ta-Feng** | No licensor statement found. The Kaggle mirror "CC0" is by uploader `chiranjivdas09` (not the owner). **UNCLEAR.** | No | Nov 2000 – Feb 2001, 61 MB CSV. |
| **Tmall (IJCAI-15) / Ali-CCP / Taobao UserBehavior** | Tianchi dataset pages are JS-rendered behind a login and could not be read. Tianchi publishes a general licensing guide but no per-dataset license was found. **UNCLEAR.** | No | Require Alibaba Cloud account + platform agreement. |
| **Diginetica** (CIKM Cup 2016) | No terms on the CodaLab page; a forum question about research use has no visible organizer answer. **UNCLEAR.** | No | Widely used in session-rec papers, but provenance is via Google Drive links. |
| **Yoochoose** (RecSys Challenge 2015) | recsys.acm.org challenge page: no terms; original site 2015.recsyschallenge.com not reachable. **UNCLEAR.** | No | 6 months of clicks/buys, European retailer. |
| **MovieLens-20M** | README "Usage License": "The user may not redistribute the data without separate permission." "The user may not use this information for any commercial or revenue-bearing purposes without first obtaining permission from a faculty member of the GroupLens Research Project at the University of Minnesota." Must acknowledge (Harper & Konstan 2015, TiiS 5(4):19). No endorsement may be implied. | No (without permission) | 20,000,263 ratings, 138,493 users, 27,278 movies, 1995-01-09 → 2015-03-31. `ml-20m.zip` = 198.7 MB. Fine for an EASE reproduction table in the paper. |
| **Million Song Dataset** | FAQ: "The Echo Nest data is released under the same terms of use as their API… if you are a researcher and want to publish results on the dataset, you are fine. If you are a company and are concerned about experimenting on the dataset, send an email to Paul at The Echo Nest." MusicBrainz tags: "Attribution-NonCommercial-ShareAlike 2.0". Code GPL. | UNCLEAR | Music; low relevance to retail. The Taste Profile subset terms were not checked. |

### 1.7 Indian e-commerce datasets

No public dataset with **real Indian shopper interactions** and a clear license was found. What exists on Kaggle are **scraped product catalogs**, licensed by uploaders who do not own the data:

| Kaggle ref | Uploader-claimed license | Content |
|---|---|---|
| `PromptCloudHQ/flipkart-products` | CC BY-SA 4.0 | 20k Flipkart listings (scraped) |
| `djagatiya/myntra-fashion-product-dataset` | CC0 | Myntra women's fashion + images (scraped, 3.07 GB) |
| `surajjha101/bigbasket-entire-product-list-28k-datapoints` | CC BY-NC-SA 4.0 | 28k BigBasket SKUs (scraped) |
| `paramaggarwal/fashion-product-images-dataset` | MIT | 44k Myntra-derived products + images (15.7 GB) |

These licenses are not credible: a scraper cannot grant rights in Flipkart's or Myntra's content, and none of them contain interactions. **Use them at most as a non-redistributed, research-only source of Indian product titles** for Misata synthetic catalogs. Better: have Misata generate the catalogs itself (MIT, own output).

---

## 2. Libraries

PyPI metadata was checked on 2026-10-19 (latest versions shown) and cross-checked with the GitHub `LICENSE` file.

| Library | Version | License | Note |
|---|---|---|---|
| numpy | 2.5.4 | BSD-3-Clause (wheel: `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` for bundled parts) | |
| scipy | 1.18.1 | BSD-3-Clause | |
| pandas | 3.0.6 | BSD-3-Clause | |
| polars | 2.0.0 | MIT | |
| duckdb | 1.5.6 | MIT | |
| scikit-learn | 1.9.1 | BSD-3-Clause | |
| lightgbm | 4.7.0 | MIT | The GitHub repo moved to `lightgbm-org/LightGBM` |
| xgboost | 3.4.1 | Apache-2.0 | |
| implicit | 0.7.3 | MIT | |
| **sansa** | 1.1.0 | Apache-2.0 | **Requires `scikit-sparse` (BSD-2) → SuiteSparse CHOLMOD. CHOLMOD's Check/Cholesky/Utility/Partition modules are LGPL-2.1+, but its *Supernodal, MatrixOps, Modify* modules are GPL-2.0+.** Internal or SaaS use triggers no GPL obligation (GPL-2 has no network clause). **Distributing** a Mitate binary or container that links a default `libcholmod` would pull in GPL. Options: ship EASE/closed-form (numpy) in the product and keep SANSA benchmark-only, or build CHOLMOD with `-DCHOLMOD_GPL=OFF`. |
| torch | 2.14.1 | BSD-3-Clause-style (repo LICENSE); wheel expression `Apache-2.0 AND … BSD-2/3 … BSL-1.0 AND MIT` | |
| optuna | 5.0.0 | MIT | |
| fastapi | 0.146.0 | MIT | |
| pydantic | 2.14.1 | MIT | |
| hnswlib | 0.8.0 | Apache-2.0 | |
| faiss (faiss-cpu) | 1.15.1 | MIT | faiss-cpu wheels are community-built |
| usearch | 2.26.5 | Apache-2.0 | |
| sentence-transformers | 6.1.0 | Apache-2.0 | Each **model** has its own license; check the specific checkpoint |
| model2vec | 0.10.0 | MIT | Same per-model caveat |
| llama.cpp | — | MIT | |
| ollama (server and Python client) | 0.6.3 (py) | MIT | The models pulled through Ollama keep **their own** licenses |
| matplotlib | 3.11.2 | Matplotlib License (PSF-based, BSD-compatible) | |
| statsmodels | 0.15.0 | BSD-3-Clause | |
| **Misata** | — | **MIT** ("MIT License / Copyright (c) 2024 Muhammed Rasin") | |

Operative grants: MIT, "Permission is hereby granted, free of charge, to any person obtaining a copy of this software … to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies…". BSD-3, "Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met". Apache-2.0 §2 grants a "perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license". All are compatible with a commercial closed or open product. Keep a `THIRD_PARTY_NOTICES` file.

### GRU4Rec (official PyTorch), not for the product

- The repo has **no LICENSE file**. README line 12: "**LICENSE:** Final license is coming soon! Meanwhile, use the license of the original, i.e. free for research and education, but contact me for commercial use."
- The original license (github.com/hidasib/GRU4Rec/blob/master/license.txt), © 2015-2020 Balázs Hidasi and Gravity Research & Development Zrt.:
  - "Use and redistribution in source or binary forms for research and educational purposes, with or without modification, are permitted provided that…"
  - "**Non-commercial use only.** Commercial use (including, but not limited to selling the distribution (with or without modifications), using it in commercial systems or for other commercial gain) requires a specific prior written permission from the copyright holders."
  - Both papers must be cited (arXiv 1511.06939, 1706.03847). Commercial licenses are paid ("you will be presented with an offer for the license fee").
- **Your belief is confirmed.** Use it as a **benchmark baseline only**, invoked from a separate script that is not vendored in Mitate's package. For the product, use an independent implementation (e.g. SASRec/GRU written from the paper) or a license-clean library.

---

## 3. Small open LLMs (offline complement-graph generator + cold-start baseline, 8 GB M1 via Ollama)

HF `cardData.license` was checked on 2026-10-19. Ollama sizes are the default Q4_K_M tag sizes from ollama.com/library.

| Model | License | Commercial | Ollama tag / size | Fit on 8 GB M1 |
|---|---|---|---|---|
| **Qwen3-4B** (and -Instruct-2507) | Apache-2.0 | Yes | `qwen3:4b` 2.5 GB | ✅ **recommended default** |
| Qwen3-8B | Apache-2.0 | Yes | `qwen3:8b` 5.2 GB | ⚠️ tight; offline batch only |
| Qwen3-1.7B | Apache-2.0 | Yes | `qwen3:1.7b` 1.4 GB | ✅ |
| Qwen3.5-4B / 9B (2026-02) | Apache-2.0 | Yes | `qwen3.5:4b` 4.0 GB, `:9b` 7.6 GB | 4B ok; 9B ✗ |
| **Phi-4-mini-instruct** (3.8B) | MIT | Yes | `phi4-mini:3.8b` 2.5 GB | ✅ |
| **Granite-4.0-micro / -h-tiny**; Granite-3.3-8B | Apache-2.0 | Yes | `granite4:micro` 2.1 GB | ✅ |
| SmolLM3-3B | Apache-2.0 | Yes | **not in the official Ollama library** (404); GGUF at `ggml-org/SmolLM3-3B-GGUF` (llama.cpp, or Ollama `hf.co/…`) | ✅ |
| Ministral 3 (2512) 3B / 8B | Apache-2.0 | Yes | `ministral-3:3b` 3.0 GB, `:8b` 6.0 GB | 3B ✅, 8B ⚠️ |
| Ministral-8B-Instruct-**2410** | **Mistral Research License** (MRL-0.1) | **No** | — | Exclude |
| Mistral-Small-3.2-24B | Apache-2.0 | Yes | 15 GB | ✗ does not fit |
| Llama 3.2 1B/3B; Llama 3.1 8B | Llama 3.2 / 3.1 Community License | Yes, with conditions | `llama3.2:3b` 2.0 GB, `llama3.1:8b` 4.9 GB | ✅ / ⚠️ |
| Gemma 3 1B/4B | Gemma Terms of Use | Yes, with conditions | `gemma3:4b` 3.4 GB | ✅ |
| Gemma 4 E2B/E4B (2026) | Apache-2.0 (HF card: "License: Apache 2.0") | Yes | `gemma4:e4b` shows 9.5 GB on Ollama | ✗ at the default tag |

Operative terms for the conditional ones:

- **MRL-0.1 §3.2:** "You shall only use the Mistral Models, Derivatives (whether or not created by Mistral AI) and Outputs for Research Purposes." "Research Purposes" means use "solely for (a) personal, scientific or academic research, and (b) for non-profit and non-commercial purposes, and not directly or indirectly connected to any commercial activities or business operations." This excludes Ministral-8B-2410 even for an offline generator whose output ships in the product.
- **Llama 3.2 Community License** (github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE):
  - §1.b.i: "If you distribute or make available the Llama Materials (or any derivative works thereof), or a product or service (including another AI model) that contains any of them, you shall (A) provide a copy of this Agreement … and (B) prominently display 'Built with Llama' … If you use the Llama Materials or any outputs or results of the Llama Materials to create, train, fine tune, or otherwise improve an AI model, which is distributed or made available, you shall also include 'Llama' at the beginning of any such AI model name."
  - §1.b.iii: the notice "Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved."
  - §1.b.iv: Acceptable Use Policy incorporated.
  - **§2:** "If, on the Llama 3.2 version release date, the monthly active users of the products or services made available by or for Licensee, or Licensee's affiliates, is greater than 700 million monthly active users in the preceding calendar month, you must request a license from Meta…"
  - The AUP's EU restriction covers only the **multimodal** 3.2 models.
  - Risk for Mitate: if Llama outputs (a complement graph) are used to "improve an AI model" that is made available, the naming clause arguably bites. Avoid Llama.
- **Gemma Terms of Use** (ai.google.dev/gemma/terms, "Last modified: April 1, 2026"):
  - §3.1 distribution notice: "Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms".
  - §3.2: "You must not use any of the Gemma Services … for the restricted uses set forth in the Gemma Prohibited Use Policy".
  - "Google reserves the right to restrict (remotely or otherwise) usage of any of the Gemma Services".
  - §3.3: "Google claims no rights in Outputs you generate using Gemma."
  - Workable, but the flow-down and remote-restriction terms add avoidable friction. Gemma 4 is Apache-2.0 but too large at the default Ollama tag.

**Recommendation:** `qwen3:4b` (Apache-2.0) as the primary offline complement-graph generator and LLM cold-start baseline. Use `phi4-mini` (MIT) or `granite4:micro` (Apache-2.0) as a second model for agreement or self-consistency checks. Record the model digest (`ollama show --modelfile`) and the prompt hash in the benchmark artifacts. Exclude Ministral-8B-2410, and avoid Llama and Gemma 3 unless there is a specific quality reason.
