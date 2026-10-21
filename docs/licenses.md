# License review

Checked 2026-10-19 against primary sources (dataset pages, competition rules, LICENSE files, HF
model cards). The operative sentences, URLs and access checks are quoted in
[research_notes/05_datasets_licenses.md](research_notes/05_datasets_licenses.md). This is an
engineering review, not legal advice.

Mitate's own code is licensed under the PolyForm Noncommercial License 1.0.0 (`LICENSE.md`);
third-party material that ships with it is listed in `NOTICE`.

## The commercial firewall

Mitate is meant to run in commercial stores, and most of its benchmark data is non-commercial.
Three rules follow:

1. **Nothing fitted on non-commercial data ships.** Product defaults, priors, hyperparameters and
   pretrained rankers must be derived from commercially usable data (OTTO, CC BY 4.0) or from the
   merchant's own data. Configurations tuned on H&M, Dunnhumby, Amazon or Instacart live under
   `experiments/` only and are never loaded by `mitate/` or `serve/` defaults.
2. **No raw data or sampled IDs are redistributed** except for OTTO (CC BY 4.0, with attribution).
   For every other dataset the repo publishes download instructions, file checksums, deterministic
   sampling code and seeds.
3. **Research-only code is fetched, not vendored.** The official GRU4Rec code is downloaded by the
   benchmark runner at run time, pinned to commit `d1fc311` and checked against its SHA-256
   (`bench/gru4rec.py`), and is never imported by the product.

## Datasets

| Dataset | License | Benchmark & publication | Commercial use | Redistribute derived samples | How Mitate uses it |
|---|---|---|---|---|---|
| OTTO (Kaggle dataset `otto/recsys-dataset`) | CC BY 4.0 (data), MIT (code) | Yes | **Yes** | Yes, with attribution | Primary benchmark; the only dataset any shipped default may be derived from. Store IDs and seeds released. |
| H&M Personalized Fashion | Kaggle competition rules: "non-commercial purposes only"; no redistribution | Yes (academic research) | No | No (seeds and code only) | Fashion benchmark only |
| RelBench `rel-hm` | Inherits H&M terms | Same as H&M | No | No | Not used (same data) |
| Dunnhumby The Complete Journey | No dataset license; dunnhumby site terms (checked 2026-10-21) allow downloaded material to be used "solely for research, personal or non-commercial purposes" and "not modified or altered" | Yes (research) | No | No | Grocery and repeat benchmark only. The source CSVs are kept unmodified, including a private Kaggle copy for compute; stores are sampled on Kaggle and stay there; only scores leave (ledger D26) |
| Amazon Reviews 2023 | None assigned; maintainers: "primarily for research purposes" | Yes (research) | Treat as no | Seeds only | Text-rich e-commerce benchmark; content and cold-start studies |
| Amazon 2023 `bought_together` | Field is empty in every category | n/a | n/a | n/a | **Not used** |
| Amazon 2018 `also_buy` | None stated; cite Ni et al. 2019 | Yes (research) | Treat as no | Seeds only | Complement ground truth (evaluation only) |
| Instacart 2017 | "non-commercial use … subject to our Terms" | Yes (research) | No | No | Pipeline validation only (reality-check reproduction) |
| MovieLens-20M | GroupLens: no redistribution, no commercial use without permission | Yes (cite Harper & Konstan) | No | No | EASE reproduction only |
| Synerise RecSys 2025 | CC BY-NC 4.0; download returns 404 | — | No | — | Excluded |
| Coveo SIGIR eCom 2021 | Non-commercial; no sharing even with colleagues | — | No | No | Excluded |
| Ta-Feng, Tmall, Ali-CCP, Diginetica, Yoochoose | No licensor statement found | — | — | — | Excluded (unclear) |
| Indian Kaggle catalogs (Flipkart, Myntra, BigBasket) | Uploader claims on scraped data | — | No | No | Excluded; Misata generates Indian-style catalogs instead |
| Amazon-M2 (KDD Cup 2023) | Apache-2.0 per the authors' paper; AIcrowd login required | Yes | Yes | Yes | Optional multilingual text study |

## Models

| Model | License | Commercial | Use |
|---|---|---|---|
| Qwen3-Embedding-0.6B | Apache-2.0 | Yes | Benchmarked text embedder |
| intfloat/multilingual-e5-small | MIT | Yes | Serving default (`EngineConfig.embedder`) until the benchmark's embedder comparison says otherwise; multilingual incl. Indic, CPU-friendly |
| minishlab/potion-multilingual-128M | MIT | Yes | Static-embedding fallback |
| BAAI/bge-m3 | MIT | Yes | Benchmarked alternative |
| mixedbread-ai/mxbai-embed-large-v1 | Apache-2.0 | Yes | Benchmarked alternative (English) |
| jina-embeddings-v3 | CC BY-NC 4.0 | No | Excluded |
| EmbeddingGemma-300m | Gemma Terms of Use | Conditional | Not used |
| Qwen3-4B (Ollama `qwen3:4b`) | Apache-2.0 | Yes | Offline complement-graph generator |
| Phi-4-mini | MIT | Yes | Second-opinion generator / judge |
| Llama 3.x | Llama Community License (attribution, naming clause, 700M MAU) | Conditional | Avoided |
| Ministral-8B-2410 | Mistral Research License | No | Excluded |

## Code dependencies

All runtime dependencies are permissive (BSD, MIT, Apache-2.0, PSF): numpy, scipy, polars, pyarrow,
duckdb, scikit-learn, lightgbm, implicit, irspack, numba, optuna, pyyaml, pydantic, torch,
sentence-transformers, model2vec, fastapi, uvicorn, matplotlib, httpx. The exact set is pinned in
`uv.lock`.

| Package | License | Note |
|---|---|---|
| GRU4Rec official (`hidasib/GRU4Rec_PyTorch_Official`) | "Non-commercial use only" | Benchmark only; fetched at run time |
| sansa (`glami/sansa`) | Apache-2.0, but requires scikit-sparse → SuiteSparse CHOLMOD (GPL-2.0+ modules) | Not a product dependency; benchmark-only above 25K items |
| Misata (`rasinmuhammed/misata`, pinned 0.9.6.9, `demo` dependency group) | MIT | Synthetic demo store only (`serve/demo.py`); never a benchmark input |
| RecTools LiGR layer (reference for our SASRec variant) | Apache-2.0 | Architecture re-implemented, no code copied |
| session-rec (VSTAN reference) | No license | Not used; VSTAN re-implemented from the published equations |
| NBR reality-check repo (`liming-7/A-Next-Basket-Recommendation-Reality-Check`) | No license file | Its Instacart split files are used **locally** to reproduce GP-TopFreq (Gate 2); nothing from it is redistributed or shipped |
| gSASRec reference (`asash/gSASRec-pytorch`) | Apache-2.0 | Loss formula checked against it; no code copied |

## Interface assets

Fonts are subset and self-hosted from google/fonts at a pinned commit, each source checked by
SHA-256 (`scripts/build_ui_assets.py`). Each family's `OFL.txt` ships beside its files in
`ui/fonts/`; subsets keep the copyright and licence name records, as the OFL requires.

| Asset | License | Use |
|---|---|---|
| Gloock | SIL OFL 1.1 | Dashboard and demo headings, large numerals |
| Anek Latin | SIL OFL 1.1 | Interface text; condensed cut for demo product names |
| IBM Plex Mono | SIL OFL 1.1 | Ledger text: counts, windows, versions, x-ray |
| Shippori Mincho Bold | SIL OFL 1.1 | Outlines of 見立て (`ui/mark.svg`) and 見 (`ui/seal.svg`); converted to SVG paths, no font file shipped |
| Demo still-life plates | Generated by `serve/plates.py` (this repository) | Pictures for the synthetic demo catalog only |

The storefront widget loads no fonts and makes no third-party request.

## Attribution required in published work

- OTTO: cite DOI 10.34740/KAGGLE/DSV/4991874 and state changes made (store sampling).
- MovieLens: Harper & Konstan, TiiS 5(4):19, 2015.
- Amazon 2023: Hou et al. 2024. Amazon 2018: Ni, Li & McAuley, EMNLP 2019.
- Instacart: "The Instacart Online Grocery Shopping Dataset 2017", accessed <date>.
- H&M: the Kaggle competition. Dunnhumby: dunnhumby Source Files.
