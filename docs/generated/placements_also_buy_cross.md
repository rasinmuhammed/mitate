| Method | 1K | 10K | 100K |
|---|---:|---:|---:|
| **Mitate** (shipped engine, store data) | 0.1236 ± 0.1561 | 0.2288 ± 0.1200 | **0.2814 ± 0.2082** |
| **Mitate, day one** (catalog only) | 0.1236 ± 0.1561 | 0.2288 ± 0.1200 | 0.2702 ± 0.1932 |
| Co-purchase (cosine) | 0.2837 ± 0.3034 | 0.1867 ± 0.1608 | 0.1136 ± 0.1339 |
| EASE (tuned) | 0.0197 ± 0.0546 | 0.1204 ± 0.0783 | 0.1224 ± 0.1201 |
| ItemKNN (tuned) | 0.0197 ± 0.0546 | 0.1018 ± 0.0808 | 0.1211 ± 0.1345 |
| Other-category popularity | **0.2969 ± 0.3131** | 0.1744 ± 0.1302 | 0.0342 ± 0.0255 |
| Popularity | 0.2821 ± 0.3046 | 0.1701 ± 0.1263 | 0.0329 ± 0.0234 |
| Same-category popularity | 0.0027 ± 0.0075 | 0.0037 ± 0.0063 | 0.0010 ± 0.0012 |
| Text similarity, e5-small | 0.1024 ± 0.1244 | 0.1539 ± 0.0852 | 0.1691 ± 0.0743 |
| Text similarity, other categories (e5-small) | 0.1236 ± 0.1561 | **0.2288 ± 0.1200** | 0.2702 ± 0.1932 |
| Text similarity, other categories (potion) | 0.0696 ± 0.0827 | 0.1336 ± 0.1012 | 0.2100 ± 0.2180 |
| Text similarity, potion-multilingual | 0.0464 ± 0.0604 | 0.0962 ± 0.0914 | 0.1439 ± 0.1285 |
| *Test anchors (stores)* | *51 (5)* | *179 (5)* | *1,587 (5)* |

| Store size | Mitate vs the baseline each store picks on its dev anchors: Δ NDCG@10, 95% CI | p (Holm) | Best baseline on test (oracle) | Mitate vs oracle: Δ, 95% CI | p (Holm) |
|---|---:|---:|---|---:|---:|
| 1K | +0.0152 [-0.0719, +0.1035] | 0.743 | Other-category popularity | +0.0519 [-0.0578, +0.1630] | 0.739 |
| 10K | +0.0094 [+0.0010, +0.0212] | 0.153 | Text similarity, other categories (e5-small) | +0.0000 [+0.0000, +0.0000] | 1 |
| 100K | +0.0036 [+0.0015, +0.0058] | 0.00462 | Text similarity, other categories (e5-small) | +0.0036 [+0.0015, +0.0058] | 0.00462 |
