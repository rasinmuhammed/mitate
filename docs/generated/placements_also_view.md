| Method | 1K | 10K | 100K |
|---|---:|---:|---:|
| **Mitate** (shipped engine, store data) | 0.4541 ± 0.1178 | **0.3924 ± 0.1514** | **0.3552 ± 0.1041** |
| **Mitate, day one** (catalog only) | 0.4541 ± 0.1178 | 0.3915 ± 0.1517 | 0.3545 ± 0.1053 |
| Co-purchase (cosine) | 0.1528 ± 0.1338 | 0.1107 ± 0.0473 | 0.1101 ± 0.0653 |
| EASE (tuned) | 0.0206 ± 0.0359 | 0.0585 ± 0.0171 | 0.1197 ± 0.0536 |
| ItemKNN (tuned) | 0.0153 ± 0.0272 | 0.0504 ± 0.0167 | 0.1232 ± 0.0745 |
| Other-category popularity | 0.0758 ± 0.1205 | 0.0368 ± 0.0449 | 0.0122 ± 0.0180 |
| Popularity | 0.1508 ± 0.1218 | 0.0884 ± 0.0654 | 0.0332 ± 0.0328 |
| Same-category popularity | 0.4283 ± 0.2809 | 0.3691 ± 0.0989 | 0.2306 ± 0.0732 |
| Text similarity, e5-small | **0.4541 ± 0.1178** | 0.3915 ± 0.1517 | 0.3545 ± 0.1053 |
| Text similarity, other categories (e5-small) | 0.1466 ± 0.1834 | 0.0847 ± 0.0852 | 0.2024 ± 0.0936 |
| Text similarity, other categories (potion) | 0.0954 ± 0.1151 | 0.0676 ± 0.0758 | 0.1492 ± 0.1248 |
| Text similarity, potion-multilingual | 0.2719 ± 0.0932 | 0.2891 ± 0.0915 | 0.2704 ± 0.0588 |
| *Test anchors (stores)* | *52 (5)* | *172 (5)* | *1,145 (5)* |

| Store size | Mitate vs the baseline each store picks on its dev anchors: Δ NDCG@10, 95% CI | p (Holm) | Best baseline on test (oracle) | Mitate vs oracle: Δ, 95% CI | p (Holm) |
|---|---:|---:|---|---:|---:|
| 1K | +0.0126 [-0.0285, +0.0632] | 0.677 | Text similarity, e5-small | +0.0000 [+0.0000, +0.0000] | 1 |
| 10K | +0.0116 [-0.0065, +0.0310] | 0.677 | Text similarity, e5-small | +0.0014 [-0.0001, +0.0039] | 0.395 |
| 100K | -0.0014 [-0.0042, +0.0013] | 0.677 | Text similarity, e5-small | +0.0004 [-0.0001, +0.0011] | 0.361 |
