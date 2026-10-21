| Method | 1K | 10K | 100K |
|---|---:|---:|---:|
| **Mitate** (shipped engine, store data) | 0.1191 ± 0.0720 | 0.1572 ± 0.0486 | 0.2011 ± 0.1204 |
| **Mitate, day one** (catalog only) | 0.1191 ± 0.0720 | 0.1572 ± 0.0486 | 0.1963 ± 0.1182 |
| Co-purchase (cosine) | 0.2730 ± 0.2246 | 0.1758 ± 0.0775 | 0.1427 ± 0.1212 |
| EASE (tuned) | 0.0658 ± 0.0490 | 0.1314 ± 0.0653 | 0.1602 ± 0.1156 |
| ItemKNN (tuned) | 0.0746 ± 0.0556 | 0.1124 ± 0.0486 | 0.1559 ± 0.1292 |
| Other-category popularity | 0.1339 ± 0.2247 | 0.0723 ± 0.0602 | 0.0270 ± 0.0222 |
| Popularity | 0.2164 ± 0.1861 | 0.1294 ± 0.0896 | 0.0434 ± 0.0278 |
| Same-category popularity | 0.2446 ± 0.2236 | 0.2755 ± 0.1138 | 0.1592 ± 0.0173 |
| Text similarity, e5-small | **0.3607 ± 0.2021** | **0.3867 ± 0.0856** | **0.2967 ± 0.0189** |
| Text similarity, other categories (e5-small) | 0.1191 ± 0.0720 | 0.1572 ± 0.0486 | 0.1963 ± 0.1182 |
| Text similarity, other categories (potion) | 0.0268 ± 0.0444 | 0.1102 ± 0.0459 | 0.1529 ± 0.1266 |
| Text similarity, potion-multilingual | 0.1752 ± 0.1476 | 0.2701 ± 0.0589 | 0.2405 ± 0.0835 |
| *Test anchors (stores)* | *77 (5)* | *246 (5)* | *1,763 (5)* |

| Store size | Mitate vs the baseline each store picks on its dev anchors: Δ NDCG@10, 95% CI | p (Holm) | Best baseline on test (oracle) | Mitate vs oracle: Δ, 95% CI | p (Holm) |
|---|---:|---:|---|---:|---:|
| 1K | -0.2917 [-0.3726, -0.2112] | 7.11e-10 | Text similarity, e5-small | -0.2779 [-0.3560, -0.2029] | 8.52e-10 |
| 10K | -0.1944 [-0.2359, -0.1535] | 1.19e-17 | Text similarity, e5-small | -0.2048 [-0.2452, -0.1647] | 1.04e-19 |
| 100K | -0.1305 [-0.1433, -0.1181] | 3.79e-84 | Text similarity, e5-small | -0.1296 [-0.1425, -0.1172] | 7.84e-83 |
