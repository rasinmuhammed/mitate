| Store size | Best single (chosen on test) | Ranker alone | Rail | Score | Δ vs best, 95% CI | p (Holm) |
|---|---|---:|---|---:|---:|---:|
| 1K | Recent popularity 0.1645 | 0.2825 | ranker trained here | 0.2856 | +0.1237 [+0.0545, +0.1965] | 0.00552 |
| 1K | Recent popularity 0.1645 | 0.2825 | shipped OTTO ranker | 0.0467 | -0.0884 [-0.1605, -0.0121] | 0.0248 |
| 3K | Recent popularity 0.3148 | 0.3018 | ranker trained here | 0.2820 | -0.0317 [-0.0797, +0.0197] | 0.551 |
| 3K | Recent popularity 0.3148 | 0.3018 | shipped OTTO ranker | 0.0701 | -0.2314 [-0.3103, -0.1517] | 6.73e-07 |
| 10K | Recent popularity 0.1650 | 0.1855 | ranker trained here | 0.1821 | +0.0143 [-0.0057, +0.0366] | 0.551 |
| 10K | Recent popularity 0.1650 | 0.1855 | shipped OTTO ranker | 0.0740 | -0.0758 [-0.1096, -0.0427] | 2.74e-05 |
| 30K | Recent popularity 0.1296 | 0.1596 | ranker trained here | 0.1388 | +0.0086 [-0.0051, +0.0225] | 0.551 |
| 30K | Recent popularity 0.1296 | 0.1596 | shipped OTTO ranker | 0.0743 | -0.0502 [-0.0690, -0.0315] | 7.35e-07 |
| 100K | Session kNN (VSTAN) 0.0745 | 0.1122 | ranker trained here | 0.0936 | +0.0132 [+0.0073, +0.0190] | 5.25e-05 |
| 100K | Session kNN (VSTAN) 0.0745 | 0.1122 | shipped OTTO ranker | 0.0640 | -0.0124 [-0.0170, -0.0078] | 5.86e-07 |
| 300K | Session kNN (VSTAN) 0.0559 | 0.0897 | ranker trained here | 0.0735 | +0.0126 [+0.0096, +0.0155] | 7.47e-16 |
| 300K | Session kNN (VSTAN) 0.0559 | 0.0897 | shipped OTTO ranker | 0.0495 | -0.0066 [-0.0087, -0.0045] | 1.16e-09 |
| 1M | Session kNN (VSTAN) 0.0573 | 0.0797 | ranker trained here | 0.0635 | +0.0061 [+0.0045, +0.0077] | 1.15e-13 |
| 1M | Session kNN (VSTAN) 0.0573 | 0.0797 | shipped OTTO ranker | 0.0501 | -0.0072 [-0.0086, -0.0058] | 5.15e-22 |
