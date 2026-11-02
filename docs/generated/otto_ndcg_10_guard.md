| Store size | Best single (chosen on test) | Shipped ranker alone | Served | Score | Blend kept | Δ vs best, 95% CI | p (Holm) |
|---|---|---:|---|---:|---:|---:|---:|
| 1K | Transition probe (PCTM) 0.1762 | 0.2467 | guard | 0.2467 | 5 of 5 | +0.0687 [+0.0345, +0.1030] | 0.000162 |
| 1K | Transition probe (PCTM) 0.1762 | 0.2467 | guard, conservative rule | 0.2467 | 5 of 5 | +0.0687 [+0.0345, +0.1030] | 0.000162 |
| 3K | Session kNN (VSTAN) 0.2155 | 0.2803 | guard | 0.2803 | 5 of 5 | +0.0691 [+0.0527, +0.0857] | 1.59e-14 |
| 3K | Session kNN (VSTAN) 0.2155 | 0.2803 | guard, conservative rule | 0.2803 | 5 of 5 | +0.0691 [+0.0527, +0.0857] | 1.59e-14 |
| 10K | Session kNN (VSTAN) 0.2130 | 0.2751 | guard | 0.2751 | 5 of 5 | +0.0638 [+0.0547, +0.0733] | 8.19e-39 |
| 10K | Session kNN (VSTAN) 0.2130 | 0.2751 | guard, conservative rule | 0.2751 | 5 of 5 | +0.0638 [+0.0547, +0.0733] | 8.19e-39 |
| 30K | GRU4Rec 0.2160 | 0.2825 | guard | 0.2825 | 5 of 5 | +0.0663 [+0.0588, +0.0740] | 1.1e-60 |
| 30K | GRU4Rec 0.2160 | 0.2825 | guard, conservative rule | 0.2825 | 5 of 5 | +0.0663 [+0.0588, +0.0740] | 1.1e-60 |
| 100K | iALS 0.2196 | 0.2864 | guard | 0.2864 | 5 of 5 | +0.0668 [+0.0637, +0.0698] | < 1e-300 |
| 100K | iALS 0.2196 | 0.2864 | guard, conservative rule | 0.2864 | 5 of 5 | +0.0668 [+0.0637, +0.0698] | < 1e-300 |
| 300K | SASRec (full softmax) 0.2202 | 0.2728 | guard | 0.2728 | 5 of 5 | +0.0526 [+0.0505, +0.0548] | < 1e-300 |
| 300K | SASRec (full softmax) 0.2202 | 0.2728 | guard, conservative rule | 0.2728 | 5 of 5 | +0.0526 [+0.0505, +0.0548] | < 1e-300 |
| 1M | SASRec (full softmax) 0.2274 | 0.2721 | guard | 0.2721 | 5 of 5 | +0.0447 [+0.0428, +0.0466] | < 1e-300 |
| 1M | SASRec (full softmax) 0.2274 | 0.2721 | guard, conservative rule | 0.2721 | 5 of 5 | +0.0447 [+0.0428, +0.0466] | < 1e-300 |
