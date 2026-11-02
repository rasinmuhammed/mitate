| Store size | Best single (chosen on test) | Shipped ranker alone | Served | Score | Blend kept | Δ vs best, 95% CI | p (Holm) |
|---|---|---:|---|---:|---:|---:|---:|
| 1K | Session kNN (VSTAN) 0.4693 | 0.5165 | guard | 0.5165 | 5 of 5 | +0.0545 [-0.0066, +0.1168] | 0.088 |
| 1K | Session kNN (VSTAN) 0.4693 | 0.5165 | guard, conservative rule | 0.5165 | 5 of 5 | +0.0545 [-0.0066, +0.1168] | 0.088 |
| 3K | Session kNN (VSTAN) 0.4864 | 0.5361 | guard | 0.5361 | 5 of 5 | +0.0514 [+0.0236, +0.0795] | 0.000819 |
| 3K | Session kNN (VSTAN) 0.4864 | 0.5361 | guard, conservative rule | 0.5361 | 5 of 5 | +0.0514 [+0.0236, +0.0795] | 0.000819 |
| 10K | BPR 0.4807 | 0.5290 | guard | 0.5290 | 5 of 5 | +0.0483 [+0.0304, +0.0659] | 3.93e-07 |
| 10K | BPR 0.4807 | 0.5290 | guard, conservative rule | 0.5290 | 5 of 5 | +0.0483 [+0.0304, +0.0659] | 3.93e-07 |
| 30K | SASRec (full softmax) 0.5060 | 0.5555 | guard | 0.5555 | 5 of 5 | +0.0497 [+0.0414, +0.0581] | 1.18e-30 |
| 30K | SASRec (full softmax) 0.5060 | 0.5555 | guard, conservative rule | 0.5555 | 5 of 5 | +0.0497 [+0.0414, +0.0581] | 1.18e-30 |
| 100K | SASRec (full softmax) 0.5357 | 0.5717 | guard | 0.5717 | 5 of 5 | +0.0361 [+0.0317, +0.0405] | 4.3e-55 |
| 100K | SASRec (full softmax) 0.5357 | 0.5717 | guard, conservative rule | 0.5717 | 5 of 5 | +0.0361 [+0.0317, +0.0405] | 4.3e-55 |
| 300K | iALS 0.5221 | 0.5515 | guard | 0.5515 | 5 of 5 | +0.0293 [+0.0260, +0.0328] | 9.17e-63 |
| 300K | iALS 0.5221 | 0.5515 | guard, conservative rule | 0.5515 | 5 of 5 | +0.0293 [+0.0260, +0.0328] | 9.17e-63 |
| 1M | SASRec (full softmax) 0.5117 | 0.5442 | guard | 0.5442 | 5 of 5 | +0.0324 [+0.0290, +0.0359] | 3.35e-72 |
| 1M | SASRec (full softmax) 0.5117 | 0.5442 | guard, conservative rule | 0.5442 | 5 of 5 | +0.0324 [+0.0290, +0.0359] | 3.35e-72 |
