"""Turn a model's top-k lists into every metric the protocol reports."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from bench.tasks import EvalSet
from mitate.data import Log
from mitate.metrics import (
    aplt,
    avg_rec_popularity,
    catalog_coverage,
    exposure_entropy,
    gini_exposure,
    novelty,
    ranking_metrics,
    short_head,
    split_repeat_explore,
)

KS = (5, 10, 20)


def evaluate(
    recs: NDArray[np.int32], ev: EvalSet, train: Log
) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
    """Per-query metric vectors (NaN where undefined) and store-level beyond-accuracy scalars."""
    per: dict[str, NDArray[np.float64]] = {}
    for k, m in ranking_metrics(recs, ev.targets, KS).items():
        for name in ("ndcg", "recall", "mrr", "hit"):
            per[f"{name}@{k}"] = getattr(m, name)
    rep, exp = split_repeat_explore(ev.targets, ev.history)
    for tag, T in (("rep", rep), ("expl", exp)):
        for k, m in ranking_metrics(recs, T, (10, 20)).items():
            per[f"recall_{tag}@{k}"] = m.recall
            per[f"hit_{tag}@{k}"] = m.hit
    # RepR@10 (Li et al. 2023): share of each top-10 that the shopper has interacted with before
    hist = ev.history
    r10 = recs[:, :10]
    repr10 = np.full(len(ev), np.nan)
    for q in range(len(ev)):
        row = r10[q][r10[q] >= 0]
        if row.size:
            repr10[q] = np.isin(row, hist.indices[hist.indptr[q] : hist.indptr[q + 1]]).mean()
    per["repr@10"] = repr10
    pop = train.item_counts()
    buyers = train.item_counts(distinct_users=True)
    n_users = max(int(np.unique(train.user[train.user >= 0]).size), 1)
    head = short_head(pop)
    per["novelty@10"] = novelty(recs, buyers, n_users, 10)
    per["arp@10"] = avg_rec_popularity(recs, pop, 10)
    per["aplt@10"] = aplt(recs, head, 10)
    n = train.n_items
    store = {
        "coverage@10": catalog_coverage(recs, n, 10),
        "coverage@20": catalog_coverage(recs, n, 20),
        "gini@10": gini_exposure(recs, n, 10),
        "entropy@10": exposure_entropy(recs, n, 10),
        "n_queries": len(ev),
        "n_units": int(np.unique(ev.unit).size),
        "n_rep_targets": int(rep.nnz),
        "n_expl_targets": int(exp.nnz),
    }
    return per, store
