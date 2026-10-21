"""Gate 2 (second check): reproduce GP-TopFreq on the NBR reality-check Instacart preprocessing.

Li, Jullien, Ariannezhad & de Rijke, "A Next Basket Recommendation Reality Check", TOIS 2023.
Data: their preprocessed files (dataset/instacart_history.csv, instacart_future.csv; 19,435 users,
13,897 items). GP-TopFreq has no training, so all users are evaluated, which makes the expected
values deterministic (docs/research_notes/03_nbr_evaluation.md, A1):
    R@10 0.3304, ndcg_li2023@10 0.3417, PHR@10 0.8422 (each +-0.0005);
    G-TopFreq Recall_rep@10 0.1002.
Their published 5-split means (0.3269 / 0.3383 / 0.8463) are within ~0.005 of these.

Our GP-TopFreq breaks ties among equally frequent personal items by global popularity; theirs
breaks them by first occurrence. ``--tiebreak first`` reproduces theirs exactly.

Usage: python -m bench.repro.gp_topfreq_instacart data/raw/nbr_reality_check [--tiebreak first]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import scipy.sparse as sp
from numpy.typing import NDArray

from mitate.data import Catalog, Event, Log, Queries
from mitate.metrics import ranking_metrics, split_repeat_explore
from mitate.models.baselines import GPTopFreq, Popularity

EXPECTED = {
    "recall@10": 0.3304,
    "ndcg_li2023@10": 0.3417,
    "phr@10": 0.8422,
    "g_topfreq_recall_rep@10": 0.1002,
}


def main(root: str, tiebreak: str = "popularity") -> dict[str, float]:
    h = pl.read_csv(Path(root) / "instacart_history.csv")
    f = pl.read_csv(Path(root) / "instacart_future.csv")
    users = np.unique(h["user_id"].to_numpy())
    n_items = int(max(cast(int, h["product_id"].max()), cast(int, f["product_id"].max()))) + 1
    uid = np.searchsorted(users, h["user_id"].to_numpy())
    # order_number is the only clock; one basket per (user, order)
    sess = uid.astype(np.int64) * 1000 + h["order_number"].to_numpy()
    log = Log.build(
        uid,
        h["product_id"].to_numpy(),
        h["order_number"].to_numpy(),
        np.full(h.height, Event.ORDER),
        sess,
        users.size,
        n_items,
    )
    order = np.lexsort((np.arange(len(log)), log.ts, log.user))
    lens = np.bincount(log.user, minlength=users.size)
    q = Queries(
        indptr=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
        items=log.item[order],
        ts=log.ts[order],
        event=log.event[order],
        now=np.full(users.size, 10**6, np.int64),
        user=np.arange(users.size, dtype=np.int32),
        session_start=lens.astype(np.int64),
        anchor=np.full(users.size, -1, np.int32),
        session=log.session[order],
    )
    fu = np.searchsorted(users, f["user_id"].to_numpy())
    T = sp.csr_matrix(
        (np.ones(f.height), (fu, f["product_id"].to_numpy())), shape=(users.size, n_items)
    )
    T.sum_duplicates()
    T.data[:] = 1
    cat = Catalog(n_items=n_items)
    gp = GPTopFreq().fit(log, cat)
    recs: NDArray[np.integer[Any]]
    if tiebreak == "first":
        # Their personal ranking: frequency, ties by first occurrence in *file order* (dict
        # insertion order); then global popularity fills the rest. Log.build re-sorts rows within
        # a basket by item id, so first occurrence is taken from the raw file here.
        first = (
            h.with_row_index("row")
            .group_by(["user_id", "product_id"])
            .agg(pl.col("row").min(), pl.len().alias("freq"))
        )
        S = np.broadcast_to(gp.pop, (users.size, n_items)).copy()  # popularity in [0, 1)
        fu_ = np.searchsorted(users, first["user_id"].to_numpy())
        rows = first["row"].to_numpy().astype(np.float64)
        # score = freq + 1 + tiny first-occurrence bonus: dominates popularity, breaks freq ties
        S[fu_, first["product_id"].to_numpy()] = (
            first["freq"].to_numpy() + 1.0 + 0.5 * (1.0 - rows / (rows.max() + 1))
        )
        recs = np.argsort(-S, axis=1, kind="stable")[:, :20]
    else:
        recs = gp.recommend(q, 20)
    m = ranking_metrics(recs, T, (10, 20))
    pop_recs = Popularity().fit(log, cat).recommend(q, 20)
    rep, _ = split_repeat_explore(T, q.matrix(n_items, binary=True))
    g_rep = ranking_metrics(pop_recs, rep, (10,))[10]
    out = {
        "recall@10": float(np.nanmean(m[10].recall)),
        "ndcg_li2023@10": float(np.nanmean(m[10].ndcg_li2023)),
        "ndcg@10": float(np.nanmean(m[10].ndcg)),
        "phr@10": float(np.nanmean(m[10].hit)),
        "recall@20": float(np.nanmean(m[20].recall)),
        "g_topfreq_recall_rep@10": float(np.nanmean(g_rep.recall)),
    }
    out_all = {
        "tiebreak": tiebreak,
        "users": int(users.size),
        "items": n_items,
        "measured": out,
        "expected": EXPECTED,
        "abs_diff": {k: round(out[k] - v, 5) for k, v in EXPECTED.items()},
    }
    dest = Path("experiments/results/repro") / f"gp_topfreq_instacart_{tiebreak}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out_all, indent=2))
    print(json.dumps(out_all, indent=2))
    return out


if __name__ == "__main__":
    main(
        sys.argv[1],
        sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--tiebreak" else "popularity",
    )
