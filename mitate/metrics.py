"""Full-catalog top-k ranking metrics, computed per query so they can be paired across models.

Conventions (documented in docs/protocol.md):

* ``recs`` is an ``(n_queries, K)`` int array of item indices, best first, padded with ``-1``.
* ``targets`` is an ``(n_queries, n_items)`` CSR matrix; any stored entry is a relevant item.
* Relevance is binary. Queries with no targets are the caller's responsibility to drop; here they
  yield NaN so they can never silently count as zeros.
* ``recall`` divides by ``|T|`` (the next-basket convention, Li et al. TOIS 2023). ``recall_capped``
  divides by ``min(k, |T|)`` (the VAE-CF convention, Liang et al. WWW 2018, used by EASE).
* NDCG's ideal DCG places ``min(k, |T|)`` relevant items at the top.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

FloatArr = NDArray[np.float64]


def hit_matrix(recs: NDArray[np.integer], targets: sp.csr_matrix) -> NDArray[np.bool_]:
    """Boolean ``(n_queries, K)``: is ``recs[q, r]`` a target of query ``q``.

    Encodes (row, item) pairs as int64 keys and binary-searches the sorted target keys, so the
    cost is O(nnz + Q*K log nnz) with no Python loop.
    """
    if recs.shape[0] != targets.shape[0]:
        raise ValueError(f"recs has {recs.shape[0]} rows, targets {targets.shape[0]}")
    t = targets.tocsr()
    t.sort_indices()
    n_items = np.int64(t.shape[1])
    rows = np.repeat(np.arange(t.shape[0], dtype=np.int64), np.diff(t.indptr))
    tkeys = rows * n_items + t.indices.astype(np.int64)  # sorted: rows asc, then cols asc
    r = recs.astype(np.int64)
    valid = r >= 0
    rkeys = np.arange(r.shape[0], dtype=np.int64)[:, None] * n_items + np.where(valid, r, 0)
    pos = np.searchsorted(tkeys, rkeys)
    pos = np.minimum(pos, max(len(tkeys) - 1, 0))
    hits = (tkeys[pos] == rkeys) if len(tkeys) else np.zeros_like(valid)
    return np.asarray(hits & valid)


def _discounts(k: int) -> FloatArr:
    return 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))


@dataclass(frozen=True)
class PerQuery:
    """Per-query metric vectors at one cutoff ``k`` (NaN where a query has no targets)."""

    k: int
    ndcg: FloatArr
    recall: FloatArr
    recall_capped: FloatArr
    mrr: FloatArr
    hit: FloatArr
    precision: FloatArr
    ndcg_li2023: FloatArr  # IDCG over all |T| targets; only to reproduce Li et al. TOIS 2023 tables

    def means(self) -> dict[str, float]:
        return {
            f"{name}@{self.k}": float(np.nanmean(getattr(self, name)))
            for name in ("ndcg", "recall", "recall_capped", "mrr", "hit", "precision")
        }


def ranking_metrics(
    recs: NDArray[np.integer], targets: sp.csr_matrix, ks: Iterable[int] = (5, 10, 20)
) -> dict[int, PerQuery]:
    ks = sorted(set(ks))
    if recs.shape[1] < ks[-1]:
        raise ValueError(f"need at least {ks[-1]} recommendations, got {recs.shape[1]}")
    hits = hit_matrix(recs[:, : ks[-1]], targets).astype(np.float64)
    n_t = np.diff(targets.indptr).astype(np.float64)
    empty = n_t == 0
    disc = _discounts(ks[-1])
    ideal_cum = np.concatenate([[0.0], np.cumsum(disc)])
    max_t = int(n_t.max()) if n_t.size else 0
    ideal_full = np.concatenate([[0.0], np.cumsum(_discounts(max(max_t, 1)))])
    first = np.where(hits.any(axis=1), hits.argmax(axis=1), -1)
    out: dict[int, PerQuery] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in ks:
            h = hits[:, :k]
            n_hit = h.sum(axis=1)
            dcg = h @ disc[:k]
            idcg = ideal_cum[np.minimum(n_t, k).astype(np.int64)]
            rr = np.where((first >= 0) & (first < k), 1.0 / (first + 1.0), 0.0)
            res = {
                "ndcg": dcg / idcg,
                "recall": n_hit / n_t,
                "recall_capped": n_hit / np.minimum(n_t, k),
                "mrr": rr,
                "hit": (n_hit > 0).astype(np.float64),
                "precision": n_hit / k,
                "ndcg_li2023": dcg / ideal_full[n_t.astype(np.int64)],
            }
            for v in res.values():
                v[empty] = np.nan
            out[k] = PerQuery(k=k, **res)
    return out


# ----------------------------------------------------------------------------- beyond accuracy


def catalog_coverage(recs: NDArray[np.integer], n_items: int, k: int) -> float:
    """Share of the catalog appearing in at least one top-k list."""
    r = recs[:, :k]
    return float(np.unique(r[r >= 0]).size / n_items)


def novelty(recs: NDArray[np.integer], item_pop: FloatArr, n_users: int, k: int) -> FloatArr:
    """Mean self-information -log2(p(i)) of the top-k, with p(i) = (pop(i)+1)/(n_users+1).

    ``item_pop`` counts distinct training users per item. Laplace smoothing keeps cold items finite.
    """
    r = recs[:, :k]
    p = (item_pop + 1.0) / (n_users + 1.0)
    si = -np.log2(p)
    vals = np.where(r >= 0, si[np.maximum(r, 0)], np.nan)
    return np.asarray(np.nanmean(vals, axis=1))


def avg_rec_popularity(recs: NDArray[np.integer], item_pop: FloatArr, k: int) -> FloatArr:
    """ARP (Abdollahpouri et al. 2019): mean training popularity of the recommended items."""
    r = recs[:, :k]
    vals = np.where(r >= 0, item_pop[np.maximum(r, 0)], np.nan)
    return np.asarray(np.nanmean(vals, axis=1))


def gini_exposure(recs: NDArray[np.integer], n_items: int, k: int) -> float:
    """Gini coefficient of how often each catalog item is recommended (0 = perfectly even)."""
    r = recs[:, :k]
    counts = np.bincount(r[r >= 0].ravel(), minlength=n_items).astype(np.float64)
    if counts.sum() == 0:
        return float("nan")
    c = np.sort(counts)
    n = c.size
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * c).sum()) / (n * c.sum()) - (n + 1.0) / n)


def intra_list_diversity(
    recs: NDArray[np.integer], item_vecs: NDArray[np.floating], k: int
) -> FloatArr:
    """ILD: mean pairwise cosine distance among the top-k (Ziegler et al. 2005).

    ``item_vecs`` must be L2-normalised. For category-based ILD pass one-hot category vectors, which
    makes the distance 1 for different categories and 0 for the same one.
    """
    r = recs[:, :k]
    out = np.full(r.shape[0], np.nan)
    for q in range(r.shape[0]):
        row = r[q][r[q] >= 0]
        m = row.size
        if m < 2:
            continue
        v = item_vecs[row]
        sim = v @ v.T
        out[q] = 1.0 - (sim.sum() - np.trace(sim)) / (m * (m - 1))
    return out


# ----------------------------------------------------------------------------- repeat / explore


def split_repeat_explore(
    targets: sp.csr_matrix, history: sp.csr_matrix
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Split targets into items the user has interacted with before (repeat) and new (explore).

    Follows Li et al., "A Next Basket Recommendation Reality Check" (TOIS 2023): an item is a
    repeat target if it occurs anywhere in the user's history before the cutoff.
    """
    t = (targets > 0).astype(np.float32).tocsr()
    h = (history > 0).astype(np.float32).tocsr()
    rep = t.multiply(h).tocsr()
    exp = (t - rep).tocsr()
    rep.eliminate_zeros()
    exp.eliminate_zeros()
    return rep, exp


def short_head(item_pop: FloatArr, share: float = 0.8) -> NDArray[np.bool_]:
    """Smallest set of most-popular items covering ``share`` of training interactions
    (Abdollahpouri et al. 2019); everything else is the long tail."""
    order = np.argsort(-item_pop, kind="stable")
    cum = np.cumsum(item_pop[order])
    total = cum[-1] if cum.size else 0.0
    head = np.zeros(item_pop.size, bool)
    if total > 0:
        n_head = int(np.searchsorted(cum, share * total)) + 1
        head[order[:n_head]] = True
    return head


def aplt(recs: NDArray[np.integer], head: NDArray[np.bool_], k: int) -> FloatArr:
    """Share of each top-k list drawn from the long tail."""
    r = recs[:, :k]
    valid = r >= 0
    tail = valid & ~head[np.maximum(r, 0)]
    return np.asarray(tail.sum(1) / np.maximum(valid.sum(1), 1), np.float64)


def exposure_entropy(recs: NDArray[np.integer], n_items: int, k: int) -> float:
    r = recs[:, :k]
    c = np.bincount(r[r >= 0].ravel(), minlength=n_items).astype(np.float64)
    if c.sum() == 0:
        return float("nan")
    p = c[c > 0] / c.sum()
    return float(-(p * np.log(p)).sum())
