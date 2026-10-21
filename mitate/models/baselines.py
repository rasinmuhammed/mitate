"""Non-personalised, frequency and neighbourhood baselines.

All of them are deliberately simple; tuned properly they are the bar every other method must clear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Self, cast

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from sklearn.preprocessing import normalize

from mitate.data import DEFAULT_WEIGHTS, Catalog, Event, Log, Queries
from mitate.models.base import F32, Recommender
from mitate.models.ease import ease_weights

HOUR = 3600.0
DAY = 24 * HOUR


def _decayed_popularity(
    log: Log, half_life_days: float | None, weights: Any
) -> NDArray[np.float64]:
    w = np.zeros(int(max(Event)) + 1)
    for k, v in weights.items():
        w[int(k)] = v
    v = w[log.event]
    if half_life_days and len(log):
        now = log.ts.max() + 1
        v = v * np.exp2(-(now - log.ts) / (half_life_days * DAY))
    # numpy's stub types bincount as intp; with float weights it returns float64
    return cast(NDArray[np.float64], np.bincount(log.item, weights=v, minlength=log.n_items))


def _tiebreak(pop: NDArray[np.float64]) -> NDArray[np.float64]:
    """Popularity rescaled into [0, 1) so it can break ties under any integer-valued score."""
    m = pop.max() if pop.size else 0.0
    return (pop / (m * (1 + 1e-6)) if m > 0 else np.zeros_like(pop)).astype(np.float64)


# --------------------------------------------------------------------------- popularity
@dataclass
class Popularity(Recommender):
    """Global popularity: weighted event counts over the whole training window."""

    name: ClassVar[str] = "pop"
    pop: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0), repr=False)

    def fit(self, log: Log, catalog: Catalog) -> Self:
        self.pop = _decayed_popularity(log, None, DEFAULT_WEIGHTS)
        return self

    def score(self, q: Queries) -> F32:
        return np.broadcast_to(self.pop.astype(np.float32), (len(q), self.pop.size)).copy()


@dataclass
class RecencyPopularity(Recommender):
    """Exponentially time-decayed popularity; the half-life is tuned."""

    name: ClassVar[str] = "recpop"
    half_life_days: float = 7.0
    pop: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0), repr=False)

    @classmethod
    def space(cls, trial: Any) -> dict[str, Any]:
        return {"half_life_days": trial.suggest_float("half_life_days", 0.25, 365, log=True)}

    def fit(self, log: Log, catalog: Catalog) -> Self:
        self.pop = _decayed_popularity(log, self.half_life_days, DEFAULT_WEIGHTS)
        return self

    def score(self, q: Queries) -> F32:
        return np.broadcast_to(self.pop.astype(np.float32), (len(q), self.pop.size)).copy()


@dataclass
class GPTopFreq(Recommender):
    """GP-TopFreq (Li et al., TOIS 2023): the shopper's most frequent items, filled with global
    popularity. Frequency counts every past interaction of the query's history."""

    name: ClassVar[str] = "gp_topfreq"
    pop: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0), repr=False)

    def fit(self, log: Log, catalog: Catalog) -> Self:
        self.pop = _tiebreak(_decayed_popularity(log, None, DEFAULT_WEIGHTS))
        return self

    def score(self, q: Queries) -> F32:
        freq = q.matrix(self.pop.size, binary=False, weights={e: 1.0 for e in Event})
        S: NDArray[np.float64] = freq.toarray().astype(np.float64)
        S += self.pop[None, :]
        return S.astype(np.float32)


# --------------------------------------------------------------------------- item-item
def _topk_sparse_rows(W: NDArray[np.floating[Any]] | sp.spmatrix, k: int) -> sp.csr_matrix:
    """Keep the k largest entries of each row (dense or sparse input)."""
    vals: list[NDArray[Any]] | NDArray[Any]  # per-row pieces if sparse, one array if dense
    if sp.issparse(W):
        W = cast(sp.spmatrix, W).tocsr()
        rows, cols, vals = [], [], []
        for r in range(W.shape[0]):
            s, e = W.indptr[r], W.indptr[r + 1]
            sel = np.argpartition(-W.data[s:e], k - 1)[:k] + s if e - s > k else np.arange(s, e)
            rows.append(np.full(sel.size, r))
            cols.append(W.indices[sel])
            vals.append(W.data[sel])
        return (
            sp.csr_matrix(
                (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=W.shape
            )
            if rows
            else sp.csr_matrix(W.shape)
        )
    n = W.shape[0]
    k = min(k, W.shape[1])
    idx = np.argpartition(-W, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(W, idx, 1)
    keep = vals > 0
    return sp.csr_matrix(
        (vals[keep], (np.repeat(np.arange(n), k).reshape(n, k)[keep], idx[keep])), shape=W.shape
    )


@dataclass
class ItemKNN(Recommender):
    """Item-based kNN with shrunk asymmetric cosine (Aiolli 2013 generalisation of cosine) and
    optional IDF/BM25-free weighting, as tuned in Dacrema et al. (TOIS 2021)."""

    name: ClassVar[str] = "itemknn"
    neighbours: int = 100
    shrink: float = 10.0
    alpha: float = 0.5  # 0.5 = cosine; 0/1 = conditional-probability-like asymmetry
    decay_half_life_days: float = 0.0  # weight on query history recency (0 = off)
    W: sp.csr_matrix | None = field(default=None, repr=False)

    @classmethod
    def space(cls, trial: Any) -> dict[str, Any]:
        return {
            "neighbours": trial.suggest_int("neighbours", 5, 1000, log=True),
            "shrink": trial.suggest_float("shrink", 0.0, 1000.0),
            "alpha": trial.suggest_float("alpha", 0.0, 1.0),
            "decay_half_life_days": trial.suggest_categorical(
                "decay_half_life_days", [0.0, 1.0, 7.0, 30.0, 180.0]
            ),
        }

    def fit(self, log: Log, catalog: Catalog) -> Self:
        X = _interaction_matrix(log)
        Xb = X.copy()
        Xb.data[:] = 1.0
        co = (Xb.T @ Xb).tocsr().astype(np.float64)
        co.setdiag(0.0)
        co.eliminate_zeros()
        n = np.asarray(Xb.sum(axis=0)).ravel()
        a = np.power(n, self.alpha)
        b = np.power(n, 1.0 - self.alpha)
        coo = co.tocoo()
        denom = a[coo.row] * b[coo.col] + self.shrink + 1e-9
        sim = sp.csr_matrix((coo.data / denom, (coo.row, coo.col)), shape=co.shape)
        self.W = _topk_sparse_rows(sim, self.neighbours)
        return self

    def score(self, q: Queries) -> F32:
        assert self.W is not None
        hl = self.decay_half_life_days * DAY or None
        Xq = q.matrix(self.W.shape[0], binary=hl is None, decay_half_life=hl)
        return np.asarray((Xq @ self.W).toarray(), np.float32)


@dataclass
class RP3beta(Recommender):
    """RP3beta (Paudel et al., TiiS 2017): 3-step random walk item->user->item with transition
    probabilities raised to alpha, penalised by target-item popularity^beta, top-k per row."""

    name: ClassVar[str] = "rp3beta"
    alpha: float = 1.0
    beta: float = 0.5
    neighbours: int = 100
    W: sp.csr_matrix | None = field(default=None, repr=False)

    @classmethod
    def space(cls, trial: Any) -> dict[str, Any]:
        return {
            "alpha": trial.suggest_float("alpha", 0.0, 2.0),
            "beta": trial.suggest_float("beta", 0.0, 2.0),
            "neighbours": trial.suggest_int("neighbours", 5, 1000, log=True),
        }

    def fit(self, log: Log, catalog: Catalog) -> Self:
        X = _interaction_matrix(log)
        X.data[:] = 1.0
        Pui = normalize(X, norm="l1", axis=1)
        Piu = normalize(X.T.tocsr(), norm="l1", axis=1)
        Pui.data **= self.alpha
        Piu.data **= self.alpha
        W = (Piu @ Pui).tocsr()
        deg = np.asarray(X.sum(axis=0)).ravel()
        pen = np.zeros_like(deg)
        nz = deg > 0
        pen[nz] = deg[nz] ** -self.beta
        W = W @ sp.diags(pen)
        W.setdiag(0.0)
        W.eliminate_zeros()
        W = normalize(_topk_sparse_rows(W, self.neighbours), norm="l1", axis=1)
        self.W = W.tocsr()
        return self

    def score(self, q: Queries) -> F32:
        assert self.W is not None
        Xq = q.matrix(self.W.shape[0], binary=True)
        return np.asarray((Xq @ self.W).toarray(), np.float32)


@dataclass
class EASE(Recommender):
    """EASE (Steck 2019). Dense, so it refuses catalogs larger than ``max_items``."""

    name: ClassVar[str] = "ease"
    lam: float = 100.0
    max_items: int = 25_000
    B: NDArray[np.floating[Any]] | None = field(default=None, repr=False)

    @classmethod
    def space(cls, trial: Any) -> dict[str, Any]:
        return {"lam": trial.suggest_float("lam", 1.0, 1e5, log=True)}

    def fit(self, log: Log, catalog: Catalog) -> Self:
        if log.n_items > self.max_items:
            raise MemoryError(f"EASE capped at {self.max_items} items, got {log.n_items}")
        X = _interaction_matrix(log)
        X.data[:] = 1.0
        self.B = ease_weights(X, self.lam, dtype=np.float32)
        return self

    def score(self, q: Queries) -> F32:
        assert self.B is not None
        Xq = q.matrix(self.B.shape[0], binary=True)
        return np.asarray(Xq @ self.B, np.float32)


def _interaction_matrix(log: Log) -> sp.csr_matrix:
    """Rows are known users plus anonymous sessions, so session-only stores still train item-item
    models. Anonymous sessions get row ids after the user block."""
    anon = log.user < 0
    if not anon.any():
        return log.user_item(binary=True)
    _, sess_row = np.unique(log.session[anon], return_inverse=True)
    rows = log.user.astype(np.int64).copy()
    rows[anon] = log.n_users + sess_row
    n_rows = log.n_users + int(sess_row.max()) + 1
    X = sp.csr_matrix(
        (np.ones(rows.size, np.float32), (rows, log.item)), shape=(n_rows, log.n_items)
    )
    X.sum_duplicates()
    X.data[:] = 1.0
    return X
