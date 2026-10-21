"""Common recommender interface.

A model is fitted on a ``Log`` that ends at a cutoff and then scores a ``Queries`` batch against the
full catalog. Hyperparameters are dataclass fields; ``space`` declares the tuning search space so
every model gets the same Optuna budget through one code path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Self

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from mitate.data import Catalog, Log, Queries

F32 = NDArray[np.float32]


@dataclass
class Recommender(ABC):
    name: ClassVar[str]
    seed: int = 0

    #: True when ``fit`` only reads the catalog (works with zero interactions)
    catalog_only: ClassVar[bool] = False

    @abstractmethod
    def fit(self, log: Log, catalog: Catalog) -> Self: ...

    @abstractmethod
    def score(self, q: Queries) -> F32:
        """``(len(q), n_items)`` float32 scores; higher is better, ``-inf`` means ineligible."""

    @classmethod
    def space(cls, trial: Any) -> dict[str, Any]:
        """Sample hyperparameters from an Optuna trial. Default: nothing to tune."""
        return {}

    def params(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    # ------------------------------------------------------------------ ranking helpers
    def recommend(
        self,
        q: Queries,
        k: int,
        exclude: sp.csr_matrix | None = None,
        batch: int = 2048,
        available_from: NDArray[np.int64] | None = None,
    ) -> NDArray[np.int32]:
        """Top-k item ids per query, ``-1`` padded.

        ``exclude`` marks per-query ineligible items. ``available_from[i]`` is the time item ``i``
        enters the catalog; items not yet available at a query's ``now`` are never recommended
        (recommending "future items" is a leak, Ji et al. TOIS 2023).
        """
        out = np.full((len(q), k), -1, np.int32)
        for s in range(0, len(q), batch):
            e = min(len(q), s + batch)
            S = self.score(q.slice(s, e))
            if exclude is not None:
                ex = exclude[s:e].tocoo()
                S[ex.row, ex.col] = -np.inf
            if available_from is not None:
                S[available_from[None, :] > q.now[s:e, None]] = -np.inf
            out[s:e] = topk(S, k)
        return out


def topk(S: F32, k: int) -> NDArray[np.int32]:
    """Row-wise top-k indices, best first; ``-inf`` entries become ``-1``.

    Ties inside the top-k break by item id; which tied items cross the k boundary is deterministic
    (argpartition) but not id-ordered.
    """
    n = S.shape[1]
    kk = min(k, n)
    if kk < n:
        part = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
    else:
        part = np.tile(np.arange(n), (S.shape[0], 1))
    vals = np.take_along_axis(S, part, 1)
    order = np.lexsort((part, -vals), axis=1)
    idx = np.take_along_axis(part, order, 1).astype(np.int32)
    idx[~np.isfinite(np.take_along_axis(S, idx, 1))] = -1
    if kk < k:
        idx = np.pad(idx, ((0, 0), (0, k - kk)), constant_values=-1)
    return idx


def history_mask(q: Queries, n_items: int) -> sp.csr_matrix:
    return q.matrix(n_items, binary=True)
