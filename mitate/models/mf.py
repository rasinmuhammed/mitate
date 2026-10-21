"""Matrix factorisation baselines on top of the ``implicit`` library.

iALS (Hu et al. 2008; tuned per Rendle et al. 2022) gets an exact fold-in: each query's user vector
is the closed-form ALS solution given its history, so new and anonymous shoppers are scored
consistently with returning ones. BPR has no closed-form fold-in, so known users keep their learned
vector and others use the mean of their history's item vectors.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self

import numpy as np
from numpy.typing import NDArray

from mitate.data import Catalog, Log, Queries
from mitate.models.base import F32, Recommender
from mitate.models.baselines import _interaction_matrix

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # implicit warns otherwise; we use Accelerate


@dataclass
class IALS(Recommender):
    name: ClassVar[str] = "ials"
    factors: int = 64
    regularization: float = 0.01
    alpha: float = 10.0  # implicit: confidence = alpha*r, so alpha<=1 ignores positives
    iterations: int = 15
    Y: NDArray[np.float64] | None = field(default=None, repr=False)

    @classmethod
    def space(cls, trial: Any) -> dict[str, Any]:
        return {
            "factors": trial.suggest_categorical("factors", [16, 32, 64, 128, 256]),
            "regularization": trial.suggest_float("regularization", 1e-4, 100.0, log=True),
            "alpha": trial.suggest_float("alpha", 0.1, 100.0, log=True),
            "iterations": trial.suggest_int("iterations", 5, 30),
        }

    def fit(self, log: Log, catalog: Catalog) -> Self:
        from implicit.cpu.als import AlternatingLeastSquares

        X = _interaction_matrix(log)
        m = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            alpha=self.alpha,
            iterations=self.iterations,
            random_state=self.seed,
            num_threads=1,  # deterministic
            calculate_training_loss=False,
        )
        m.fit(X, show_progress=False)
        self.Y = np.asarray(m.item_factors, np.float64)
        return self

    def user_vectors(self, q: Queries) -> NDArray[np.float64]:
        assert self.Y is not None
        Y = self.Y
        f = Y.shape[1]
        reg = self.regularization * np.eye(f)
        YtY = Y.T @ Y
        H = q.matrix(Y.shape[0], binary=True)
        U = np.zeros((len(q), f))
        for r in range(len(q)):
            s, e = H.indptr[r], H.indptr[r + 1]
            if s == e:
                continue
            Yh = Y[H.indices[s:e]]
            c = self.alpha * H.data[s:e]  # implicit's convention: confidence = alpha * r
            A = YtY + (Yh.T * (c - 1.0)) @ Yh + reg
            U[r] = np.linalg.solve(A, Yh.T @ c)
        return U

    def score(self, q: Queries) -> F32:
        assert self.Y is not None
        return np.asarray(self.user_vectors(q) @ self.Y.T, np.float32)


@dataclass
class BPR(Recommender):
    name: ClassVar[str] = "bpr"
    factors: int = 64
    learning_rate: float = 0.01
    regularization: float = 0.01
    iterations: int = 100
    Y: F32 | None = field(default=None, repr=False)
    U: F32 | None = field(default=None, repr=False)

    @classmethod
    def space(cls, trial: Any) -> dict[str, Any]:
        return {
            "factors": trial.suggest_categorical("factors", [16, 32, 64, 128, 256]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
            "regularization": trial.suggest_float("regularization", 1e-5, 1.0, log=True),
            "iterations": trial.suggest_int("iterations", 10, 300, log=True),
        }

    def fit(self, log: Log, catalog: Catalog) -> Self:
        from implicit.cpu.bpr import BayesianPersonalizedRanking

        X = _interaction_matrix(log)
        m = BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.seed,
            num_threads=1,
        )
        m.fit(X, show_progress=False)
        self.Y = np.asarray(m.item_factors, np.float32)
        self.U = np.asarray(m.user_factors, np.float32)[: log.n_users]
        self._trained_users = np.unique(log.user[log.user >= 0])
        return self

    def score(self, q: Queries) -> F32:
        assert self.Y is not None and self.U is not None
        H = q.matrix(self.Y.shape[0], binary=True)
        cnt = np.maximum(np.diff(H.indptr), 1)[:, None]
        V = np.asarray(H @ self.Y) / cnt
        # implicit's BPR appends an item-bias column and a constant 1 to user factors
        V[:, -1] = 1.0
        known = np.isin(q.user, self._trained_users)
        V[known] = self.U[q.user[known]]
        return np.asarray(V @ self.Y.T, np.float32)
