"""Pairwise-transition sequence models: Markov chain, PCTM and SeqRules.

Definitions follow docs/research_notes/02_sequential_session.md (Implementation spec), which were
checked against the Apache-2.0 reference code of Petrov et al., "Do Sequential Recommendation
Benchmarks Really Require Higher-Order Sequence Modelling?" (RecSys 2026, arXiv 2608.19833) and
Ludewig & Jannach (UMUAI 2018).

A training sequence is one known user's time-ordered events, or one anonymous session. Consecutive
duplicates are collapsed (a shopper re-viewing the same product is not a transition).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Self

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from mitate.data import I32, I64, Catalog, Log, Queries
from mitate.models.base import F32, Recommender


# --------------------------------------------------------------------------- sequences
def training_sequences(log: Log) -> tuple[I64, I32]:
    """(seq_id per event, item per event), grouped by sequence, time-ordered, duplicates collapsed.

    Sequence key: user id for known users, otherwise -(session+1) so the two never collide.
    """
    key = np.where(log.user >= 0, log.user.astype(np.int64), -(log.session + 1))
    order = np.lexsort((np.arange(len(log)), log.ts, key))  # stable within (key, ts)
    k, it = key[order], log.item[order]
    keep = np.ones(k.size, bool)
    keep[1:] = (k[1:] != k[:-1]) | (it[1:] != it[:-1])
    return k[keep], it[keep]


def kernel(spec: str, d: NDArray[Any]) -> NDArray[np.float64]:
    """Distance/position kernels. ``spec`` is ``name`` or ``name:param``."""
    name, _, p = spec.partition(":")
    d = d.astype(np.float64)
    if name in ("u", "same"):
        return np.ones_like(d)
    if name == "exp":
        return float(p) ** (d - 1)
    if name == "inv":
        return d ** -float(p or 1.0)
    if name == "div":
        return 1.0 / d
    if name == "quadratic":
        return 1.0 / d**2
    if name == "log":
        return 1.0 / np.log10(d + 1.7)
    if name == "linear":
        return np.maximum(0.0, 1.0 - 0.1 * d)
    raise ValueError(f"unknown kernel {spec}")


def pair_counts(
    seq: I64, items: I32, n_items: int, window: int, count_kernel: str
) -> sp.csr_matrix:
    """C[b, a] = sum_{d=1..window} gamma(d) * #(a occurs exactly d steps after b in a sequence)."""
    rows, cols, vals = [], [], []
    for d in range(1, window + 1):
        if d >= items.size:
            break
        same = seq[d:] == seq[:-d]
        if not same.any():
            break
        rows.append(items[:-d][same])
        cols.append(items[d:][same])
        vals.append(np.full(int(same.sum()), kernel(count_kernel, np.array([d]))[0]))
    if not rows:
        return sp.csr_matrix((n_items, n_items))
    C = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_items, n_items),
    )
    C.sum_duplicates()
    return C


def history_weights(q: Queries, n_items: int, hist_kernel: str, max_len: int) -> sp.csr_matrix:
    """Query x item weights from the last ``max_len`` distinct-consecutive history items.

    ``hist_kernel``: ``last`` | ``exp:r`` | ``inv:p`` | ``tail:k:rho:r`` (head-tail) | SeqRules
    position kernels (``div``, ``quadratic``, ...) where the newest item has weight 1.
    """
    rows, cols, vals = [], [], []
    for r in range(len(q)):
        h = q.history(r)
        if h.size == 0:
            continue
        keep = np.ones(h.size, bool)
        keep[1:] = h[1:] != h[:-1]
        h = h[keep][::-1][:max_len]  # newest first
        L = h.size
        j = np.arange(1, L + 1, dtype=np.float64)
        name, _, rest = hist_kernel.partition(":")
        if name == "last":
            w = (j == 1).astype(np.float64)
        elif name == "exp":
            w = float(rest) ** (j - 1)
            w /= w.sum()
        elif name == "inv":
            w = j ** -float(rest)
            w /= w.sum()
        elif name == "tail":
            k_s, rho_s, r_s = rest.split(":")
            k, rho, rr = int(k_s), float(rho_s), float(r_s)
            head = rr ** (np.arange(min(k, L)))
            if k >= L:
                w = head / head.sum()
            else:
                w = np.concatenate([rho * head / head.sum(), np.full(L - k, (1 - rho) / (L - k))])
        else:  # SeqRules position weights: v(1) = 1, v(j) = g(j) for j >= 2
            w = np.ones(L)
            if L > 1:
                w[1:] = kernel(hist_kernel, j[1:])
        rows.append(np.full(L, r))
        cols.append(h)
        vals.append(w)
    if not rows:
        return sp.csr_matrix((len(q), n_items))
    W = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(q), n_items),
    )
    W.sum_duplicates()
    return W


def _background(pop: NDArray[np.float64]) -> NDArray[np.float64]:
    """Tiny, strictly-below-any-evidence popularity score used only to order fill-in items."""
    m = pop.max() if pop.size else 0.0
    return (pop / (m * 1e6)).astype(np.float64) if m > 0 else np.zeros_like(pop, np.float64)


# --------------------------------------------------------------------------- models
@dataclass
class MarkovChain(Recommender):
    """First-order MC on the last item, popularity fill. Sanity baseline."""

    name: ClassVar[str] = "markov"
    P: sp.csr_matrix | None = field(default=None, repr=False)
    bg: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0), repr=False)

    def fit(self, log: Log, catalog: Catalog) -> Self:
        seq, items = training_sequences(log)
        C = pair_counts(seq, items, log.n_items, 1, "u")
        rs = np.asarray(C.sum(axis=1)).ravel()
        rs[rs == 0] = 1
        self.P = sp.diags(1.0 / rs) @ C
        self.bg = _background(log.item_counts())
        return self

    def score(self, q: Queries) -> F32:
        assert self.P is not None
        W = history_weights(q, self.P.shape[0], "last", 1)
        return np.asarray((W @ self.P).toarray() + self.bg, np.float32)


@dataclass
class PCTM(Recommender):
    name: ClassVar[str] = "pctm"
    window: int = 10
    count_kernel: str = "inv:0.5"
    tau_rel: float = 1.0  # tau = tau_rel * |I|
    hist_kernel: str = "exp:0.6"
    max_len: int = 50
    lam: float = 0.0
    E: sp.csr_matrix | None = field(default=None, repr=False)
    logpop: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0), repr=False)

    @classmethod
    def space(cls, trial: Any) -> dict[str, Any]:
        return {
            "window": trial.suggest_categorical("window", [3, 5, 10, 20]),
            "count_kernel": trial.suggest_categorical(
                "count_kernel", ["inv:0.5", "inv:1", "exp:0.5", "exp:0.7", "u"]
            ),
            "tau_rel": trial.suggest_float("tau_rel", 0.01, 10.0, log=True),
            "hist_kernel": trial.suggest_categorical(
                "hist_kernel", ["last", "exp:0.4", "exp:0.6", "exp:0.8", "tail:7:0.8:0.6"]
            ),
            "lam": trial.suggest_float("lam", -1.0, 0.5),
        }

    def fit(self, log: Log, catalog: Catalog) -> Self:
        seq, items = training_sequences(log)
        n = log.n_items
        C = pair_counts(seq, items, n, self.window, self.count_kernel)
        tau = self.tau_rel * n
        E = C.copy()
        E.data = np.log1p(n * C.data / tau)
        self.E = E.tocsr()
        f = log.item_counts()
        self.logpop = np.log((f + 1.0) / (f.sum() + n))
        return self

    def score(self, q: Queries) -> F32:
        assert self.E is not None
        W = history_weights(q, self.E.shape[0], self.hist_kernel, self.max_len)
        S = (W @ self.E).toarray() + self.lam * self.logpop[None, :]
        return np.asarray(S, np.float32)


@dataclass
class SeqRules(Recommender):
    name: ClassVar[str] = "seqrules"
    window: int = 10
    rule_kernel: str = "log"
    idf: bool = False
    prune: int = 0  # 0 = keep all
    max_len: int = 10
    hist_kernel: str = "quadratic"
    R: sp.csr_matrix | None = field(default=None, repr=False)
    bg: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0), repr=False)

    @classmethod
    def space(cls, trial: Any) -> dict[str, Any]:
        return {
            "window": trial.suggest_categorical("window", [1, 3, 5, 10, 20, 30]),
            "rule_kernel": trial.suggest_categorical(
                "rule_kernel", ["same", "div", "quadratic", "log", "linear"]
            ),
            "idf": trial.suggest_categorical("idf", [False, True]),
            "prune": trial.suggest_categorical("prune", [0, 20, 100, 200]),
            "max_len": trial.suggest_categorical("max_len", [1, 3, 5, 10, 20]),
            "hist_kernel": trial.suggest_categorical(
                "hist_kernel", ["same", "div", "quadratic", "log", "linear"]
            ),
        }

    def fit(self, log: Log, catalog: Catalog) -> Self:
        from mitate.models.baselines import _topk_sparse_rows

        seq, items = training_sequences(log)
        n = log.n_items
        R = pair_counts(seq, items, n, self.window, self.rule_kernel)
        if self.idf:
            f = np.bincount(items, minlength=n).astype(np.float64)
            U = max(np.unique(seq).size, 1)
            raw = np.log(U / np.maximum(f, 1))
            lo, hi = raw.min(), raw.max()
            w = (raw - lo) / (hi - lo) if hi > lo else np.ones(n)
            R = sp.diags(w) @ R
        if self.prune:
            R = _topk_sparse_rows(R.tocsr(), self.prune)
        self.R = R.tocsr()
        self.bg = _background(log.item_counts())
        return self

    def score(self, q: Queries) -> F32:
        assert self.R is not None
        W = history_weights(q, self.R.shape[0], self.hist_kernel, self.max_len)
        return np.asarray((W @ self.R).toarray() + self.bg, np.float32)
