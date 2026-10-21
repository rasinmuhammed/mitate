"""Session-based nearest neighbours: VSTAN (Ludewig et al., ACM TOIS 2021), which contains STAN
(Garg et al., SIGIR 2019) and plain SKNN as special cases.

Formulas follow docs/research_notes/02_sequential_session.md section D, re-implemented from the
equations (the session-rec reference code has no license). Two documented deviations from that
code: IDF counts sessions rather than events, and the "most recent shared position" i* is the
maximum position, not iteration-order dependent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self

import numba as nb
import numpy as np
from numpy.typing import NDArray

from mitate.data import I32, I64, Catalog, Log, Queries
from mitate.models.base import F32, Recommender

DAY = 86_400.0


def _sessions(log: Log) -> tuple[I64, I32, I64]:
    """CSR of training sessions: items in order with duplicates reduced to their last occurrence,
    plus each session's last timestamp."""
    order = np.lexsort((np.arange(len(log)), log.ts, log.session))
    s, it, ts = log.session[order], log.item[order], log.ts[order]
    # keep the last occurrence of each (session, item)
    key = s.astype(np.int64) * (log.n_items + 1) + it
    rev_first = np.unique(key[::-1], return_index=True)[1]
    keep = np.zeros(key.size, bool)
    keep[key.size - 1 - rev_first] = True
    s, it, ts = s[keep], it[keep], ts[keep]
    bounds = np.flatnonzero(np.diff(s)) + 1
    ptr = np.concatenate([[0], bounds, [s.size]]).astype(np.int64)
    last_ts = ts[ptr[1:] - 1] if s.size else np.zeros(0, np.int64)
    return ptr, it.astype(np.int32), last_ts.astype(np.int64)


@nb.njit(cache=True)
def _score_batch(
    q_ptr: I64,
    q_items: I32,
    q_now: I64,
    s_ptr: I64,
    s_items: I32,
    s_ts: I64,
    i_ptr: I64,
    i_sess: I64,
    idf: NDArray[np.float64],
    n_items: int,
    k: int,
    sample: int,
    max_len: int,
    l_spw: float,
    l_snh: float,
    l_inh: float,
    l_ipw: float,
    l_idf: float,
) -> F32:
    nq = q_ptr.size - 1
    out = np.zeros((nq, n_items), np.float32)
    pos_in_q = np.full(n_items, -1, np.int64)
    seen = np.zeros(s_ptr.size - 1, np.int64) - 1
    for q in range(nq):
        a, b = q_ptr[q], q_ptr[q + 1]
        # last max_len distinct-consecutive items, positions 0..L-1 (L-1 newest)
        tmp = q_items[a:b]
        if tmp.size == 0:
            continue
        start = max(0, tmp.size - max_len)
        cur = tmp[start:]
        L = cur.size
        for p in range(L):
            pos_in_q[cur[p]] = p  # last occurrence wins
        wsq = 0.0
        for p in range(L):
            if pos_in_q[cur[p]] == p:
                w = math.exp((p + 1 - L) / l_spw)
                wsq += w * w
        # candidate pool: most recent `sample` sessions sharing an item
        cand: I64 = np.empty(0, np.int64)
        buf = []
        for p in range(L):
            it = cur[p]
            if pos_in_q[it] != p:
                continue
            for j in range(i_ptr[it], min(i_ptr[it + 1], i_ptr[it] + sample)):
                sid = i_sess[j]
                if seen[sid] != q:
                    seen[sid] = q
                    buf.append(sid)
        if len(buf) == 0:
            for p in range(L):
                pos_in_q[cur[p]] = -1
            continue
        cand = np.array(buf, dtype=np.int64)
        if cand.size > sample:
            order = np.argsort(-s_ts[cand])[:sample]
            cand = cand[order]
        # similarity
        sims = np.zeros(cand.size)
        istar = np.zeros(cand.size, np.int64)
        pstar = np.zeros(cand.size, np.int64)
        for c in range(cand.size):
            sid = cand[c]
            lo, hi = s_ptr[sid], s_ptr[sid + 1]
            acc = 0.0
            best_q = -1
            best_n = -1
            for j in range(lo, hi):
                p = pos_in_q[s_items[j]]
                if p >= 0:
                    acc += math.exp((p + 1 - L) / l_spw)
                    if p > best_q:
                        best_q = p
                    best_n = j - lo  # positions ascend, so this ends as the last shared one
            if acc == 0.0:
                continue
            age = max(0.0, (q_now[q] - s_ts[sid]) / 86400.0)
            sims[c] = acc / (math.sqrt(wsq) * math.sqrt(hi - lo)) * math.exp(-age / l_snh)
            istar[c] = best_q
            pstar[c] = best_n
        kk = min(k, cand.size)
        top = np.argsort(-sims)[:kk]
        for t in range(kk):
            c = top[t]
            if sims[c] <= 0.0:
                continue
            sid = cand[c]
            lo, hi = s_ptr[sid], s_ptr[sid + 1]
            seqw = math.exp((istar[c] + 1 - L) / l_ipw) if l_ipw > 0 else 1.0
            for j in range(lo, hi):
                d = math.exp(-abs((j - lo) - pstar[c]) / l_inh)
                out[q, s_items[j]] += sims[c] * d * seqw
        if l_idf > 0:
            for i in range(n_items):
                if out[q, i] != 0.0:
                    out[q, i] *= 1.0 + l_idf * idf[i]
        for p in range(L):
            pos_in_q[cur[p]] = -1
    return out


@dataclass
class VSTAN(Recommender):
    """``l_ipw = 0`` and ``l_idf = 0`` give STAN; large ``l_spw``, ``l_snh`` and ``l_inh`` approach
    plain SKNN."""

    name: ClassVar[str] = "vstan"
    k: int = 100
    sample: int = 1000
    max_len: int = 20
    l_spw: float = 1.02
    l_snh_days: float = 5.0
    l_inh: float = 2.05
    l_ipw: float = 1.02
    l_idf: float = 5.0
    _tables: tuple[NDArray[Any], ...] | None = field(default=None, repr=False)
    _bg: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0), repr=False)

    @classmethod
    def space(cls, trial: Any) -> dict[str, Any]:
        return {
            "k": trial.suggest_categorical("k", [50, 100, 200, 500, 1000]),
            "sample": trial.suggest_categorical("sample", [500, 1000, 2500, 5000]),
            "max_len": trial.suggest_categorical("max_len", [5, 10, 20, 50]),
            "l_spw": trial.suggest_float("l_spw", 0.25, 10.0, log=True),
            "l_snh_days": trial.suggest_float("l_snh_days", 0.5, 365.0, log=True),
            "l_inh": trial.suggest_float("l_inh", 0.25, 10.0, log=True),
            "l_ipw": trial.suggest_categorical("l_ipw", [0.0, 0.5, 1.02, 2.0, 5.0]),
            "l_idf": trial.suggest_categorical("l_idf", [0.0, 1.0, 5.0, 10.0]),
        }

    def fit(self, log: Log, catalog: Catalog) -> Self:
        s_ptr, s_items, s_ts = _sessions(log)
        n_sess = s_ptr.size - 1
        sess_of = np.repeat(np.arange(n_sess), np.diff(s_ptr))
        # inverted index item -> sessions, most recent first
        order = np.lexsort((-s_ts[sess_of], s_items))
        i_sess = sess_of[order].astype(np.int64)
        i_ptr = np.concatenate([[0], np.cumsum(np.bincount(s_items, minlength=log.n_items))])
        df = np.bincount(s_items, minlength=log.n_items).astype(np.float64)
        idf = np.where(df > 0, np.log(max(n_sess, 1) / np.maximum(df, 1)), 0.0)
        self._tables = (s_ptr, s_items, s_ts, i_ptr.astype(np.int64), i_sess, idf)
        cnt = log.item_counts()
        self._bg = (cnt / (cnt.max() * 1e9)) if cnt.max() > 0 else cnt
        self._n_items = log.n_items
        return self

    def score(self, q: Queries) -> F32:
        assert self._tables is not None
        s_ptr, s_items, s_ts, i_ptr, i_sess, idf = self._tables
        S = _score_batch(
            q.indptr,
            q.items,
            q.now,
            s_ptr,
            s_items,
            s_ts,
            i_ptr,
            i_sess,
            idf,
            self._n_items,
            self.k,
            self.sample,
            self.max_len,
            self.l_spw,
            self.l_snh_days,
            self.l_inh,
            self.l_ipw,
            self.l_idf,
        )
        S += self._bg[None, :].astype(np.float32)
        return S
