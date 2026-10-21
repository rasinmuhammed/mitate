"""Canonical data model: one store's interaction log, its catalog, and ragged query batches.

Ids are dense ``0..n-1`` per store. Every array is numpy so models never pay dataframe overhead
in their inner loops. ``Log.until(t)`` is the only sanctioned way to obtain training data, which is
what the leakage tests hook into.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

I32 = NDArray[np.int32]
I64 = NDArray[np.int64]
F32 = NDArray[np.float32]


class Event(IntEnum):
    VIEW = 0
    CART = 1
    ORDER = 2


DEFAULT_WEIGHTS: Mapping[Event, float] = {Event.VIEW: 1.0, Event.CART: 3.0, Event.ORDER: 6.0}


@dataclass(frozen=True)
class Log:
    """Interactions sorted by (ts, user, item). ``session`` groups events of one visit; for
    transactional data it is the basket (order) id. ``-1`` user means anonymous."""

    user: I32
    item: I32
    ts: I64
    event: NDArray[np.int8]
    session: I64
    n_users: int
    n_items: int
    horizon: int = field(default=np.iinfo(np.int64).max)  # exclusive upper bound on ts

    def __post_init__(self) -> None:
        n = self.item.size
        for name in ("user", "ts", "event", "session"):
            if getattr(self, name).size != n:
                raise ValueError(f"column {name} has wrong length")
        if n and (self.ts.max() >= self.horizon):
            raise ValueError("log contains events at or after its horizon")

    def __len__(self) -> int:
        return int(self.item.size)

    @staticmethod
    def build(
        user: Any, item: Any, ts: Any, event: Any, session: Any, n_users: int, n_items: int
    ) -> Log:
        u = np.asarray(user, np.int32)
        i = np.asarray(item, np.int32)
        t = np.asarray(ts, np.int64)
        order = np.lexsort((i, u, t))
        return Log(
            user=u[order],
            item=i[order],
            ts=t[order],
            event=np.asarray(event, np.int8)[order],
            session=np.asarray(session, np.int64)[order],
            n_users=n_users,
            n_items=n_items,
        )

    def _mask(self, m: NDArray[np.bool_], horizon: int | None = None) -> Log:
        return Log(
            user=self.user[m],
            item=self.item[m],
            ts=self.ts[m],
            event=self.event[m],
            session=self.session[m],
            n_users=self.n_users,
            n_items=self.n_items,
            horizon=self.horizon if horizon is None else horizon,
        )

    def until(self, t: int) -> Log:
        """Events strictly before ``t``; the result refuses to hold anything later."""
        hi = int(np.searchsorted(self.ts, t, side="left"))
        m = np.zeros(len(self), dtype=bool)
        m[:hi] = True
        return self._mask(m, horizon=min(t, self.horizon))

    def between(self, t0: int, t1: int) -> Log:
        lo = int(np.searchsorted(self.ts, t0, side="left"))
        hi = int(np.searchsorted(self.ts, t1, side="left"))
        m = np.zeros(len(self), dtype=bool)
        m[lo:hi] = True
        return self._mask(m, horizon=min(t1, self.horizon))

    def where(self, m: NDArray[np.bool_]) -> Log:
        return self._mask(m)

    def user_item(
        self,
        weights: Mapping[Event, float] | None = DEFAULT_WEIGHTS,
        binary: bool = False,
        n_users: int | None = None,
    ) -> sp.csr_matrix:
        """User x item matrix of summed event weights (anonymous rows dropped)."""
        known = self.user >= 0
        u, i, e = self.user[known], self.item[known], self.event[known]
        if binary or weights is None:
            v: F32 = np.ones(u.size, np.float32)
        else:
            w = np.zeros(int(max(Event)) + 1, np.float32)
            for k, val in weights.items():
                w[int(k)] = val
            v = w[e]
        X = sp.csr_matrix(
            (v, (u, i)), shape=(n_users or self.n_users, self.n_items), dtype=np.float32
        )
        X.sum_duplicates()
        if binary:
            X.data[:] = 1.0
        return X

    def item_counts(self, distinct_users: bool = False) -> NDArray[np.float64]:
        if distinct_users:
            known = self.user >= 0
            key = np.unique(self.user[known].astype(np.int64) * self.n_items + self.item[known])
            return np.bincount(key % self.n_items, minlength=self.n_items).astype(np.float64)
        return np.bincount(self.item, minlength=self.n_items).astype(np.float64)


@dataclass(frozen=True)
class Catalog:
    """Item side information. Every field is optional except the item count."""

    n_items: int
    title: Sequence[str] | None = None
    text: Sequence[str] | None = None  # description + attributes rendered as text
    category: I32 | None = None  # leaf category id per item
    category_names: Sequence[str] | None = None
    category_parent: I32 | None = None  # parent id per category (-1 = root)
    brand: I32 | None = None
    brand_names: Sequence[str] | None = None  # dense brand id -> name
    price: NDArray[np.float64] | None = None
    first_seen: I64 | None = None  # catalog launch time when known; else first interaction
    raw_ids: Sequence[Any] | None = None  # original dataset ids, for traceability

    def subset(self, items: I64) -> Catalog:
        def take(x: Any) -> Any:
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                return x[items]
            return [x[j] for j in items]

        return replace(
            self,
            n_items=int(items.size),
            title=take(self.title),
            text=take(self.text),
            category=take(self.category),
            brand=take(self.brand),
            price=take(self.price),
            first_seen=take(self.first_seen),
            raw_ids=take(self.raw_ids),
        )

    def price_band(self, n_bands: int = 5) -> I32:
        """Quantile price band per item; -1 when unknown. Bands are catalog-only, so leak-free."""
        if self.price is None:
            return np.full(self.n_items, -1, np.int32)
        p = self.price
        ok = np.isfinite(p) & (p > 0)
        out = np.full(self.n_items, -1, np.int32)
        if ok.sum() >= n_bands:
            edges = np.quantile(p[ok], np.linspace(0, 1, n_bands + 1)[1:-1])
            out[ok] = np.searchsorted(edges, p[ok], side="right").astype(np.int32)
        return out


@dataclass(frozen=True)
class Queries:
    """A ragged batch of recommendation requests.

    Query q's history is ``items[indptr[q]:indptr[q+1]]`` (with ts/event), oldest first, and
    includes everything the shopper did before ``now[q]``: past visits and the current session.
    ``session_start[q]`` is the index (relative to the query's slice) where the current session
    begins, so session models can separate the live session from older history.
    ``user[q]`` is -1 for anonymous shoppers. ``anchor[q]`` is the item on screen (product page),
    -1 if none. ``session`` gives each history event's session or basket id (basket models group
    by it).
    """

    indptr: I64
    items: I32
    ts: I64
    event: NDArray[np.int8]
    now: I64
    user: I32
    session_start: I64
    anchor: I32
    session: I64 | None = None

    def __len__(self) -> int:
        return int(self.now.size)

    def history(self, q: int) -> I32:
        return self.items[self.indptr[q] : self.indptr[q + 1]]

    def slice(self, s: int, e: int) -> Queries:
        a, b = int(self.indptr[s]), int(self.indptr[e])
        return Queries(
            indptr=self.indptr[s : e + 1] - a,
            items=self.items[a:b],
            ts=self.ts[a:b],
            event=self.event[a:b],
            now=self.now[s:e],
            user=self.user[s:e],
            session_start=self.session_start[s:e],
            anchor=self.anchor[s:e],
            session=None if self.session is None else self.session[a:b],
        )

    def sessions(self, q: int) -> I64:
        """Session ids of query q's history (falls back to one session per event)."""
        a, b = int(self.indptr[q]), int(self.indptr[q + 1])
        if self.session is None:
            return np.arange(a, b, dtype=np.int64)
        return self.session[a:b]

    def lengths(self) -> I64:
        return np.diff(self.indptr)

    def matrix(
        self,
        n_items: int,
        weights: Mapping[Event, float] | None = DEFAULT_WEIGHTS,
        binary: bool = False,
        decay_half_life: float | None = None,
    ) -> sp.csr_matrix:
        """Query x item matrix of (optionally time-decayed) event weights."""
        rows = np.repeat(np.arange(len(self)), self.lengths())
        if binary or weights is None:
            v: F32 = np.ones(self.items.size, np.float32)
        else:
            w = np.zeros(int(max(Event)) + 1, np.float32)
            for k, val in weights.items():
                w[int(k)] = val
            v = w[self.event]
        if decay_half_life:
            age = (self.now[rows] - self.ts).astype(np.float64)
            v = v * np.exp2(-age / decay_half_life).astype(np.float32)
        X = sp.csr_matrix((v, (rows, self.items)), shape=(len(self), n_items), dtype=np.float32)
        X.sum_duplicates()
        if binary:
            X.data[:] = 1.0
        return X

    @staticmethod
    def from_lists(
        histories: Sequence[Sequence[int]],
        now: Sequence[int] | None = None,
        anchor: Sequence[int] | None = None,
    ) -> Queries:
        """Convenience constructor (tests, serving): plain item histories, uniform timestamps."""
        lens = np.array([len(h) for h in histories], np.int64)
        indptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        items = (
            np.concatenate([np.asarray(h, np.int32) for h in histories])
            if len(histories) and lens.sum()
            else np.zeros(0, np.int32)
        )
        n = len(histories)
        nw = np.asarray(now if now is not None else [10**9] * n, np.int64)
        ts = (
            np.concatenate([nw[q] - (lens[q] - np.arange(lens[q])) for q in range(n)]).astype(
                np.int64
            )
            if lens.sum()
            else np.zeros(0, np.int64)
        )
        return Queries(
            indptr=indptr,
            items=items,
            ts=ts,
            event=np.zeros(items.size, np.int8),
            now=nw,
            user=np.full(n, -1, np.int32),
            session_start=np.zeros(n, np.int64),
            anchor=np.asarray(anchor if anchor is not None else [-1] * n, np.int32),
        )
