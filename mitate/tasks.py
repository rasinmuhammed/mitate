"""Evaluation tasks under a global temporal split (Gusak et al., "Time to Split", RecSys 2025).

Models are always fitted on ``log.until(t0)``. Targets come from ``[t0, t1)``. Each target is its
own query whose history is every event of that shopper strictly before the target time, including
window events that precede it: a deployed model keeps serving live histories until its next retrain,
but those events never reach training.

Target selection (``target=``)
------------------------------
``successive``  every eligible target in the window (protocol of record for test)
``last``        each shopper's last eligible target (used for validation, as in Time to Split)
``random``      one uniformly drawn eligible target per shopper (seeded)
``first``       the first one; only to reproduce other protocols (Time to Split advises against)

Every query carries ``unit``: the user id for known shoppers, ``-(session+1)`` for anonymous ones.
Per-target metrics are averaged within unit before averaging across units and before any paired
test, so heavy shoppers do not dominate and the test's independence assumption holds.

Tasks
-----
``next_basket``  Returning-customer placement. Targets are ORDER baskets of known shoppers.
``next_item``    Session placement. Targets are positions 1..n-1 of sessions starting in the window
                 (consecutive duplicates collapsed; the first item has no in-session context and is
                 never a target, as in Time to Split).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from mitate.data import I64, Event, Log, Queries

Target = Literal["successive", "last", "random", "first"]


class EmptyWindowError(ValueError):
    """An evaluation window holds no query, so the store cannot be scored on it."""


@dataclass(frozen=True)
class EvalSet:
    queries: Queries
    targets: sp.csr_matrix
    exclude: sp.csr_matrix | None
    history: sp.csr_matrix  # binary pre-query history, for repeat/explore splits
    unit: I64  # averaging unit per query
    segments: dict[str, NDArray[Any]] = field(default_factory=dict)
    t0: int = 0
    t1: int = 0
    task: str = ""

    def __len__(self) -> int:
        return len(self.queries)

    def subset(self, idx: NDArray[np.integer[Any]]) -> EvalSet:
        idx = np.asarray(idx, np.int64)
        q = self.queries
        starts, ends = q.indptr[idx], q.indptr[idx + 1]
        lens = ends - starts
        flat = (
            np.concatenate([np.arange(a, b) for a, b in zip(starts, ends, strict=True)])
            if (idx.size)
            else np.zeros(0, np.int64)
        )
        qq = Queries(
            indptr=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
            items=q.items[flat],
            ts=q.ts[flat],
            event=q.event[flat],
            now=q.now[idx],
            user=q.user[idx],
            session_start=q.session_start[idx],
            anchor=q.anchor[idx],
            session=None if q.session is None else q.session[flat],
        )
        return EvalSet(
            qq,
            self.targets[idx],
            None if self.exclude is None else self.exclude[idx],
            self.history[idx],
            self.unit[idx],
            {k: v[idx] for k, v in self.segments.items()},
            self.t0,
            self.t1,
            self.task,
        )


class _UserIndex:
    """Known-user events sorted by (user, ts) for fast "history before t" lookups."""

    def __init__(self, log: Log) -> None:
        known = log.user >= 0
        order = np.lexsort((np.arange(int(known.sum())), log.ts[known], log.user[known]))
        self.u = log.user[known][order]
        self.it = log.item[known][order]
        self.ts = log.ts[known][order]
        self.ev = log.event[known][order]
        self.sess = log.session[known][order]

    def before(self, user: int, t: int) -> slice:
        s = int(np.searchsorted(self.u, user, side="left"))
        e = int(np.searchsorted(self.u, user, side="right"))
        e = s + int(np.searchsorted(self.ts[s:e], t, side="left"))
        return slice(s, e)


@dataclass
class _Builder:
    n_items: int
    items: list[NDArray[np.integer[Any]]] = field(default_factory=list)
    ts: list[I64] = field(default_factory=list)
    ev: list[NDArray[np.integer[Any]]] = field(default_factory=list)
    sess: list[NDArray[np.integer[Any]]] = field(default_factory=list)
    now: list[int] = field(default_factory=list)
    user: list[int] = field(default_factory=list)
    sstart: list[int] = field(default_factory=list)
    unit: list[int] = field(default_factory=list)
    tr: list[I64] = field(default_factory=list)
    tc: list[NDArray[np.integer[Any]]] = field(default_factory=list)
    extra: dict[str, list[float]] = field(default_factory=dict)

    def add(self, items, ts, ev, sess, now, user, sstart, unit, targets, **extra) -> None:  # type: ignore[no-untyped-def]
        r = len(self.now)
        self.items.append(items)
        self.ts.append(ts)
        self.ev.append(ev)
        self.sess.append(sess)
        self.now.append(int(now))
        self.user.append(int(user))
        self.sstart.append(int(sstart))
        self.unit.append(int(unit))
        t = np.unique(targets)
        self.tr.append(np.full(t.size, r))
        self.tc.append(t)
        for k, v in extra.items():
            self.extra.setdefault(k, []).append(v)

    def build(self, task: str, t0: int, t1: int, exclude_seen: bool) -> EvalSet:
        if not self.now:
            raise ValueError(f"{task}: no eligible targets in window")
        lens = np.array([a.size for a in self.items], np.int64)

        def cat(xs: Sequence[NDArray[np.integer[Any]]], dt: type) -> NDArray[Any]:
            return np.concatenate(xs).astype(dt) if lens.sum() else np.zeros(0, dt)

        q = Queries(
            indptr=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
            items=cat(self.items, np.int32),
            ts=cat(self.ts, np.int64),
            event=cat(self.ev, np.int8),
            now=np.asarray(self.now, np.int64),
            user=np.asarray(self.user, np.int32),
            session_start=np.asarray(self.sstart, np.int64),
            anchor=np.full(len(self.now), -1, np.int32),
            session=cat(self.sess, np.int64),
        )
        rows, cols = np.concatenate(self.tr), np.concatenate(self.tc)
        T = sp.csr_matrix(
            (np.ones(rows.size, np.float32), (rows, cols)), shape=(len(self.now), self.n_items)
        )
        H = q.matrix(self.n_items, binary=True)
        hist_len = np.asarray(self.sstart, np.int64)
        seg = {
            "returning": hist_len > 0,
            "hist_len": hist_len,
            "hist_bucket": hist_bucket(hist_len),
            "known_user": np.asarray(self.user) >= 0,
        }
        seg |= {k: np.asarray(v) for k, v in self.extra.items()}
        ev = EvalSet(
            q, T, H if exclude_seen else None, H, np.asarray(self.unit, np.int64), seg, t0, t1, task
        )
        return drop_unreachable(ev) if exclude_seen else ev


def hist_bucket(n: I64) -> NDArray[np.intp]:
    """0 | 1-2 | 3-5 | 6-10 | 11-20 | 21-50 | 51+ prior events."""
    return np.searchsorted(np.array([0, 1, 3, 6, 11, 21, 51]), n, side="right") - 1


def _select(n: int, target: Target, rng: np.random.Generator) -> NDArray[np.intp]:
    if target == "successive":
        return np.arange(n)
    if target == "last":
        return np.array([n - 1])
    if target == "first":
        return np.array([0])
    return np.array([int(rng.integers(n))])


def next_basket(
    log: Log,
    t0: int,
    t1: int,
    target: Target = "successive",
    exclude_seen: bool = False,
    seed: int = 0,
    max_units: int | None = None,
) -> EvalSet:
    """One query per selected ORDER basket of a known shopper in ``[t0, t1)``."""
    rng = np.random.default_rng(seed)
    win = log.between(t0, t1)
    m = (win.event == Event.ORDER) & (win.user >= 0)
    if not m.any():
        raise EmptyWindowError("next_basket: no orders in window")
    u, it, ts, s = win.user[m], win.item[m], win.ts[m], win.session[m]
    # basket = (user, session); basket time = its first order event
    order = np.lexsort((ts, s, u))
    u, it, ts, s = u[order], it[order], ts[order], s[order]
    new_b = np.ones(u.size, bool)
    new_b[1:] = (u[1:] != u[:-1]) | (s[1:] != s[:-1])
    b_start = np.flatnonzero(new_b)
    b_end = np.append(b_start[1:], u.size)
    b_user, b_time, b_sess = u[b_start], ts[b_start], s[b_start]
    # Never split an order across the cutoff: baskets that already had events before t0 are out.
    past = log.until(t0)
    pk = past.user[past.user >= 0].astype(np.int64) * (1 << 32) + past.session[past.user >= 0]
    started_before = np.isin(b_user.astype(np.int64) * (1 << 32) + b_sess, pk)
    b_user = np.where(started_before, -1, b_user)
    users = np.unique(b_user[b_user >= 0])
    if max_units is not None and users.size > max_units:
        users = np.sort(rng.choice(users, max_units, replace=False))
    idx = _UserIndex(log)
    out = _Builder(log.n_items)
    for user in users:
        bs = np.flatnonzero(b_user == user)
        bs = bs[np.argsort(b_time[bs], kind="stable")]
        for j in _select(bs.size, target, rng):
            b = bs[j]
            sl = idx.before(int(user), int(b_time[b]))
            out.add(
                idx.it[sl],
                idx.ts[sl],
                idx.ev[sl],
                idx.sess[sl],
                b_time[b],
                user,
                sl.stop - sl.start,
                user,
                it[b_start[b] : b_end[b]],
                basket_size=float(np.unique(it[b_start[b] : b_end[b]]).size),
            )
    return out.build(f"next_basket/{target}", t0, t1, exclude_seen)


def next_item(
    log: Log,
    t0: int,
    t1: int,
    target: Target = "successive",
    exclude_seen: bool = False,
    seed: int = 0,
    max_units: int | None = None,
    target_events: tuple[Event, ...] | None = None,
) -> EvalSet:
    """One query per selected position of sessions that start inside ``[t0, t1)``.

    ``target_events`` restricts which events may be targets (e.g. only carts/orders); the history
    always contains every event type.
    """
    rng = np.random.default_rng(seed)
    order = np.lexsort((np.arange(len(log)), log.ts, log.session))
    s, it, ts, ev, us = (
        log.session[order],
        log.item[order],
        log.ts[order],
        log.event[order],
        log.user[order],
    )
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(s)) + 1, [s.size]])
    starts, ends = bounds[:-1], bounds[1:]
    cand = np.flatnonzero((ts[starts] >= t0) & (ts[starts] < t1))
    if max_units is not None and cand.size > max_units:
        cand = np.sort(rng.choice(cand, max_units, replace=False))
    idx = _UserIndex(log)
    out = _Builder(log.n_items)
    allowed = None if target_events is None else np.isin(ev, [int(e) for e in target_events])
    for c in cand:
        a, b = int(starts[c]), int(ends[c])
        sit = it[a:b]
        pos = np.flatnonzero(sit[1:] != sit[:-1]) + 1  # change points: a new item appears
        pos = pos[ts[a + pos] < t1]
        if allowed is not None:
            pos = pos[allowed[a + pos]]
        if pos.size == 0:
            continue
        user = int(us[a])
        if user >= 0:
            sl = idx.before(user, int(ts[a]))
            hi, ht, he, hs = idx.it[sl], idx.ts[sl], idx.ev[sl], idx.sess[sl]
        else:
            hi = ht = he = hs = np.zeros(0, np.int64)
        unit = user if user >= 0 else -(int(s[a]) + 1)
        for j in _select(pos.size, target, rng):
            p = int(pos[j])
            # events at the target's own second are not "before" it
            k = a + p
            while k > a and ts[k - 1] >= ts[a + p]:
                k -= 1
            out.add(
                np.concatenate([hi, it[a:k]]),
                np.concatenate([ht, ts[a:k]]),
                np.concatenate([he, ev[a:k]]),
                np.concatenate([hs, s[a:k]]),
                ts[a + p],
                user,
                hi.size,
                unit,
                it[a + p : a + p + 1],
                session_len=float(k - a),
            )
    return out.build(f"next_item/{target}", t0, t1, exclude_seen)


def drop_unreachable(ev: EvalSet) -> EvalSet:
    """With seen-item exclusion, targets already in history can never be hit: remove them, then
    drop queries left without targets. Documented in protocol.md; repeat metrics are then N/A."""
    assert ev.exclude is not None
    T = (ev.targets - ev.targets.multiply(ev.exclude)).tocsr()
    T.eliminate_zeros()
    keep = np.flatnonzero(np.diff(T.indptr) > 0)
    ev2 = EvalSet(
        ev.queries, T, ev.exclude, ev.history, ev.unit, ev.segments, ev.t0, ev.t1, ev.task
    )
    return ev2.subset(keep)


def available_from(log: Log) -> I64:
    """Catalog entry time per item: its first event in the store's log. A store knows its own
    catalog, so this is legitimate information at query time; items never seen are unavailable."""
    first = np.full(log.n_items, np.iinfo(np.int64).max, np.int64)
    np.minimum.at(first, log.item, log.ts)
    return first


def cold_items(train: Log) -> NDArray[np.bool_]:
    """Boolean mask of items without any training interaction."""
    return cast(NDArray[np.bool_], np.bincount(train.item, minlength=train.n_items) == 0)
