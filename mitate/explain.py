"""Factual explanations.

A reason is a sentence built from a counted fact about the (shopper, item) pair, never from the
ranker's internals, and never invented. Candidates for a recommended item, in placement-specific
priority order:

* ``bought_with``   N shoppers who bought the anchor (or a history item) also bought this item.
* ``viewed_next``   N shoppers viewed this item right after the anchor (or last-viewed item).
* ``similar``       Concrete shared attributes with the anchor: category, brand, price band.
* ``pairs_with``    The merchant approved this category pairing (complement graph).
* ``bought_before`` The shopper bought this item before (their own data, shown only to them).
* ``popular``       In the top 10% of its category by distinct shoppers over the last 7 days.
* ``new``           Launched within the last 14 days.
* ``featured``      Pinned by the merchant.

Counts of other shoppers are shown only when at least ``min_support`` distinct shoppers stand
behind them, so a reason can never reveal one person's behaviour. If no fact holds, the item has
no reason and the widget shows none.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from mitate.constraints import Placement
from mitate.data import Catalog, Event, Log

DAY = 86_400
MAX_ITEMS_PER_UNIT = 200  # bounds the quadratic pair count for very active shoppers or bots

PRIORITY: dict[Placement, tuple[str, ...]] = {
    Placement.SIMILAR: ("featured", "similar", "viewed_next", "popular", "new"),
    Placement.COMPLEMENTS: ("featured", "bought_with", "pairs_with", "viewed_next", "popular"),
    Placement.NEXT: ("featured", "viewed_next", "bought_with", "similar", "popular", "new"),
    Placement.RETURNING: ("featured", "bought_before", "bought_with", "popular", "new"),
}


@dataclass(frozen=True)
class Reason:
    code: str
    text: str
    evidence: dict[str, Any] = field(default_factory=dict)


def _units(log: Log) -> NDArray[np.int64]:
    """Shopper unit per event: known user, else the anonymous session (offset to stay disjoint)."""
    u = log.user.astype(np.int64)
    off = int(u.max()) + 1 if (u >= 0).any() else 0
    return np.where(u >= 0, u, off + log.session.astype(np.int64))


def _co_units(unit: NDArray[np.int64], item: NDArray[np.int64], n: int) -> sp.csr_matrix:
    """Item x item count of distinct units that have both items (diagonal = units per item)."""
    if unit.size == 0:
        return sp.csr_matrix((n, n), dtype=np.int32)
    key = np.unique(unit * n + item)
    u, i = key // n, key % n
    # cap per unit (deterministic: lowest item ids); binds only for extreme units such as bots
    _, start, cnt = np.unique(u, return_index=True, return_counts=True)
    keep = np.ones(u.size, bool)
    for s, c in zip(start[cnt > MAX_ITEMS_PER_UNIT], cnt[cnt > MAX_ITEMS_PER_UNIT], strict=True):
        keep[s + MAX_ITEMS_PER_UNIT : s + c] = False
    u, i = u[keep], i[keep]
    _, row = np.unique(u, return_inverse=True)
    B = sp.csr_matrix((np.ones(u.size, np.int32), (row, i)), shape=(int(row.max()) + 1, n))
    return (B.T @ B).tocsr()


def _transitions(log: Log, unit: NDArray[np.int64], n: int) -> sp.csr_matrix:
    """a -> b count of distinct units with b viewed right after a (distinct items) in a session."""
    order = np.lexsort((log.ts, log.session))
    s, it, un = log.session[order], log.item[order].astype(np.int64), unit[order]
    same = (s[1:] == s[:-1]) & (it[1:] != it[:-1])
    a, b, uu = it[:-1][same], it[1:][same], un[:-1][same]
    if a.size == 0:
        return sp.csr_matrix((n, n), dtype=np.int32)
    trip = np.unique(np.stack([uu, a, b], axis=1), axis=0)
    M = sp.csr_matrix((np.ones(len(trip), np.int32), (trip[:, 1], trip[:, 2])), shape=(n, n))
    M.sum_duplicates()
    return M


@dataclass
class Evidence:
    """Counted facts from one store's log, refreshed at every retrain."""

    min_support: int = 5
    popular_quantile: float = 0.9
    new_days: float = 14.0

    def fit(self, log: Log, catalog: Catalog) -> Evidence:
        n = log.n_items
        unit = _units(log)
        buy = (log.event == Event.ORDER) | (log.event == Event.CART)
        self.bought = _co_units(unit[buy], log.item[buy].astype(np.int64), n)
        self.viewed_next = _transitions(log, unit, n)
        now = int(log.ts.max()) + 1 if len(log) else 0
        recent = log.ts >= now - 7 * DAY
        key = np.unique(unit[recent] * n + log.item[recent])
        self.shoppers_7d = np.bincount(key % n, minlength=n)
        cat = catalog.category if catalog.category is not None else np.full(n, -1, np.int32)
        self.category = np.asarray(cat)
        self.popular = np.zeros(n, bool)
        for c in np.unique(self.category):
            m = self.category == c
            v = self.shoppers_7d[m]
            if v.max(initial=0) < self.min_support:
                continue
            cut = max(np.quantile(v, self.popular_quantile), self.min_support)
            self.popular[np.flatnonzero(m)[v >= cut]] = True
        first = np.full(n, np.iinfo(np.int64).max)
        np.minimum.at(first, log.item, log.ts)
        if catalog.first_seen is not None:
            first = np.minimum(first, np.asarray(catalog.first_seen, np.int64))
        self.first_seen = first
        self.now = now
        self.catalog = catalog
        self.band = catalog.price_band()
        return self

    # ------------------------------------------------------------------ facts
    def _title(self, i: int) -> str:
        t = self.catalog.title
        return str(t[i]) if t is not None and t[i] else "this item"

    def _cat_name(self, c: int) -> str | None:
        names = self.catalog.category_names
        return str(names[c]) if names is not None and 0 <= c < len(names) else None

    def _best_pair(
        self, M: sp.csr_matrix, sources: Sequence[int], item: int
    ) -> tuple[int, int] | None:
        best: tuple[int, int] | None = None
        for a in sources:
            if a < 0 or a == item:  # the diagonal is the item's own count, not a pairing
                continue
            v = int(M[a, item])
            if v >= self.min_support and (best is None or v > best[1]):
                best = (a, v)
        return best

    def reason(
        self,
        placement: Placement,
        item: int,
        *,
        anchor: int = -1,
        history: Sequence[int] = (),
        orders: Sequence[int] = (),
        pinned: bool = False,
        approved_pair: tuple[int, int] | None = None,
    ) -> Reason | None:
        """The highest-priority fact that holds for ``item``. ``history`` is the shopper's own
        item history (oldest first), ``orders`` their purchased items, ``approved_pair`` the
        (anchor category, item category) edge when the complement graph nominated the item."""
        context = [anchor] if anchor >= 0 else list(dict.fromkeys(reversed(history)))[:10]
        last = anchor if anchor >= 0 else (history[-1] if len(history) else -1)
        for code in PRIORITY[placement]:
            r = self._fact(code, item, anchor, last, context, orders, pinned, approved_pair)
            if r is not None:
                return r
        return None

    def _fact(
        self,
        code: str,
        item: int,
        anchor: int,
        last: int,
        context: Sequence[int],
        orders: Sequence[int],
        pinned: bool,
        approved_pair: tuple[int, int] | None,
    ) -> Reason | None:
        a: int | str | None  # an item id in the pair branches, a category name in pairs_with
        if code == "featured":
            return Reason("featured", "Featured by the store") if pinned else None
        if code == "bought_with":
            hit = self._best_pair(self.bought, context, item)
            if hit is None:
                return None
            a, v = hit
            return Reason(
                "bought_with",
                f"{v} shoppers who bought {self._title(a)} also bought this",
                {"with": a, "shoppers": v},
            )
        if code == "viewed_next":
            hit = self._best_pair(self.viewed_next, [last], item) if last >= 0 else None
            if hit is None:
                return None
            a, v = hit
            return Reason(
                "viewed_next",
                f"{v} shoppers viewed this right after {self._title(a)}",
                {"after": a, "shoppers": v},
            )
        if code == "similar" and anchor >= 0:
            shared = []
            ca, ci = int(self.category[anchor]), int(self.category[item])
            if ca >= 0 and ca == ci and (name := self._cat_name(ci)):
                shared.append(f"same category ({name})")
            br = self.catalog.brand
            if br is not None and br[anchor] >= 0 and br[anchor] == br[item]:
                shared.append("same brand")
            if not shared:  # a price band alone does not make two products similar
                return None
            if self.band[anchor] >= 0 and self.band[anchor] == self.band[item]:
                shared.append("similar price")
            text = f"Like {self._title(anchor)}: {', '.join(shared)}"
            return Reason("similar", text, {"anchor": anchor, "shared": shared})
        if code == "pairs_with" and approved_pair is not None:
            a, b = (self._cat_name(c) for c in approved_pair)
            if a is None or b is None:
                return None
            return Reason(
                "pairs_with", f"Pairs with {a} (store pairing)", {"categories": list(approved_pair)}
            )
        if code == "bought_before":
            n_b = sum(1 for o in orders if o == item)
            if n_b == 0:
                return None
            times = "once" if n_b == 1 else f"{n_b} times"
            return Reason("bought_before", f"You bought this {times} before", {"times": n_b})
        if code == "popular" and self.popular[item]:
            name = self._cat_name(int(self.category[item]))
            where = f"in {name}" if name else "in this store"
            return Reason(
                "popular",
                f"Popular {where} this week",
                {"shoppers_7d": int(self.shoppers_7d[item])},
            )
        if code == "new":
            age = (self.now - self.first_seen[item]) / DAY
            if 0 <= age <= self.new_days:
                return Reason("new", "New in the store", {"age_days": round(float(age), 1)})
        return None
