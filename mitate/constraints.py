"""Business constraints: hard filters and list-level reranking applied after scoring.

Hard filters remove items outright (blocked, out of stock, not deliverable to the shopper's
region, implausible price for the placement, already in the cart or on screen). The list is then
filled greedily in score order under a per-category cap (diversity), and merchant pins are placed
at the top. Pins obey the hard filters too: a pinned item that is out of stock is not shown.

Everything is vectorised over the catalog; per-region masks are cached because shoppers share a
handful of regions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class Placement(StrEnum):
    SIMILAR = "similar"  # product page: items like this one
    COMPLEMENTS = "complements"  # product page and cart: goes well with
    NEXT = "next"  # session or home: next item for the current shopper
    RETURNING = "returning"  # returning customer: repeat purchases and new interests


@dataclass(frozen=True)
class ItemState:
    """Operational catalog state that changes between retrains (stock, delivery coverage)."""

    stock: NDArray[np.float64]  # units in stock; NaN = not tracked
    category: NDArray[np.int32]  # -1 = none
    price: NDArray[np.float64]  # NaN = unknown
    regions: Sequence[frozenset[str] | None]  # deliverable region codes / PIN prefixes; None = all

    @property
    def n_items(self) -> int:
        return int(self.stock.size)

    @staticmethod
    def unconstrained(n: int) -> ItemState:
        return ItemState(
            stock=np.full(n, np.nan),
            category=np.full(n, -1, np.int32),
            price=np.full(n, np.nan),
            regions=[None] * n,
        )


@dataclass(frozen=True)
class Rules:
    """Merchant rules. Item and category ids are internal indices (serve/ maps external ids)."""

    blocked_items: frozenset[int] = frozenset()
    blocked_categories: frozenset[int] = frozenset()
    pins: Mapping[str, Sequence[int]] = field(default_factory=dict)  # placement -> items
    hide_untracked_stock: bool = False  # NaN stock: show (default) or hide
    max_per_category: int | None = None  # diversity cap within one list; None = off
    # price sanity relative to the anchor item (product page placements)
    complement_max_price_ratio: float | None = 3.0
    similar_price_ratio: float | None = 2.0  # similar items within [p / r, p * r]


@dataclass(frozen=True)
class Context:
    """What is known about this request. ``region`` is a coarse delivery region or PIN prefix
    (never a full address). ``exclude`` holds items in the cart or on screen."""

    region: str | None = None
    anchor: int = -1
    exclude: tuple[int, ...] = ()


HIDDEN_BY = (  # audit order: the first rule an item fails is the one reported
    "blocked",
    "out_of_stock",
    "not_delivered",
    "on_screen_or_in_cart",
    "price_band",
    "category_cap",
)


@dataclass(frozen=True)
class Slate:
    items: list[int]
    pinned: list[bool]


class Constraints:
    def __init__(self, state: ItemState, rules: Rules) -> None:
        self.state, self.rules = state, rules
        n = state.n_items
        unblocked = np.ones(n, bool)
        if rules.blocked_items:
            unblocked[np.fromiter(rules.blocked_items, np.int64)] = False
        if rules.blocked_categories:
            unblocked &= ~np.isin(state.category, np.fromiter(rules.blocked_categories, np.int64))
        stock = state.stock
        in_stock = np.where(np.isnan(stock), not rules.hide_untracked_stock, stock > 0)
        self._unblocked, self._in_stock = unblocked, in_stock
        self._base = unblocked & in_stock
        # items sharing a region rule share one mask row
        keys: dict[frozenset[str] | None, int] = {}
        self._region_group = np.array([keys.setdefault(r, len(keys)) for r in state.regions])
        self._region_sets = list(keys)
        self._region_cache: dict[str | None, NDArray[np.bool_]] = {}

    def _region_mask(self, region: str | None) -> NDArray[np.bool_]:
        hit = self._region_cache.get(region)
        if hit is not None:
            return hit

        def ships(prefixes: frozenset[str] | None) -> bool:
            if prefixes is None:
                return True
            return region is not None and any(region.startswith(p) for p in prefixes)

        ok: NDArray[np.bool_] = np.array([ships(s) for s in self._region_sets], bool)[
            self._region_group
        ]
        if len(self._region_cache) >= 256:  # shoppers share few regions; bound it anyway
            self._region_cache.clear()
        self._region_cache[region] = ok
        return ok

    def _off_screen(self, ctx: Context) -> NDArray[np.bool_]:
        m = np.ones(self.state.n_items, bool)
        if ctx.exclude:
            m[np.asarray(ctx.exclude, np.int64)] = False
        if ctx.anchor >= 0:
            m[ctx.anchor] = False
        return m

    def _price_ok(self, placement: Placement, anchor: int) -> NDArray[np.bool_] | None:
        """Price sanity relative to the product on screen; None when no rule applies."""
        if anchor < 0:
            return None
        p = self.state.price
        pa = float(p[anchor])
        if not (np.isfinite(pa) and pa > 0):
            return None
        known: NDArray[np.bool_] = np.isfinite(p)
        if placement is Placement.COMPLEMENTS and self.rules.complement_max_price_ratio:
            return ~known | (p <= pa * self.rules.complement_max_price_ratio)
        if placement is Placement.SIMILAR and self.rules.similar_price_ratio:
            r = self.rules.similar_price_ratio
            return ~known | ((p >= pa / r) & (p <= pa * r))
        return None

    def allowed(self, placement: Placement, ctx: Context) -> NDArray[np.bool_]:
        m: NDArray[np.bool_] = self._base & self._region_mask(ctx.region)
        if ctx.exclude or ctx.anchor >= 0:
            m = m & self._off_screen(ctx)
        price = self._price_ok(placement, ctx.anchor)
        return m if price is None else m & price

    def audit(
        self, scores: NDArray[np.floating], slate: Slate, placement: Placement, ctx: Context
    ) -> dict[str, int]:
        """Items that scored above the lowest recommendation shown (or, for an empty list, any
        scored item) but were held back, counted by the first rule each failed (``HIDDEN_BY``)."""
        shown = [i for i, pin in zip(slate.items, slate.pinned, strict=True) if not pin]
        floor = float(np.min(scores[shown])) if shown else -np.inf
        above = np.flatnonzero(np.isfinite(scores) & (scores > floor))
        above = np.setdiff1d(above, np.asarray([*slate.items, ctx.anchor], np.int64))
        if above.size == 0:
            return {}
        price = self._price_ok(placement, ctx.anchor)
        checks = (
            self._unblocked,
            self._in_stock,
            self._region_mask(ctx.region),
            self._off_screen(ctx),
            np.ones(self.state.n_items, bool) if price is None else price,
        )
        out: dict[str, int] = {}
        for i in above.tolist():
            failed = next((k for k, ok in enumerate(checks) if not ok[i]), len(checks))
            key = HIDDEN_BY[failed]  # passing every hard filter means the diversity cap held it
            out[key] = out.get(key, 0) + 1
        return out

    def slate(
        self, scores: NDArray[np.floating], k: int, placement: Placement, ctx: Context
    ) -> Slate:
        """Top-k under the constraints. ``scores`` covers the whole catalog; -inf = not eligible."""
        ok = self.allowed(placement, ctx) & np.isfinite(scores)
        pins = [int(i) for i in self.rules.pins.get(placement.value, ()) if ok[int(i)]][:k]
        items, pinned = list(pins), [True] * len(pins)
        ok = ok.copy()
        ok[pins] = False
        cap = self.rules.max_per_category
        per_cat: dict[int, int] = {}
        if cap is not None:
            for i in pins:
                c = int(self.state.category[i])
                per_cat[c] = per_cat.get(c, 0) + 1
        cand = np.flatnonzero(ok)
        if cand.size == 0 or len(items) >= k:
            return Slate(items[:k], pinned[:k])
        # stable descending order; ties broken by item index for determinism
        order = cand[np.lexsort((cand, -scores[cand]))]
        if cap is None:
            take = order[: k - len(items)].tolist()
            return Slate(items + take, pinned + [False] * len(take))
        skipped: list[int] = []
        for i in order.tolist():
            c = int(self.state.category[i])
            if c >= 0 and per_cat.get(c, 0) >= cap:
                skipped.append(i)
                continue
            per_cat[c] = per_cat.get(c, 0) + 1
            items.append(i)
            pinned.append(False)
            if len(items) == k:
                break
        # a catalog too narrow for the cap still returns a full list, best skipped items first
        for i in skipped[: k - len(items)]:
            items.append(i)
            pinned.append(False)
        return Slate(items, pinned)
