"""The product engine: one store's models behind the four placements.

The method shifts with the store's data volume:

* **Catalog only** (fewer than ``min_behaviour_events`` events): similar items from text
  embeddings; complements from the merchant-approved category graph, then text-related items
  from other categories; and a catalog prior (new arrivals, then content similarity to whatever
  the shopper has viewed this visit). Complements move to measured co-purchase as soon as enough
  shoppers stand behind a pair, whatever the stage.
* **Behavioural**: the Mitate blend over every source whose volume threshold the store has
  reached, ordered by the shipped ranker trained on pooled OTTO stores of every size with
  data-volume features, so it already knows which source to trust at this store's size. When a
  source is below its threshold the ranker trained without that source is used; with no ranker
  for the active sources, the blend falls back to rank fusion.

Thresholds come from the benchmark (docs/results.md), not from intuition; until a threshold is
measured the source is always on. Every list passes the business constraints and every item
carries a factual reason or none.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from mitate.blend import Blend
from mitate.complements import CategoryGraph, Complements
from mitate.constraints import Constraints, Context, ItemState, Placement, Rules, Slate
from mitate.content import embed, item_texts
from mitate.data import Catalog, Event, Log, Queries
from mitate.explain import Evidence, Reason
from mitate.ltr import TreeEnsemble
from mitate.models.base import Recommender
from mitate.models.registry import MODELS

# LambdaMART rankers trained on pooled OTTO benchmark stores (CC BY 4.0; research ledger D25),
# written with their provenance by scripts/ship_rankers.py
SHIPPED_RANKERS = Path(__file__).parent / "rankers"


def ranker_sources(name: str, feature_names: Sequence[str]) -> set[str] | None:
    """The sources a ranker serves, from the benchmark's naming: ``full`` was trained on every
    source in its features, ``no_<source>`` on all but that one (whose features it saw as
    missing). Any other ranker is a research ablation and never serves (None)."""
    trained = {f[2:-5] for f in feature_names if f.startswith("s_") and f.endswith("_rank")}
    if name == "full":
        return trained
    if name.startswith("no_") and name[3:] in trained:
        return trained - {name[3:]}
    return None


def shipped_rankers() -> Path | None:
    return SHIPPED_RANKERS if (SHIPPED_RANKERS / "full.json").exists() else None


# the benchmark's blend sources (experiments/configs/benchmark.yaml)
DEFAULT_SOURCES = (
    *("recpop", "itemknn", "rp3beta", "ease", "ials"),
    *("pctm", "seqrules", "vstan", "sasrec_ce"),
)


@dataclass(frozen=True)
class EngineConfig:
    min_behaviour_events: int = 200
    sources: tuple[str, ...] = DEFAULT_SOURCES
    source_min_events: Mapping[str, int] = field(default_factory=dict)
    source_params: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    booster_dir: Path | None = field(default_factory=shipped_rankers)  # <variant>.json rankers
    n_cand: int = 30
    embedder: str = "e5-small"  # see mitate.content.EMBEDDERS; "none" = no text model
    min_support: int = 5
    returning_repeat_share: float = 0.5  # share of a returning-customer slate for repurchases
    seed: int = 0


@dataclass(frozen=True)
class Shopper:
    """One request's view of the shopper: their own history (oldest first), possibly empty."""

    items: Sequence[int] = ()
    ts: Sequence[int] = ()
    events: Sequence[int] = ()
    session_start: int = 0  # index where the current visit begins
    known: bool = False
    now: int = 0

    def queries(self, anchor: int = -1) -> Queries:
        """One query. The product on screen is the shopper's latest view: appended to the history
        unless it already ends it, so history-based sources (all but content) see it."""
        items, ts, ev = list(self.items), list(self.ts), list(self.events)
        if anchor >= 0 and (not items or items[-1] != anchor):
            items.append(anchor)
            ts.append(max([self.now - 1, *ts[-1:]]))
            ev.append(int(Event.VIEW))
        n = len(items)
        return Queries(
            indptr=np.array([0, n], np.int64),
            items=np.asarray(items, np.int32),
            ts=np.asarray(ts, np.int64),
            event=np.asarray(ev, np.int8),
            now=np.array([self.now], np.int64),
            user=np.array([0 if self.known else -1], np.int32),
            session_start=np.array([self.session_start], np.int64),
            anchor=np.array([anchor], np.int32),
        )

    def orders(self) -> list[int]:
        return [i for i, e in zip(self.items, self.events, strict=True) if e == Event.ORDER]


@dataclass(frozen=True)
class Recommendation:
    item: int
    score: float
    pinned: bool
    reason: Reason | None


@dataclass(frozen=True)
class Ranked:
    items: list[Recommendation]
    hidden: dict[str, int]  # better-scored items the rules held back, by rule (when audited)


@dataclass(frozen=True)
class EngineInfo:
    stage: str  # catalog_only | behavioural
    active_sources: tuple[str, ...]
    ranker: str  # booster variant name, "rrf", or "none"
    n_events: int
    n_items: int
    fit_seconds: float
    trained_at: int  # data horizon (latest event time + 1)


class Engine:
    def __init__(self, cfg: EngineConfig | None = None) -> None:
        self.cfg = cfg or EngineConfig()

    # ------------------------------------------------------------------ fitting
    def fit(
        self,
        log: Log,
        catalog: Catalog,
        state: ItemState | None = None,
        rules: Rules | None = None,
        graph: CategoryGraph | None = None,
        vectors: NDArray[np.float32] | None = None,
    ) -> EngineInfo:
        t0 = time.monotonic()
        cfg = self.cfg
        n = catalog.n_items
        self.catalog = catalog
        self.state = state or ItemState.unconstrained(n)
        self.constraints = Constraints(self.state, rules or Rules())
        self.graph = graph or CategoryGraph()
        has_text = catalog.title is not None or catalog.text is not None
        if vectors is None and has_text and cfg.embedder != "none":
            vectors = embed(item_texts(catalog), cfg.embedder)
        self.vectors = vectors
        self.evidence = Evidence(min_support=cfg.min_support).fit(log, catalog)
        self.complements = Complements(
            self.evidence,
            self.graph,
            self.state.category if catalog.category is None else np.asarray(catalog.category),
            vectors,
        )
        self.first_seen = self.evidence.first_seen
        self.blend: Blend | None = None
        active: tuple[str, ...] = ()
        ranker = "none"
        if len(log) >= cfg.min_behaviour_events:
            active = tuple(s for s in cfg.sources if len(log) >= cfg.source_min_events.get(s, 0))
            sources: list[Recommender] = [
                MODELS[s](**dict(cfg.source_params.get(s, {})), seed=cfg.seed) for s in active
            ]
            booster, ranker = self._booster(active)
            self.blend = Blend(sources=sources, n_cand=cfg.n_cand, booster=booster)
            self.blend.fit(log, catalog)
        horizon = int(log.ts.max()) + 1 if len(log) else 0
        self.info = EngineInfo(
            stage="behavioural" if self.blend is not None else "catalog_only",
            active_sources=active,
            ranker=ranker,
            n_events=len(log),
            n_items=n,
            fit_seconds=time.monotonic() - t0,
            trained_at=horizon,
        )
        return self.info

    def _booster(self, active: tuple[str, ...]) -> tuple[TreeEnsemble | None, str]:
        """The ranker trained on exactly the active sources, or rank fusion if none was: a ranker
        given features it never saw would score out of its training distribution."""
        d = self.cfg.booster_dir
        if d is None or not d.is_dir():
            return None, "rrf"
        # only these names can serve this source set, so no other ranker is loaded
        names = ["full", *(f"no_{s}" for s in self.cfg.sources if s not in active)]
        for name in names:
            path = d / f"{name}.json"
            if not path.exists():
                continue
            ens = TreeEnsemble.load(path)
            if ranker_sources(name, ens.feature_names) == set(active):
                return ens, name
        return None, "rrf"

    # ------------------------------------------------------------------ serving
    def recommend(
        self,
        placement: Placement,
        k: int = 10,
        *,
        anchors: Sequence[int] = (),
        shopper: Shopper | None = None,
        ctx: Context | None = None,
    ) -> list[Recommendation]:
        return self.rank(placement, k, anchors=anchors, shopper=shopper, ctx=ctx).items

    def rank(
        self,
        placement: Placement,
        k: int = 10,
        *,
        anchors: Sequence[int] = (),
        shopper: Shopper | None = None,
        ctx: Context | None = None,
        audit: bool = False,
    ) -> Ranked:
        """``recommend`` plus, with ``audit``, which rules held back better-scored items."""
        shopper = shopper or Shopper()
        anchor = anchors[0] if anchors else -1
        ctx = ctx or Context(anchor=anchor)
        edge_of: dict[int, tuple[int, int]] = {}
        if placement is Placement.SIMILAR:
            S = self._similar(anchor)
        elif placement is Placement.COMPLEMENTS:
            S, edge_of = self.complements.score(list(anchors))
        elif placement is Placement.NEXT:
            S = self._next(shopper, anchor)
        else:
            S = self._returning(shopper, k)
        slate = self.constraints.slate(S, k, placement, ctx)
        hidden = self.constraints.audit(S, slate, placement, ctx) if audit else {}
        return Ranked(self._explain(placement, slate, S, anchor, shopper, edge_of), hidden)

    def _similar(self, anchor: int) -> NDArray[np.float64]:
        n = self.catalog.n_items
        if anchor < 0:
            return np.full(n, -np.inf)
        pop = self._pop_prior()
        if self.vectors is not None:
            S: NDArray[np.float64] = (self.vectors @ self.vectors[anchor]).astype(np.float64)
            return S + 1e-3 * pop  # popularity only breaks near-ties
        cat = self.complements.item_category
        same = (cat == cat[anchor]) & (cat >= 0)
        return np.where(same, pop, -np.inf)

    def _pop_prior(self) -> NDArray[np.float64]:
        p = np.log1p(self.evidence.shoppers_7d.astype(np.float64))
        return p / p.max() if p.max() > 0 else p

    def _next(self, shopper: Shopper, anchor: int) -> NDArray[np.float64]:
        if self.blend is not None:
            return cast(
                NDArray[np.float64], self.blend.score(shopper.queries(anchor))[0].astype(np.float64)
            )
        # catalog only: content similarity to this visit's items, else newest items first
        recent = list(shopper.items[shopper.session_start :]) + ([anchor] if anchor >= 0 else [])
        if self.vectors is not None and recent:
            w = 0.7 ** np.arange(len(recent))[::-1]
            S: NDArray[np.float64] = (self.vectors @ (w @ self.vectors[np.asarray(recent)])).astype(
                np.float64
            )
            return S
        fs = self.first_seen.astype(np.float64)
        known = fs < np.iinfo(np.int64).max
        return np.where(known, fs / max(fs[known].max(initial=1.0), 1.0), 0.0)

    def _returning(self, shopper: Shopper, k: int) -> NDArray[np.float64]:
        """Repurchases (by personal order frequency, then recency) fill up to
        ``returning_repeat_share`` of the slate; the rest is the next-item ranking."""
        explore = self._next(shopper, -1)
        orders = shopper.orders()
        if not orders:
            return explore
        cnt: dict[int, int] = {}
        last: dict[int, int] = {}
        for pos, i in enumerate(orders):
            cnt[i] = cnt.get(i, 0) + 1
            last[i] = pos
        n_rep = round(k * self.cfg.returning_repeat_share)
        rep = sorted(cnt, key=lambda i: (-cnt[i], -last[i]))[:n_rep]
        S = explore.copy()
        fin = S[np.isfinite(S)]
        top = fin.max() if fin.size else 0.0
        for r, i in enumerate(rep):
            S[i] = top + 1.0 + (len(rep) - r)
        return S

    def _explain(
        self,
        placement: Placement,
        slate: Slate,
        S: NDArray[np.float64],
        anchor: int,
        shopper: Shopper,
        edge_of: dict[int, tuple[int, int]],
    ) -> list[Recommendation]:
        orders = shopper.orders()
        out = []
        for i, pinned in zip(slate.items, slate.pinned, strict=True):
            reason = self.evidence.reason(
                placement,
                i,
                anchor=anchor,
                history=list(shopper.items),
                orders=orders,
                pinned=pinned,
                approved_pair=edge_of.get(i),
            )
            out.append(Recommendation(i, float(S[i]), pinned, reason))
        return out
