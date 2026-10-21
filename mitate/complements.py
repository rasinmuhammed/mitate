"""Complementary items ("goes well with").

Two sources, used in this order of trust:

1. **Co-purchase evidence** from the store's own baskets: items bought or carted by the same
   shoppers as the anchor, scored by lift over the item's base rate, with a minimum number of
   distinct shoppers (`Evidence.bought`).
2. **Category complement graph**: which categories go with which, proposed once by an LLM from
   the store's taxonomy and *reviewed by the merchant*. Only approved edges are ever served.
   Within an approved target category, items are ranked by popularity and price sanity. This is
   what serves a store with no orders yet.

The LLM runs offline (at catalog import or on request from the dashboard), never in the request
path. Proposals are constrained to category names that exist in the taxonomy; anything else the
model says is discarded.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mitate.explain import Evidence


class EdgeStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class Edge:
    src: int  # anchor category
    dst: int  # complementary category
    status: EdgeStatus = EdgeStatus.PROPOSED
    origin: str = "llm"  # llm | merchant | data
    note: str = ""


@dataclass
class CategoryGraph:
    edges: list[Edge] = field(default_factory=list)

    def approved(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for e in self.edges:
            if e.status is EdgeStatus.APPROVED:
                out.setdefault(e.src, []).append(e.dst)
        return out

    def review(self, src: int, dst: int, status: EdgeStatus) -> None:
        for e in self.edges:
            if (e.src, e.dst) == (src, dst):
                e.status = status
                return
        self.edges.append(Edge(src, dst, status, origin="merchant"))

    def merge_proposals(self, proposals: Sequence[Edge]) -> int:
        """Add new proposals; reviewed edges are never overwritten. Returns how many were new."""
        seen = {(e.src, e.dst) for e in self.edges}
        new = [e for e in proposals if (e.src, e.dst) not in seen and e.src != e.dst]
        self.edges.extend(new)
        return len(new)

    def to_json(self) -> str:
        return json.dumps([asdict(e) for e in self.edges])

    @staticmethod
    def from_json(s: str) -> CategoryGraph:
        return CategoryGraph(
            [Edge(**{**d, "status": EdgeStatus(d["status"])}) for d in json.loads(s)]
        )

    def save(self, path: Path) -> None:
        path.write_text(self.to_json())

    @staticmethod
    def load(path: Path) -> CategoryGraph:
        return CategoryGraph.from_json(path.read_text()) if path.exists() else CategoryGraph()


# ---------------------------------------------------------------------------- LLM proposals
PROMPT = """You help a small online shop. Below is the shop's list of product categories.
For each category, list the other categories from the SAME list whose products a shopper buying
from the first category would plausibly also buy in the same order (complements, e.g. phone ->
phone case), not substitutes (items that replace each other). Use only names from the list,
copied exactly. Return JSON only: {{"pairs": [{{"category": "...", "goes_with": ["...", "..."]}}]}}.
At most {max_per} entries in each goes_with list. Omit categories without clear complements.

Categories:
{categories}
"""


def category_paths(names: Sequence[str], parent: NDArray[np.int32] | None) -> list[str]:
    """Human-readable 'Parent > Child' names so the LLM sees the taxonomy context."""
    out = []
    for c in range(len(names)):
        parts, p, guard = [names[c]], c, 0
        while parent is not None and parent[p] >= 0 and guard < 10:
            p = int(parent[p])
            parts.append(names[p])
            guard += 1
        out.append(" > ".join(reversed(parts)))
    return out


def parse_proposals(text: str, paths: Sequence[str]) -> list[Edge]:
    """Keep only edges between categories that exist; tolerate code fences and extra prose."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    index = {p.strip().lower(): i for i, p in enumerate(paths)}
    # also accept the leaf name alone when it is unambiguous
    leaves: dict[str, list[int]] = {}
    for i, p in enumerate(paths):
        leaves.setdefault(p.split(" > ")[-1].strip().lower(), []).append(i)

    def resolve(name: Any) -> int | None:
        if not isinstance(name, str):
            return None
        k = name.strip().lower()
        if k in index:
            return index[k]
        hit = leaves.get(k.split(" > ")[-1].strip())
        return hit[0] if hit and len(hit) == 1 else None

    edges = []
    for row in data.get("pairs", []) if isinstance(data, dict) else []:
        if not isinstance(row, dict):
            continue
        s = resolve(row.get("category"))
        goes = row.get("goes_with")
        if s is None or not isinstance(goes, list):
            continue
        for g in goes:
            d = resolve(g)
            if d is not None and d != s:
                edges.append(Edge(s, d, EdgeStatus.PROPOSED, "llm"))
    return edges


def propose_with_llm(
    names: Sequence[str],
    parent: NDArray[np.int32] | None = None,
    model: str = "qwen3:4b",
    host: str = "http://localhost:11434",
    max_per: int = 5,
    chunk: int = 150,
    timeout_s: float = 300.0,
) -> list[Edge]:
    """Ask a local Ollama model for category complements. Offline only; deterministic decoding
    (temperature 0, fixed seed) so a re-run proposes the same edges."""
    import httpx

    paths = category_paths(names, parent)
    edges: list[Edge] = []
    for s in range(0, len(paths), chunk):
        # every chunk sees the full list as targets but only proposes for its own sources
        prompt = PROMPT.format(max_per=max_per, categories="\n".join(paths))
        prompt += "\nOnly answer for these categories:\n" + "\n".join(paths[s : s + chunk])
        r = httpx.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "seed": 0},
            },
            timeout=timeout_s,
        )
        r.raise_for_status()
        edges += parse_proposals(r.json()["response"], paths)
    return edges


# ---------------------------------------------------------------------------- scoring
@dataclass
class Complements:
    """Scores complements for anchor items in three tiers; returns -inf for non-candidates.

    1. **Bought together**: co-bought with an anchor by at least ``min_support`` shoppers, by lift,
       in a category none of the anchors belong to. Same-category co-purchases are mostly
       substitutes (a second pan), which the similar placement serves.
    2. **Approved pairing**: in a category the merchant approved as a complement of the anchor's,
       by recent popularity.
    3. **Related, other category** (with item vectors): the most text-similar items outside the
       anchors' own categories. Measured on Amazon Grocery development stores, this was the
       strongest day-one predictor of cross-category co-purchase (docs/results.md); it fills the
       slots the first two tiers leave empty and never outranks them.
    """

    evidence: Evidence
    graph: CategoryGraph
    item_category: NDArray[np.int32]
    vectors: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        ev = self.evidence
        n = self.item_category.size
        self._base = np.asarray(ev.bought.diagonal(), np.float64)  # buyers per item
        self._n_buyers = max(float(self._base.sum()), 1.0)
        self._pop = np.log1p(ev.shoppers_7d.astype(np.float64))
        self._approved = self.graph.approved()
        self._by_cat: dict[int, NDArray[np.int64]] = {}
        for c in np.unique(self.item_category[self.item_category >= 0]):
            self._by_cat[int(c)] = np.flatnonzero(self.item_category == c)
        assert self._base.size == n

    def score(
        self, anchors: Sequence[int]
    ) -> tuple[NDArray[np.float64], dict[int, tuple[int, int]]]:
        """(scores, graph_edge_per_item). Data-backed items score above graph-only items; the
        returned map records which approved category edge nominated a graph-only item, so the
        explanation can cite it."""
        n = self.item_category.size
        S = np.full(n, -np.inf)
        ms = self.evidence.min_support
        own = self.item_category[list(anchors)] if len(anchors) else self.item_category[:0]
        own = own[own >= 0]  # uncategorised anchors exclude nothing
        for a in anchors:
            row = self.evidence.bought.getrow(a)
            idx, cnt = row.indices, row.data.astype(np.float64)
            keep = (cnt >= ms) & (idx != a) & ~np.isin(self.item_category[idx], own)
            idx, cnt = idx[keep], cnt[keep]
            if idx.size:
                # lift = P(b | a) / P(b); +1000 puts data-backed items above graph-only ones
                lift = (cnt / max(self._base[a], 1.0)) / (self._base[idx] / self._n_buyers)
                S[idx] = np.maximum(S[idx], 1000.0 + np.log(lift) + np.log(cnt))
        edge_of: dict[int, tuple[int, int]] = {}
        for a in anchors:
            ca = int(self.item_category[a])
            for dst in self._approved.get(ca, []):
                items = self._by_cat.get(dst)
                if items is None:
                    continue
                g = self._pop[items]
                fresh = ~np.isfinite(S[items])
                S[items[fresh]] = g[fresh]
                for i in items[fresh].tolist():
                    edge_of[i] = (ca, dst)
        if self.vectors is not None and len(anchors):
            sim = (self.vectors[list(anchors)] @ self.vectors.T).max(axis=0).astype(np.float64)
            other = ~np.isin(self.item_category, own)
            fill = other & ~np.isfinite(S)
            S[fill] = -10.0 + sim[fill]  # cosine in [-1, 1]: always below the first two tiers
        S[list(anchors)] = -np.inf
        return S, edge_of
