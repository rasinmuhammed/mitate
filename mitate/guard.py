"""The guard: per store, whether the shipped ranker or one of its sources should rank the lists.

A ranker learned on one kind of shop can rank another kind badly. The ranker that ships was
trained on OTTO's session-heavy general e-commerce, and on H&M fashion stores of up to 30K events
it scores below plain recent popularity (docs/results.md). So before a store is served, the guard
replays the store's most recent week: the blend under the shipped ranker, and each of its sources
alone, are fitted on the log before that week and rank that week's shoppers. The candidate with
the highest mean NDCG@10 per shopper serves; a tie goes to the blend.

The replay is the benchmark's own validation protocol (``mitate.tasks``: each shopper's last
target, scored on the history strictly before it), so what the guard sees on a store is what the
benchmark measures, and the guard itself is scored by the benchmark (ledger D30).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mitate.blend import DAY, Blend
from mitate.data import Catalog, Event, Log
from mitate.metrics import ranking_metrics
from mitate.stats import per_user_mean
from mitate.tasks import EmptyWindowError, EvalSet, available_from, next_basket, next_item

BLEND = "blend"
K = 10  # the headline metric's cutoff


@dataclass(frozen=True)
class GuardReport:
    choice: str  # BLEND, or the name of the source that serves alone
    ndcg: dict[str, float]  # mean NDCG@10 per held-out shopper, per candidate
    n_shoppers: int
    task: str


def holdout_task(log: Log) -> str:
    """Next basket for a log of purchases only (as in the grocery and fashion benchmarks), next
    item for a log that also records browsing."""
    browsing = bool(np.isin(log.event, [int(Event.VIEW), int(Event.CART)]).any())
    return "next_item" if browsing else "next_basket"


def holdout(log: Log, days: int, max_shoppers: int, seed: int = 0) -> EvalSet | None:
    """Each shopper's last target in the final ``days`` of ``log``, or None if there is none."""
    if not len(log):
        return None
    t1 = int(log.ts.max()) + 1
    make = next_item if holdout_task(log) == "next_item" else next_basket
    try:
        ev = make(log, t1 - days * DAY, t1, target="last", seed=seed, max_units=max_shoppers)
    except EmptyWindowError:
        return None
    return ev if len(ev) else None


def shopper_ndcg(
    blend: Blend, ev: EvalSet, avail: NDArray[np.int64]
) -> dict[str, NDArray[np.float64]]:
    """Per-shopper NDCG@10 on ``ev`` of the fitted blend and of each of its sources alone."""
    out: dict[str, NDArray[np.float64]] = {}
    for name, model in [(BLEND, blend), *((s.name, s) for s in blend.sources)]:
        recs = model.recommend(ev.queries, K, exclude=ev.exclude, available_from=avail)
        _, out[name] = per_user_mean(ranking_metrics(recs, ev.targets, (K,))[K].ndcg, ev.unit)
    return out


def choose(scores: Mapping[str, NDArray[np.float64]]) -> str:
    """The candidate with the highest mean; the first listed (the blend) wins ties."""
    means = {m: float(np.mean(v)) if v.size else -np.inf for m, v in scores.items()}
    return max(means, key=lambda m: means[m])


def run(
    blend: Blend,
    log: Log,
    catalog: Catalog,
    days: int = 7,
    max_shoppers: int = 2000,
    min_events: int = 0,
    seed: int = 0,
) -> GuardReport | None:
    """Fit the unfitted ``blend`` on all but the last ``days`` of ``log`` and replay them. None
    when there is nothing to replay or too little before it (fewer than ``min_events``)."""
    ev = holdout(log, days, max_shoppers, seed)
    if ev is None:
        return None
    train = log.until(ev.t0)
    if len(train) < max(min_events, 1):
        return None
    blend.fit(train, catalog)
    scores = shopper_ndcg(blend, ev, available_from(log))
    return GuardReport(
        choice=choose(scores),
        ndcg={m: float(np.mean(v)) for m, v in scores.items()},
        n_shoppers=int(np.unique(ev.unit).size),
        task=ev.task,
    )
