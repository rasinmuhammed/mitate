"""The guard: per store, whether the shipped ranker or one of its sources should rank the lists.

A ranker learned on one kind of shop can rank another kind badly. The ranker that ships was
trained on OTTO's session-heavy general e-commerce, and on H&M fashion stores of up to 30K events
it scores below plain recent popularity (docs/results.md). So before a store is served, the guard
replays the store's most recent week. The candidates are the blend under the shipped ranker, the
blend under a ranker trained on the store's own weeks before it (``mitate.own_ranker``), and each
source alone; all are fitted on the log before the replayed week and rank its shoppers. The one
with the highest mean NDCG@10 per shopper serves; a tie goes to the one listed first.

The replay is the benchmark's own validation protocol (``mitate.tasks``: each shopper's last
target, scored on the history strictly before it), so what the guard sees on a store is what the
benchmark measures, and the guard itself is scored by the benchmark (ledger D30).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mitate import own_ranker
from mitate.blend import DAY, Blend
from mitate.data import Catalog, Event, Log
from mitate.ltr import TreeEnsemble
from mitate.metrics import ranking_metrics
from mitate.models.base import Recommender
from mitate.stats import per_user_mean
from mitate.tasks import EmptyWindowError, EvalSet, available_from, next_basket, next_item

BLEND = "blend"  # the blend under the shipped ranker
OWN = "own"  # the blend under the store's own ranker
K = 10  # the headline metric's cutoff


@dataclass(frozen=True)
class GuardReport:
    choice: str  # BLEND, OWN, or the name of the source that serves alone
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
    blend: Blend,
    ev: EvalSet,
    avail: NDArray[np.int64],
    rankers: Mapping[str, Any] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Per-shopper NDCG@10 on ``ev`` of the fitted blend under each of ``rankers`` (by default
    its own, as BLEND), then of each of its sources alone."""
    own = blend.booster
    out: dict[str, NDArray[np.float64]] = {}

    def score(name: str, model: Recommender) -> None:
        recs = model.recommend(ev.queries, K, exclude=ev.exclude, available_from=avail)
        _, out[name] = per_user_mean(ranking_metrics(recs, ev.targets, (K,))[K].ndcg, ev.unit)

    try:
        for name, booster in (rankers or {BLEND: own}).items():
            blend.booster = booster
            score(name, blend)
    finally:
        blend.booster = own
    for s in blend.sources:
        score(s.name, s)
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
    own_days: int = 14,
    own_shoppers: int = 2000,
    min_events: int = 0,
    seed: int = 0,
) -> tuple[GuardReport, TreeEnsemble | None] | None:
    """Replay the last ``days`` of ``log`` with ``blend`` (unfitted, under the shipped ranker),
    after training the store's own ranker on the ``own_days`` before them (0: none). Returns the
    report and the own ranker, if one could be trained; None when there is nothing to replay or
    too little before it (fewer than ``min_events``)."""
    ev = holdout(log, days, max_shoppers, seed)
    if ev is None:
        return None
    train = log.until(ev.t0)
    if len(train) < max(min_events, 1):
        return None
    rankers: dict[str, Any] = {BLEND: blend.booster}
    own = None
    if own_days > 0:
        t0 = ev.t0 - own_days * DAY
        own = own_ranker.train(
            blend, log, catalog, t0, ev.t0, holdout_task(log), own_shoppers, seed
        )
        if own is not None:
            rankers[OWN] = own
    blend.fit(train, catalog)
    scores = shopper_ndcg(blend, ev, available_from(log), rankers)
    report = GuardReport(
        choice=choose(scores),
        ndcg={m: float(np.mean(v)) for m, v in scores.items()},
        n_shoppers=int(np.unique(ev.unit).size),
        task=ev.task,
    )
    return report, own
