"""A ranker trained on one store's own recent purchases.

The shipped ranker learned from OTTO's stores which sources to trust, and that lesson does not hold
in every kind of shop: on fashion stores it ranks below plain popularity (docs/results.md). Every
store, though, labels its own data with what its shoppers went on to buy. This module fits the
blend's sources on the log before a window, describes their candidates for every target inside
it, and trains LambdaMART on those labels with one fixed configuration: a single store has too few
labels to tune on. Whether the result serves is decided by the guard (``mitate.guard``), on a later
week that this training never sees.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from mitate.blend import Blend, ltr_groups
from mitate.data import Catalog, Log
from mitate.ltr import TreeEnsemble, train_isolated
from mitate.tasks import EmptyWindowError, next_basket, next_item

# shallow, shrunk and bagged on features: a store's own labels number in the hundreds or thousands
PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "lambda_l2": 1.0,
    "num_boost_round": 200,
}
MIN_QUERIES = 50  # labelled queries (a purchase among the candidates) needed to train at all


def train(
    blend: Blend,
    log: Log,
    catalog: Catalog,
    t0: int,
    t1: int,
    task: str,
    max_shoppers: int = 2000,
    seed: int = 0,
) -> TreeEnsemble | None:
    """Fit ``blend``'s sources on the log before ``t0`` and learn to rank their candidates for
    every target in ``[t0, t1)``. None when fewer than ``MIN_QUERIES`` queries can be labelled."""
    make = next_item if task == "next_item" else next_basket
    try:
        ev = make(log, t0, t1, target="successive", seed=seed, max_units=max_shoppers)
    except EmptyWindowError:
        return None
    train_log = log.until(t0)
    if not len(ev) or not len(train_log):
        return None
    blend.fit(train_log, catalog)
    frames = [
        blend.candidates(ev.queries.slice(s, min(len(ev), s + blend.batch)))
        for s in range(0, len(ev), blend.batch)
    ]
    X, y, g = ltr_groups(frames, ev.targets)
    if g.size < MIN_QUERIES:
        return None
    with tempfile.TemporaryDirectory() as d:
        return train_isolated(
            Path(d) / "job.npz",
            Path(d) / "model.json",
            X,
            y,
            g,
            blend.feature_names(),
            seed=seed,
            params=PARAMS,
        )
