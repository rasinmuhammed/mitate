"""Leakage guards.

The central check is a perturbation test: rebuild the entire pipeline (statistics, embeddings,
fitting, candidate generation, ranking) on a log whose events at or after the cutoff are replaced
by random garbage, and require bitwise-identical predictions for queries issued before the
cutoff's information is legitimately available. Any path by which future data reaches a model
(a popularity table, an item-age feature, a fitted scaler, a learned embedding, an LTR label)
changes the predictions and fails the test, without having to enumerate the paths in advance.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mitate.data import Log


def perturb_after(log: Log, t: int, seed: int = 0) -> Log:
    """Same events before ``t``; at/after ``t`` a different number of random events over the same
    id spaces, with shuffled timestamps inside the future range."""
    rng = np.random.default_rng(seed)
    past = log.until(t)
    fut = log.ts >= t
    n_fut = int(fut.sum())
    m = max(1, int(n_fut * rng.uniform(0.5, 1.5)))
    t_hi = int(log.ts.max()) + 1 if len(log) else t + 1
    user = rng.integers(-1, max(log.n_users, 1), m)
    item = rng.integers(0, log.n_items, m)
    ts = rng.integers(t, max(t_hi, t + 1), m)
    ev = rng.integers(0, 3, m)
    sess = rng.integers(0, max(int(log.session.max()) + 1 if len(log) else 1, 1), m)
    return Log.build(
        np.concatenate([past.user, user]),
        np.concatenate([past.item, item]),
        np.concatenate([past.ts, ts]),
        np.concatenate([past.event, ev]),
        np.concatenate([past.session, sess]),
        log.n_users,
        log.n_items,
    )


def assert_future_invariant(
    pipeline: Callable[[Log], NDArray[Any]], log: Log, t: int, seeds: tuple[int, ...] = (1, 2)
) -> None:
    """``pipeline(log)`` must return identical arrays when everything at/after ``t`` changes."""
    ref = pipeline(log)
    for s in seeds:
        got = pipeline(perturb_after(log, t, seed=s))
        if ref.shape != got.shape or not np.array_equal(ref, got, equal_nan=True):
            diff = np.argwhere(ref != got) if ref.shape == got.shape else "shape"
            raise AssertionError(
                f"future data changed predictions (perturbation {s}): {diff!r}"[:500]
            )
