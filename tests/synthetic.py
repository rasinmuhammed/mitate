"""A small synthetic marketplace for runner tests: categories, sessions, repeat shoppers."""

from __future__ import annotations

import numpy as np
import polars as pl

from bench.datasets import DAY, Dataset


def marketplace(
    seed: int = 0, n_items: int = 400, n_users: int = 300, n_sessions: int = 6000, days: int = 60
) -> Dataset:
    rng = np.random.default_rng(seed)
    n_cat = 20
    item_cat = rng.integers(0, n_cat, n_items)
    by_cat = [np.flatnonzero(item_cat == c) for c in range(n_cat)]
    rows = []
    for s in range(n_sessions):
        user = int(rng.integers(n_users)) if rng.random() < 0.6 else -1
        t = int(rng.integers(0, days * DAY))
        c = int(rng.integers(n_cat)) if user < 0 else int((user * 7 + rng.integers(3)) % n_cat)
        pool = by_cat[c] if by_cat[c].size else np.arange(n_items)
        cur = int(rng.choice(pool))
        for j in range(int(rng.integers(2, 9))):
            ev = int(rng.choice(3, p=[0.7, 0.2, 0.1]))
            rows.append((user, s, cur, t + 30 * j, ev))
            cur = int(rng.choice(pool)) if rng.random() < 0.5 else int((cur + 1) % n_items)
    ev = pl.DataFrame(
        rows,
        schema={
            "user": pl.Int64,
            "session": pl.Int64,
            "item": pl.Int64,
            "ts": pl.Int64,
            "event": pl.UInt8,
        },
        orient="row",
    )
    items = pl.DataFrame(
        {
            "item": np.arange(n_items),
            "category": item_cat.astype(np.int64),
            "title": [f"item {i}" for i in range(n_items)],
            "price": rng.lognormal(3, 0.5, n_items),
        }
    )
    cats = pl.DataFrame(
        {
            "category": np.arange(n_cat),
            "parent": np.full(n_cat, -1),
            "name": [f"cat {c}" for c in range(n_cat)],
        }
    )
    return Dataset(
        "synth",
        ev.lazy(),
        items,
        cats,
        (days - 14) * DAY,
        (days - 7) * DAY,
        days * DAY,
        "cobrowse",
        "next_item",
        "CC BY 4.0",
    )
