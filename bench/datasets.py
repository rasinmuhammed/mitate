"""Canonical benchmark datasets.

Every dataset is exposed as a ``Dataset``: an event table (polars LazyFrame with columns
``user, session, item, ts, event``), an item table, the global split points and the sub-catalog
strategy the store simulator should use. Raw ids are kept; dense per-store ids are assigned by the
simulator.

Split points are global per dataset (Time to Split): ``t_val < t_test < t_end``. Tuning fits on
``[.., t_val)`` and validates on ``[t_val, t_test)``; the final model fits on ``[.., t_test)``
and is tested on ``[t_test, t_end)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

DATA = Path("data")
DAY = 86_400


@dataclass(frozen=True)
class Dataset:
    name: str
    # user (Int64, -1 = anonymous), session (Int64), item (Int64), ts (Int64 seconds), event (UInt8)
    events: pl.LazyFrame
    items: pl.DataFrame  # item (Int64) + optional title, text, category (Int64), brand, price
    categories: pl.DataFrame | None  # category (Int64), parent (Int64, -1 root), name (Utf8)
    t_val: int
    t_test: int
    t_end: int
    subcatalog: Literal["taxonomy", "cobrowse"]
    task: Literal["next_item", "next_basket"]
    license: str


def otto(root: Path = DATA / "otto") -> Dataset:
    """OTTO: anonymous sessions with clicks, carts and orders over 4 weeks; no item text.

    Only the official *train* file is used. The official test week is unusable for evaluation:
    its sessions are truncated at a random point (the remainder are the competition's hidden
    labels), so it holds about a quarter of a normal week's events. The split is therefore inside
    the fully observed four weeks: test = the last week, validation = the week before
    (research_ledger.md, deviation D15).
    """
    events = pl.scan_parquet(root / "train.parquet").select(
        pl.lit(-1, pl.Int64).alias("user"),
        pl.col("session").cast(pl.Int64),
        pl.col("aid").cast(pl.Int64).alias("item"),
        pl.col("ts").cast(pl.Int64),
        pl.col("type").alias("event"),
    )
    hi = events.select(pl.col("ts").max()).collect().item()
    t_end = int(hi) + 1
    t_test = t_end - 7 * DAY
    items = events.select("item").unique().sort("item").collect()
    return Dataset(
        "otto",
        events,
        items,
        None,
        t_test - 7 * DAY,
        t_test,
        t_end,
        "cobrowse",
        "next_item",
        "CC BY 4.0",
    )


def amazon_grocery(root: Path = DATA / "amazon") -> Dataset:
    """Amazon Reviews 2023, Grocery & Gourmet Food (research use), from 2018 on.

    Reviews are purchase events; one basket per (user, day). Returning customers make this the
    next-basket task. Item text (title, features, description), a category taxonomy and prices
    are available, so content methods are evaluated here. Built by ``bench.data.amazon``.
    Test = the last 182 days, validation = the 182 days before.
    """
    events = pl.scan_parquet(root / "events.parquet")
    hi = events.select(pl.col("ts").max()).collect().item()
    t_end = int(hi) + 1
    t_test = t_end - 182 * DAY
    items = pl.read_parquet(root / "items.parquet").select(
        "item", "title", "text", "category", "price", "brand_name"
    )
    categories = pl.read_parquet(root / "categories.parquet")
    return Dataset(
        "amazon_grocery",
        events,
        items,
        categories,
        t_test - 182 * DAY,
        t_test,
        t_end,
        "taxonomy",
        "next_basket",
        "Amazon Reviews 2023: research use (McAuley Lab); not commercially cleared",
    )


def hm(root: Path = DATA / "hm") -> Dataset:
    """H&M Personalized Fashion Recommendations (Kaggle competition data, research use only).

    Purchases only, one basket per (customer, day), with a three-level product taxonomy and
    product text. Built by ``bench.data.hm``. Test = the last 7 days of the competition's
    training file (the horizon the competition itself predicts), validation = the 7 days before.
    """
    events = pl.scan_parquet(root / "events.parquet")
    hi = events.select(pl.col("ts").max()).collect().item()
    t_end = int(hi) + DAY  # timestamps are day starts: the last day ends one day later
    t_test = t_end - 7 * DAY
    items = pl.read_parquet(root / "items.parquet").select(
        "item", "title", "text", "category", "price"
    )
    categories = pl.read_parquet(root / "categories.parquet")
    return Dataset(
        "hm",
        events,
        items,
        categories,
        t_test - 7 * DAY,
        t_test,
        t_end,
        "taxonomy",
        "next_basket",
        "H&M competition rules: non-commercial research use; no redistribution",
    )


DATASETS = {"otto": otto, "amazon_grocery": amazon_grocery, "hm": hm}


def load(name: str) -> Dataset:
    return DATASETS[name]()
