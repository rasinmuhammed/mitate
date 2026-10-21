"""H&M Personalized Fashion Recommendations ingestion: competition CSVs -> benchmark Parquet.

Inputs (Kaggle competition data; research use under the competition rules, never redistributed,
see docs/licenses.md): ``transactions_train.csv`` and ``articles.csv``.

* Events: every transaction is an ORDER. Dates carry no time of day, so one basket is one
  (customer, day); an article bought twice in one basket counts once. ``ts`` is the day's start
  in UTC seconds, which makes a whole basket share one timestamp, so no item of a basket is ever
  history for another (docs/protocol.md, section 2).
* Items: the product name is the title; product type, group, colour, department and the free-text
  description form the text. The category tree is index group > product group > product type.
  ``price`` is the article's median transaction price (H&M scales prices; only ratios matter).
* Customers and articles get dense ids in sorted order of their raw ids, so ingestion is
  deterministic.

Usage: python -m bench.data.hm <competition-data-dir> data/hm
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import polars as pl

DAY = 86_400
LEVELS = ("index_group_name", "product_group_name", "product_type_name")
TEXT = (
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "perceived_colour_value_name",
    "department_name",
    "section_name",
    "garment_group_name",
    "detail_desc",
)


def build(raw: Path, dst: Path) -> dict[str, Any]:
    dst.mkdir(parents=True, exist_ok=True)
    tx = (
        pl.scan_csv(
            raw / "transactions_train.csv",
            schema_overrides={"customer_id": pl.Utf8, "article_id": pl.Int64, "price": pl.Float64},
        )
        .select(
            "customer_id",
            "article_id",
            "price",
            (pl.col("t_dat").str.to_date("%Y-%m-%d").cast(pl.Datetime("ms")).dt.epoch("s"))
            .cast(pl.Int64)
            .alias("ts"),
        )
        .collect()
    )
    articles = pl.read_csv(raw / "articles.csv", schema_overrides={"article_id": pl.Int64})
    # every article that was ever sold, plus its catalog row (all sold articles have one)
    items = (
        tx.group_by("article_id")
        .agg(pl.col("price").median())
        .join(articles, on="article_id", how="left")
        .sort("article_id")
        .with_row_index("item")
        .with_columns(pl.col("item").cast(pl.Int64))
    )

    paths: dict[tuple[str, ...], int] = {}
    parent: list[int] = []
    names: list[str] = []
    leaf: list[int] = []
    for row in items.select(LEVELS).iter_rows():
        p = tuple(str(v) for v in row if v is not None and str(v).strip())
        node = -1
        for depth in range(1, len(p) + 1):
            key = p[:depth]
            if key not in paths:
                paths[key] = len(names)
                parent.append(node)
                names.append(key[-1])
            node = paths[key]
        leaf.append(node)
    items = items.with_columns(pl.Series("category", leaf, dtype=pl.Int64))
    categories = pl.DataFrame(
        {"category": list(range(len(names))), "parent": parent, "name": names},
        schema={"category": pl.Int64, "parent": pl.Int64, "name": pl.Utf8},
    )

    users = tx.select("customer_id").unique().sort("customer_id").with_row_index("user")
    ev = (
        tx.join(users, on="customer_id")
        .join(items.select("article_id", "item"), on="article_id")
        .select(pl.col("user").cast(pl.Int64), "item", "ts")
        .unique()  # one row per (customer, day, article)
        .sort("user", "ts", "item")
    )
    ev = ev.with_columns(
        pl.struct("user", "ts").rank("dense").cast(pl.Int64).alias("session"),
        pl.lit(2, pl.UInt8).alias("event"),
    ).select("user", "session", "item", "ts", "event")

    items_out = items.select(
        "item",
        "article_id",
        pl.col("prod_name").fill_null("").alias("title"),
        pl.concat_str([pl.col(c).fill_null("") for c in TEXT], separator=". ").alias("text"),
        "category",
        "price",
    )
    ev.write_parquet(dst / "events.parquet", compression="zstd")
    items_out.write_parquet(dst / "items.parquet", compression="zstd")
    categories.write_parquet(dst / "categories.parquet")
    manifest = {
        "events": ev.height,
        "transactions": tx.height,
        "users": users.height,
        "items": items_out.height,
        "categories": categories.height,
        "first_day": int(ev["ts"].min()),  # type: ignore[arg-type]
        "last_day": int(ev["ts"].max()),  # type: ignore[arg-type]
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(Path(sys.argv[1]), Path(sys.argv[2])), indent=1))
