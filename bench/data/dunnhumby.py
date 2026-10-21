"""dunnhumby The Complete Journey ingestion: source CSVs -> benchmark Parquet.

Inputs (dunnhumby Source Files; research use under the site terms, never redistributed, see
docs/licenses.md): ``transaction_data.csv`` and ``product.csv``.

* Events: every transaction line is an ORDER. A basket is one ``BASKET_ID``, which in the source
  always has a single household, day and time; ``ts`` is that moment in seconds from the start
  of day 1, so a whole basket shares one timestamp and no item of a basket is ever history for
  another (docs/protocol.md, section 2).
* Only merchandise is kept (ledger D26): voids and free lines (quantity or value at or below
  zero) and the departments in ``NOT_MERCHANDISE`` (fuel, coupons, postage, prescriptions and
  other store services) are dropped.
* Items: the source has no product names, so the sub-commodity, brand type and pack size form
  the title; department, commodity and sub-commodity form the text and the three-level category
  tree. ``price`` is the product's median unit price over the kept lines.
* Households, baskets and products get dense ids in sorted order of their raw ids, so ingestion
  is deterministic.

Usage: python -m bench.data.dunnhumby data/raw/dunnhumby data/dunnhumby
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import polars as pl

DAY = 86_400
LEVELS = ("DEPARTMENT", "COMMODITY_DESC", "SUB_COMMODITY_DESC")
NOT_MERCHANDISE = frozenset(
    {
        "",
        "KIOSK-GAS",
        "MISC SALES TRAN",
        "MISC. TRANS.",
        "COUP/STR & MFG",
        "POSTAL CENTER",
        "CHARITABLE CONT",
        "RX",
        "PHARMACY SUPPLY",
        "PHOTO",
        "VIDEO RENTAL",
        "CNTRL/STORE SUP",
        "PROD-WHS SALES",
    }
)


def _label(*cols: str) -> pl.Expr:
    """Non-empty values of ``cols`` joined into one phrase."""
    return (
        pl.concat_list([pl.col(c) for c in cols])
        .list.eval(pl.element().filter(pl.element() != ""))
        .list.join(", ")
    )


def build(raw: Path, dst: Path) -> dict[str, Any]:
    dst.mkdir(parents=True, exist_ok=True)
    products = pl.read_csv(raw / "product.csv", infer_schema_length=None).with_columns(
        pl.col(c).fill_null("").str.strip_chars()
        for c in (*LEVELS, "BRAND", "CURR_SIZE_OF_PRODUCT")
    )
    lines = pl.scan_csv(raw / "transaction_data.csv", infer_schema_length=None).collect()
    tx = (
        lines.join(products.select("PRODUCT_ID", "DEPARTMENT"), on="PRODUCT_ID", how="left")
        .filter(
            (pl.col("QUANTITY") > 0)
            & (pl.col("SALES_VALUE") > 0)
            & ~pl.col("DEPARTMENT").fill_null("").is_in(list(NOT_MERCHANDISE))
        )
        .with_columns(
            (
                (pl.col("DAY") - 1) * DAY
                + (pl.col("TRANS_TIME") // 100) * 3600
                + (pl.col("TRANS_TIME") % 100) * 60
            ).alias("ts")
        )
    )

    items = (
        tx.group_by("PRODUCT_ID")
        .agg((pl.col("SALES_VALUE") / pl.col("QUANTITY")).median().alias("price"))
        .join(products, on="PRODUCT_ID", how="left")
        .sort("PRODUCT_ID")
        .with_row_index("item")
        .with_columns(pl.col("item").cast(pl.Int64))
    )

    paths: dict[tuple[str, ...], int] = {}
    parent: list[int] = []
    names: list[str] = []
    leaf: list[int] = []
    for row in items.select(LEVELS).iter_rows():
        p = tuple(v for v in row if v)
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

    users = tx.select("household_key").unique().sort("household_key").with_row_index("user")
    baskets = tx.select("BASKET_ID").unique().sort("BASKET_ID").with_row_index("session")
    ev = (
        tx.join(users, on="household_key")
        .join(baskets, on="BASKET_ID")
        .join(items.select("PRODUCT_ID", "item"), on="PRODUCT_ID")
        .select(
            pl.col("user").cast(pl.Int64),
            pl.col("session").cast(pl.Int64),
            "item",
            pl.col("ts").cast(pl.Int64),
            pl.lit(2, pl.UInt8).alias("event"),
        )
        .unique(["session", "item"])  # one row per (basket, product)
        .sort("user", "ts", "session", "item")
    )

    items_out = items.select(
        "item",
        pl.col("PRODUCT_ID").alias("product_id"),
        _label("SUB_COMMODITY_DESC", "BRAND", "CURR_SIZE_OF_PRODUCT").alias("title"),
        pl.concat_str([pl.col(c) for c in LEVELS], separator=". ").alias("text"),
        "category",
        "price",
    )
    ev.write_parquet(dst / "events.parquet", compression="zstd")
    items_out.write_parquet(dst / "items.parquet", compression="zstd")
    categories.write_parquet(dst / "categories.parquet")
    manifest = {
        "events": ev.height,
        "lines": lines.height,
        "lines_kept": tx.height,
        "users": users.height,
        "baskets": baskets.height,
        "items": items_out.height,
        "categories": categories.height,
        "first_ts": int(ev["ts"].min()),  # type: ignore[arg-type]
        "last_ts": int(ev["ts"].max()),  # type: ignore[arg-type]
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(Path(sys.argv[1]), Path(sys.argv[2])), indent=1))
