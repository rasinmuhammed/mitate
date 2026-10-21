"""Amazon Grocery ingestion: local gzip JSONL (hash-checked by scripts/reproduce.sh) -> Parquet.

Inputs in ``data/raw/amazon`` (McAuley Lab, research use; see docs/licenses.md):

* ``Grocery_and_Gourmet_Food.jsonl.gz``: Amazon Reviews 2023 reviews. Each review is treated as a
  purchase event at the review time, the standard proxy (not every purchase is reviewed, and
  reviews lag purchases; stated in docs/protocol.md).
* ``meta_Grocery_and_Gourmet_Food.jsonl.gz``: 2023 item metadata (title, store, categories,
  price, features, description).
* ``v2018_meta_Grocery_and_Gourmet_Food.json.gz``: Amazon 2018 metadata, used only for its
  ``also_buy`` lists (complement ground truth; the 2023 ``bought_together`` field is empty,
  ledger D5) and ``also_view`` lists (substitute ground truth for the similar-items placement,
  following McAuley et al., KDD 2015).

Outputs in ``data/amazon``: ``reviews.parquet``, ``meta.parquet``, ``also_buy.parquet`` (2018 asin
-> co-bought asin, one row per pair), the benchmark tables ``events/items/categories.parquet``
(see ``build_tables``) and ``manifest.json`` with row counts.

Usage: python -m bench.data.amazon data/raw/amazon data/amazon
"""

from __future__ import annotations

import gzip
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

CHUNK = 500_000
CATEGORY = "Grocery_and_Gourmet_Food"

REVIEW_SCHEMA = pa.schema(
    [
        ("user_id", pa.string()),
        ("parent_asin", pa.string()),
        ("asin", pa.string()),
        ("ts", pa.int64()),
        ("rating", pa.float32()),
        ("verified", pa.bool_()),
    ]
)
META_SCHEMA = pa.schema(
    [
        ("parent_asin", pa.string()),
        ("title", pa.string()),
        ("store", pa.string()),
        ("main_category", pa.string()),
        ("categories", pa.list_(pa.string())),
        ("price", pa.float64()),
        ("features", pa.string()),
        ("description", pa.string()),
        ("rating_number", pa.int64()),
    ]
)
ALSO_BUY_SCHEMA = pa.schema([("asin", pa.string()), ("also_buy", pa.string())])
ALSO_VIEW_SCHEMA = pa.schema([("asin", pa.string()), ("also_view", pa.string())])


def price(v: object) -> float | None:
    if v in (None, "", "None"):
        return None
    try:
        return float(str(v).replace("$", "").replace(",", "").strip())
    except ValueError:
        return None  # ranges ("$5.99 - $9.99") and HTML fragments in the 2018 files


def text(x: object, limit: int = 2000) -> str:
    if isinstance(x, list):
        return " ".join(str(t) for t in x)[:limit]
    return str(x or "")[:limit]


def review_rows(d: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield {
        "user_id": d["user_id"],
        "parent_asin": d["parent_asin"],
        "asin": d.get("asin"),
        "ts": int(d["timestamp"]) // 1000,  # milliseconds in the 2023 release
        "rating": float(d.get("rating") or 0),
        "verified": bool(d.get("verified_purchase")),
    }


def meta_rows(d: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield {
        "parent_asin": d["parent_asin"],
        "title": d.get("title") or "",
        "store": d.get("store") or "",
        "main_category": d.get("main_category") or "",
        "categories": [str(c) for c in (d.get("categories") or [])],
        "price": price(d.get("price")),
        "features": text(d.get("features")),
        "description": text(d.get("description")),
        "rating_number": int(d.get("rating_number") or 0),
    }


def also_buy_rows(d: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for b in d.get("also_buy") or []:
        yield {"asin": d["asin"], "also_buy": str(b)}


def also_view_rows(d: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for b in d.get("also_view") or []:
        yield {"asin": d["asin"], "also_view": str(b)}


def convert(
    src: Path,
    dest: Path,
    schema: pa.Schema,
    rows: Callable[[dict[str, Any]], Iterator[dict[str, Any]]],
) -> dict[str, int]:
    tmp = dest.with_suffix(".partial")
    buf: dict[str, list[Any]] = {f.name: [] for f in schema}
    n_in = n_out = n_bad = 0
    with (
        gzip.open(src, "rt", encoding="utf-8") as f,
        pq.ParquetWriter(tmp, schema, compression="zstd") as w,
    ):
        for line in f:
            if not line.strip():
                continue
            n_in += 1
            try:
                recs = list(rows(json.loads(line)))
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                n_bad += 1
                continue
            for r in recs:
                for k, v in r.items():
                    buf[k].append(v)
                n_out += 1
            if len(buf[schema[0].name]) >= CHUNK:
                w.write_table(pa.table(buf, schema=schema))
                buf = {f.name: [] for f in schema}
                print(f"  {dest.name}: {n_in:,} lines", flush=True)
        if buf[schema[0].name]:
            w.write_table(pa.table(buf, schema=schema))
    tmp.rename(dest)
    return {"lines": n_in, "rows": n_out, "unparseable_lines": n_bad}


def main(raw: str, out: str) -> None:
    src, dst = Path(raw), Path(out)
    dst.mkdir(parents=True, exist_ok=True)
    manifest_path = dst / "manifest.json"
    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    )
    jobs = [
        ("reviews", f"{CATEGORY}.jsonl.gz", REVIEW_SCHEMA, review_rows),
        ("meta", f"meta_{CATEGORY}.jsonl.gz", META_SCHEMA, meta_rows),
        ("also_buy", f"v2018_meta_{CATEGORY}.json.gz", ALSO_BUY_SCHEMA, also_buy_rows),
        ("also_view", f"v2018_meta_{CATEGORY}.json.gz", ALSO_VIEW_SCHEMA, also_view_rows),
    ]
    for name, file, schema, rows in jobs:
        dest = dst / f"{name}.parquet"
        if dest.exists():
            print(f"skip {dest} (exists)")
            continue
        print(f"converting {file}", flush=True)
        manifest[name] = {"source": file, **convert(src / file, dest, schema, rows)}
        manifest_path.write_text(json.dumps(manifest, indent=2))
    if not (dst / "events.parquet").exists():
        manifest["benchmark_tables"] = build_tables(dst)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


SINCE = 1514764800  # 2018-01-01 UTC: five-plus years of history before the test window
DAY = 86_400


def build_tables(dst: Path) -> dict[str, Any]:
    """Benchmark tables with dense integer ids: ``events``, ``items`` and ``categories``.

    One basket per (user, day). Categories are the 2023 metadata paths with the shared root
    ("Grocery & Gourmet Food") dropped; items without a path get category -1 and are never chosen
    when growing a store's assortment.
    """
    import polars as pl

    rev = (
        pl.scan_parquet(dst / "reviews.parquet")
        .filter(pl.col("ts") >= SINCE)
        .select("user_id", "parent_asin", "ts")
        .collect()
    )
    meta = pl.read_parquet(dst / "meta.parquet").unique("parent_asin", keep="first")
    items = (
        rev.select("parent_asin")
        .unique()
        .join(meta, on="parent_asin", how="left")
        .sort("parent_asin")
    )
    items = items.with_row_index("item").with_columns(pl.col("item").cast(pl.Int64))
    # category tree from path prefixes
    paths: dict[tuple[str, ...], int] = {}
    parent: list[int] = []
    names: list[str] = []
    leaf: list[int] = []
    for cats in items["categories"].to_list():
        p = tuple(c for c in (cats or []) if c != "Grocery & Gourmet Food")
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
    users = rev.select("user_id").unique().sort("user_id").with_row_index("user")
    ev = (
        rev.join(users, on="user_id")
        .join(items.select("parent_asin", "item"), on="parent_asin")
        .with_columns(
            pl.col("user").cast(pl.Int64),
            (pl.col("ts") // DAY).alias("day"),
        )
        .sort("user", "ts", "item")
    )
    # basket id = dense (user, day)
    ev = ev.with_columns(
        pl.struct("user", "day").rank("dense").cast(pl.Int64).alias("session"),
        pl.lit(2, pl.UInt8).alias("event"),
    ).select("user", "session", "item", "ts", "event")
    items_out = items.select(
        "item",
        "parent_asin",
        pl.col("title").fill_null(""),
        pl.concat_str(
            [pl.col("features").fill_null(""), pl.col("description").fill_null("")], separator=" "
        ).alias("text"),
        "category",
        "price",
        pl.col("store").fill_null("").alias("brand_name"),
    )
    ev.write_parquet(dst / "events.parquet", compression="zstd")
    items_out.write_parquet(dst / "items.parquet", compression="zstd")
    categories.write_parquet(dst / "categories.parquet")
    return {
        "events": ev.height,
        "users": users.height,
        "items": items_out.height,
        "categories": categories.height,
        "uncategorised_items": int((items_out["category"] < 0).sum()),
        "since": SINCE,
    }


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
