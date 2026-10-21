"""Small-store simulator: carve realistic small shops out of large real datasets.

A simulated store is defined by (dataset, size bucket, seed) and built in three steps:

1. **Coherent sub-catalog.** A shop sells a related assortment, not a random slice of a
   marketplace.
   * ``taxonomy`` datasets: start from a random leaf category and climb the tree, adding whole
     sibling subtrees until the catalog target is reached; trim to the target uniformly at random
     within the final subtree.
   * ``cobrowse`` datasets (no taxonomy, e.g. OTTO): best-first growth from a random seed item
     over the item co-browsing graph (consecutive-event transitions before ``t_test``): repeatedly
     add the frontier item with the largest total edge weight into the current set.
   Both use only pre-``t_test`` information, so the store's definition cannot encode the test
   period.
2. **Restrict the log** to the sub-catalog: the shop only observes its own products.
3. **Thin traffic.** Customers (known users, or anonymous sessions) are put in a seeded random order
   and taken until the store has ``n_interactions`` events before ``t_test``. All events of chosen
   customers are kept (before and after ``t_test``), so per-customer behaviour is intact and the
   test period receives new and returning customers at the dataset's natural rate.

Size buckets are defined by pre-test interaction volume (what a deployed model would have seen);
catalog size scales with the bucket. Both grids live in experiments/configs/stores.yaml.
"""

from __future__ import annotations

import heapq
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl
from numpy.typing import NDArray

from bench.datasets import Dataset
from mitate.data import Catalog, Log


@dataclass(frozen=True)
class Store:
    dataset: str
    bucket: int
    seed: int
    log: Log
    catalog: Catalog
    t_val: int
    t_test: int
    t_end: int
    item_raw: NDArray[np.int64]  # raw (dataset) item id per dense item index
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.dataset}-n{self.bucket}-s{self.seed}"


# --------------------------------------------------------------------------- co-browse graph
def cobrowse_graph(ds: Dataset, cache: Path, top_k: int = 20, mem: str = "3GB") -> pl.DataFrame:
    """Undirected top-k neighbour lists from consecutive distinct events within sessions, using
    events before ``t_test`` only. Cached as Parquet (src, dst, w)."""
    if cache.exists():
        return pl.read_parquet(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.parent / "_cobrowse_events.parquet"
    ds.events.filter(pl.col("ts") < ds.t_test).select("session", "item", "ts").sink_parquet(tmp)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{mem}'; SET threads=4; SET preserve_insertion_order=false;")
    con.execute(f"SET temp_directory='{cache.parent / '_duckdb_tmp'}'")
    con.execute(
        f"""
        COPY (
          WITH seq AS (
            SELECT session, item,
                   lead(item) OVER (PARTITION BY session ORDER BY ts, item) AS nxt
            FROM read_parquet('{tmp}')
          ), pairs AS (
            SELECT least(item, nxt) AS a, greatest(item, nxt) AS b, count(*) AS w
            FROM seq WHERE nxt IS NOT NULL AND nxt <> item GROUP BY 1, 2
          ), both_dir AS (
            SELECT a AS src, b AS dst, w FROM pairs UNION ALL SELECT b, a, w FROM pairs
          )
          SELECT src, dst, w FROM (
            SELECT *, row_number() OVER (PARTITION BY src ORDER BY w DESC, dst) AS r FROM both_dir
          ) WHERE r <= {top_k}
        ) TO '{cache}' (FORMAT parquet, COMPRESSION zstd)
        """
    )
    tmp.unlink()
    return pl.read_parquet(cache)


def grow_cobrowse(graph: pl.DataFrame, seed_item: int, target: int) -> NDArray[np.int64]:
    """Best-first expansion: next item = frontier item with the largest summed edge weight into
    the current set (ties by item id)."""
    g = graph.sort(["src", "w"], descending=[False, True])
    src = g["src"].to_numpy()
    dst = g["dst"].to_numpy()
    w = g["w"].to_numpy().astype(np.float64)
    starts = np.searchsorted(src, np.unique(src))
    keys = np.unique(src)
    index = {int(k): int(s) for k, s in zip(keys, starts, strict=True)}
    ends = {int(k): int(e) for k, e in zip(keys, np.append(starts[1:], src.size), strict=True)}
    chosen: list[int] = []
    in_set: set[int] = set()
    gain: dict[int, float] = {seed_item: np.inf}
    heap: list[tuple[float, int]] = [(-np.inf, seed_item)]
    while heap and len(chosen) < target:
        negw, it = heapq.heappop(heap)
        if it in in_set or -negw != gain.get(it):
            continue  # stale entry
        chosen.append(it)
        in_set.add(it)
        s = index.get(it)
        if s is None:
            continue
        for j in range(s, ends[it]):
            nb = int(dst[j])
            if nb in in_set:
                continue
            gain[nb] = gain.get(nb, 0.0) + w[j]
            heapq.heappush(heap, (-gain[nb], nb))
    return np.asarray(chosen, np.int64)


# --------------------------------------------------------------------------- taxonomy growth
def grow_taxonomy(
    items: pl.DataFrame,
    categories: pl.DataFrame,
    target: int,
    rng: np.random.Generator,
    active: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Climb from a random leaf until the subtree holds >= target active items, then trim. If even
    the top-level subtree is smaller, the store is that whole subtree (smaller than the target,
    and the caller widens or gives up), never items from unrelated departments."""
    parent = dict(
        zip(categories["category"].to_list(), categories["parent"].to_list(), strict=True)
    )
    # uncategorised items (category -1) never seed or join a taxonomy-grown assortment
    item_cat = items.filter(
        pl.col("item").is_in(pl.Series(active).implode()) & (pl.col("category") >= 0)
    )
    cats = item_cat["category"].to_numpy()
    item_ids = item_cat["item"].to_numpy()

    # ancestor chain per category
    def chain(c: int) -> list[int]:
        out = [c]
        while parent.get(out[-1], -1) not in (-1, None):
            out.append(parent[out[-1]])
        return out

    leaves = np.unique(cats)
    start = int(rng.choice(leaves))
    pool = item_ids[cats == start]
    for node in chain(start):
        members = np.array([c for c in leaves if node in chain(int(c))])
        pool = item_ids[np.isin(cats, members)]
        if pool.size >= target:
            return np.sort(rng.choice(pool, target, replace=False))
    return np.sort(pool)


# --------------------------------------------------------------------------- store sampling
def sample_store(
    ds: Dataset,
    n_interactions: int,
    n_items: int,
    seed: int,
    graph: pl.DataFrame | None = None,
    max_grow: int = 3,
) -> Store | None:
    rng = np.random.default_rng(seed)
    pre = ds.events.filter(pl.col("ts") < ds.t_test)
    # polars group_by order is not deterministic: sort before any seeded draw
    activity = pre.group_by("item").len().filter(pl.col("len") >= 20).sort("item").collect()
    active = activity["item"].to_numpy()
    target = n_items
    for _ in range(max_grow):
        if ds.subcatalog == "cobrowse":
            assert graph is not None
            seed_item = int(rng.choice(active))
            sub = grow_cobrowse(graph, seed_item, target)
        else:
            assert ds.categories is not None
            sub = grow_taxonomy(ds.items, ds.categories, target, rng, active)
        ev = ds.events.filter(pl.col("item").is_in(pl.Series(sub).implode())).collect()
        ev = ev.with_columns(
            pl.when(pl.col("user") >= 0)
            .then(pl.col("user"))
            .otherwise(-(pl.col("session") + 1))
            .alias("cust")
        )
        pre_counts = (
            ev.filter(pl.col("ts") < ds.t_test).group_by("cust").len().rename({"len": "n_pre"})
        )
        if int(pre_counts["n_pre"].sum()) >= n_interactions:
            break
        target = int(target * 1.5)  # sub-catalog too quiet: widen the assortment and retry
    else:
        return None
    custs = ev.select("cust").unique().sort("cust").join(pre_counts, on="cust", how="left")
    custs = custs.with_columns(pl.col("n_pre").fill_null(0))
    order = rng.permutation(custs.height)
    n_pre = custs["n_pre"].to_numpy()[order]
    k = int(np.searchsorted(np.cumsum(n_pre), n_interactions)) + 1
    keep = custs["cust"].to_numpy()[order[:k]]
    ev = ev.filter(pl.col("cust").is_in(pl.Series(keep).implode())).sort(["ts", "cust", "item"])
    return _densify(ds, ev, n_interactions, seed, target, n_items)


def widened_targets(n_items: int, max_grow: int = 3) -> list[int]:
    """Every assortment size ``sample_store`` may settle on when asked for ``n_items``."""
    out = [n_items]
    for _ in range(max_grow - 1):
        out.append(int(out[-1] * 1.5))
    return out


def _densify(
    ds: Dataset,
    ev: pl.DataFrame,
    bucket: int,
    seed: int,
    n_items_target: int,
    n_items_requested: int,
) -> Store:
    item_raw = np.unique(ev["item"].to_numpy())
    item_idx = np.searchsorted(item_raw, ev["item"].to_numpy())
    users = ev["user"].to_numpy()
    known_raw = np.unique(users[users >= 0])
    uid = np.where(users >= 0, np.searchsorted(known_raw, users), -1)
    sess_raw, sess = np.unique(ev["session"].to_numpy(), return_inverse=True)
    log = Log.build(
        uid,
        item_idx,
        ev["ts"].to_numpy(),
        ev["event"].to_numpy(),
        sess,
        int(known_raw.size),
        int(item_raw.size),
    )
    catalog = _catalog(ds, item_raw)
    pre = log.ts < ds.t_test
    stats = {
        "n_items": int(item_raw.size),
        "n_items_target": n_items_target,  # after any widening
        "n_items_requested": n_items_requested,
        "n_events_pre_test": int(pre.sum()),
        "n_events_test": int((~pre).sum()),
        "n_users_known": int(known_raw.size),
        "n_sessions": int(sess_raw.size),
        "anon_share": float((users < 0).mean()),
        "cold_test_items": int(
            np.setdiff1d(np.unique(log.item[~pre]), np.unique(log.item[pre])).size
        ),
    }
    return Store(
        ds.name, bucket, seed, log, catalog, ds.t_val, ds.t_test, ds.t_end, item_raw, stats
    )


def _catalog(ds: Dataset, item_raw: NDArray[np.int64]) -> Catalog:
    it = ds.items.filter(pl.col("item").is_in(pl.Series(item_raw).implode())).sort("item")
    assert np.array_equal(it["item"].to_numpy(), item_raw), "item table must cover all events"

    def col(name: str) -> Any:
        return it[name].to_list() if name in it.columns else None

    cat_names: dict[int, str] | list[str] | None  # id -> name, then dense index -> name
    cat_names = None
    cat = None
    if "category" in it.columns and ds.categories is not None:
        cat = it["category"].to_numpy().astype(np.int32)
        cat_names = dict(
            zip(ds.categories["category"].to_list(), ds.categories["name"].to_list(), strict=True)
        )
        uniq = np.unique(cat)
        names = [cat_names[int(c)] for c in uniq]
        cat = np.searchsorted(uniq, cat).astype(np.int32)
        cat_names = names
    brand = brand_names = None
    if "brand_name" in it.columns:
        labels = it["brand_name"].fill_null("").to_numpy().astype(str)
        brand_names, inv = np.unique(labels, return_inverse=True)
        brand = np.where(labels == "", -1, inv).astype(np.int32)
        brand_names = brand_names.tolist()
    return Catalog(
        n_items=int(item_raw.size),
        title=col("title"),
        text=col("text"),
        category=cat,
        category_names=cat_names,
        brand=brand,
        brand_names=brand_names,
        price=it["price"].to_numpy() if "price" in it.columns else None,
        raw_ids=item_raw.tolist(),
    )


def save_manifest(
    stores: list[Store],
    path: Path,
    include_items: bool,
    skipped: Sequence[dict[str, Any]] = (),
) -> None:
    """Seeds and stats for every store; item lists only where the license allows (OTTO).

    ``skipped`` rows record seeds that could not be sampled or evaluated, with the reason. Rows
    already in the file for stores not touched by this run are kept, so runs over different
    buckets add to one manifest.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict[str, Any]] = (
        {r["store"]: r for r in json.loads(path.read_text())} if path.exists() else {}
    )
    for s in stores:
        row: dict[str, Any] = {
            "store": s.name,
            "dataset": s.dataset,
            "bucket": s.bucket,
            "seed": s.seed,
            **s.stats,
        }
        if include_items:
            row["items"] = s.item_raw.tolist()
        rows[s.name] = row
    for r in skipped:
        rows[r["store"]] = r
    ordered = sorted(rows.values(), key=lambda r: (r["bucket"], r["seed"]))
    path.write_text(json.dumps(ordered, indent=1))
