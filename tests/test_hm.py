from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from bench.data.hm import build
from bench.datasets import DAY, hm


def _raw(tmp: Path) -> Path:
    """Twenty customers over thirty days; customer c buys articles c%5 and 10+c%3 on odd days."""
    start = date(2020, 8, 24)
    rows = [
        ((start + timedelta(days=d)).isoformat(), f"{c:064x}", 108775000 + a, 0.01 * (a + 1), 2)
        for c in range(20)
        for d in range(1, 30, 2)
        for a in (c % 5, 10 + c % 3, c % 5)  # the repeat inside a basket must collapse
    ]
    cols = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
    pl.DataFrame(rows, schema=cols, orient="row").write_csv(tmp / "transactions_train.csv")
    groups = ["Ladieswear", "Menswear"]
    pl.DataFrame(
        {
            "article_id": [108775000 + a for a in range(15)],
            "prod_name": [f"Item {a}" for a in range(15)],
            "index_group_name": [groups[a % 2] for a in range(15)],
            "product_group_name": ["Garment Upper body"] * 10 + ["Accessories"] * 5,
            "product_type_name": ["T-shirt"] * 10 + ["Bag"] * 5,
            **{
                c: ["x"] * 15
                for c in (
                    "graphical_appearance_name",
                    "colour_group_name",
                    "perceived_colour_value_name",
                    "department_name",
                    "section_name",
                    "garment_group_name",
                )
            },
            "detail_desc": [None] * 15,
        }
    ).write_csv(tmp / "articles.csv")
    return tmp


def test_hm_baskets_taxonomy_and_split(tmp_path: Path) -> None:
    out = tmp_path / "hm"
    m = build(_raw(tmp_path), out)
    # 20 customers x 15 baskets x 3 rows, of which one repeats inside its basket
    assert m["transactions"] == 20 * 15 * 3 and m["events"] == 20 * 15 * 2
    ds = hm(out)
    ev = ds.events.collect()
    # one basket per (customer, day), all at the day's start
    per = ev.group_by("session").agg(pl.col("ts").n_unique(), pl.col("user").n_unique())
    assert per["ts"].max() == 1 and per["user"].max() == 1
    assert (ev["ts"] % DAY == 0).all() and (ev["event"] == 2).all()
    # the last day is inside the test window, which is exactly seven days long
    assert ds.t_end - ds.t_test == 7 * DAY and ds.t_test - ds.t_val == 7 * DAY
    assert ev["ts"].max() < ds.t_end and ev["ts"].max() >= ds.t_test
    # three-level tree: index group > product group > product type
    cats = ds.categories
    assert cats is not None
    leaf = ds.items.join(cats, on="category").select("name").to_series().to_list()
    assert set(leaf) == {"T-shirt", "Bag"}
    depth = {}
    parents = dict(zip(cats["category"].to_list(), cats["parent"].to_list(), strict=True))
    for c in cats["category"].to_list():
        d, p = 1, parents[c]
        while p >= 0:
            d, p = d + 1, parents[p]
        depth[c] = d
    assert max(depth.values()) == 3
    # deterministic ids: rebuilding gives identical tables
    again = build(_raw(tmp_path), tmp_path / "hm2")
    assert again == m
    a = pl.read_parquet(out / "events.parquet")
    b = pl.read_parquet(tmp_path / "hm2" / "events.parquet")
    assert a.equals(b) and np.array_equal(a["session"].to_numpy(), b["session"].to_numpy())


def test_hm_feeds_the_store_sampler_and_next_basket_task(tmp_path: Path) -> None:
    from bench import tasks
    from bench.simulator import sample_store

    build(_raw(tmp_path), tmp_path / "hm")
    ds = hm(tmp_path / "hm")
    st = sample_store(ds, 200, 8, seed=0)
    assert st is not None and st.catalog.category is not None
    pre = st.log.ts < ds.t_test
    assert 200 <= int(pre.sum()) < 200 + 40  # at most one shopper's events over the target
    es = tasks.next_basket(st.log, ds.t_test, ds.t_end)
    assert es.targets.shape[0] > 0 and es.targets.nnz > 0
