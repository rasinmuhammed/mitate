from pathlib import Path

import numpy as np
import polars as pl

from bench.data.dunnhumby import build
from bench.datasets import DAY, dunnhumby

COLS = [
    "household_key",
    "BASKET_ID",
    "DAY",
    "PRODUCT_ID",
    "QUANTITY",
    "SALES_VALUE",
    "STORE_ID",
    "RETAIL_DISC",
    "TRANS_TIME",
    "WEEK_NO",
    "COUPON_DISC",
    "COUPON_MATCH_DISC",
]
FUEL = 6_000_000


def _raw(tmp: Path) -> Path:
    """Twenty households, one basket every third day for ninety days. Each basket holds products
    h%5 and 10+h%3, a void line, and a fuel line; neither of the last two is merchandise."""
    rows = []
    for h in range(1, 21):
        for day in range(1, 91, 3):
            basket = 10_000_000 + h * 1000 + day
            time = 800 + h * 10 + day % 7  # HHMM
            for p, qty, value in ((h % 5, 1, 1.5), (10 + h % 3, 2, 3.0), (h % 5 + 20, 0, 0.0)):
                rows.append((h, basket, day, 1_000_000 + p, qty, value, 1, 0.0, time, 1, 0.0, 0.0))
            rows.append((h, basket, day, FUEL, 400, 40.0, 1, 0.0, time, 1, 0.0, 0.0))
    pl.DataFrame(rows, schema=COLS, orient="row").write_csv(tmp / "transaction_data.csv")
    ids = [1_000_000 + p for p in range(25)] + [FUEL]
    pl.DataFrame(
        {
            "PRODUCT_ID": ids,
            "MANUFACTURER": [1] * len(ids),
            "DEPARTMENT": ["GROCERY"] * 10 + ["PRODUCE"] * 15 + ["KIOSK-GAS"],
            "BRAND": ["National", "Private"] * 13,
            "COMMODITY_DESC": ["SOUP"] * 10 + ["FRUIT"] * 15 + ["GASOLINE"],
            "SUB_COMMODITY_DESC": ["CONDENSED SOUP"] * 10 + ["APPLES"] * 15 + ["GASOLINE-REG"],
            "CURR_SIZE_OF_PRODUCT": ["10.5 OZ"] * 10 + [None] * 16,
        }
    ).write_csv(tmp / "product.csv")
    return tmp


def test_dunnhumby_baskets_merchandise_taxonomy_and_split(tmp_path: Path) -> None:
    out = tmp_path / "dunnhumby"
    m = build(_raw(tmp_path), out)
    baskets = 20 * 30
    # four lines per basket; the void and the fuel line are dropped
    assert m["lines"] == baskets * 4 and m["events"] == baskets * 2 and m["baskets"] == baskets
    ds = dunnhumby(out)
    ev = ds.events.collect()
    # one timestamp and one household per basket, at the basket's time of day
    per = ev.group_by("session").agg(pl.col("ts").n_unique(), pl.col("user").n_unique())
    assert per["ts"].max() == 1 and per["user"].max() == 1
    assert ev["ts"].min() == 8 * 3600 + 11 * 60 and (ev["event"] == 2).all()
    titles = set(ds.items["title"].to_list())
    assert "CONDENSED SOUP, National, 10.5 OZ" in titles and "APPLES, Private" in titles
    assert not any("GASOLINE" in t for t in titles)
    # the split is in whole days: 28-day test and validation windows ending after the last basket
    assert ds.t_end % DAY == 0 and ds.t_end - ds.t_test == 28 * DAY
    assert ds.t_test - ds.t_val == 28 * DAY
    assert ds.t_test <= ev["ts"].max() < ds.t_end
    # three-level tree: department > commodity > sub-commodity
    cats = ds.categories
    assert cats is not None
    leaf = ds.items.join(cats, on="category").select("name").to_series().to_list()
    assert set(leaf) == {"CONDENSED SOUP", "APPLES"}
    assert len(cats) == 6
    # deterministic ids: rebuilding gives identical tables
    assert build(_raw(tmp_path), tmp_path / "again") == m
    a = pl.read_parquet(out / "events.parquet")
    b = pl.read_parquet(tmp_path / "again" / "events.parquet")
    assert a.equals(b) and np.array_equal(a["session"].to_numpy(), b["session"].to_numpy())


def test_dunnhumby_feeds_the_store_sampler_and_next_basket_task(tmp_path: Path) -> None:
    from bench import tasks
    from bench.simulator import sample_store

    build(_raw(tmp_path), tmp_path / "dunnhumby")
    ds = dunnhumby(tmp_path / "dunnhumby")
    st = sample_store(ds, 200, 8, seed=0)
    assert st is not None and st.catalog.category is not None
    pre = st.log.ts < ds.t_test
    assert 200 <= int(pre.sum()) < 200 + 60  # at most one household's events over the target
    es = tasks.next_basket(st.log, ds.t_test, ds.t_end)
    assert es.targets.shape[0] > 0 and es.targets.nnz > 0
