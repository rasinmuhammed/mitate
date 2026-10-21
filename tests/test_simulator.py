import numpy as np
import polars as pl

from bench.simulator import grow_cobrowse, grow_taxonomy, sample_store, widened_targets
from tests.synthetic import marketplace


def test_grow_cobrowse_prefers_strongly_connected_items() -> None:
    # two cliques {0,1,2,3} (weight 10) and {4,5,6} (weight 10), joined weakly via 3-4 (weight 1)
    edges = [(a, b, 10) for c in ([0, 1, 2, 3], [4, 5, 6]) for a in c for b in c if a != b]
    edges += [(3, 4, 1), (4, 3, 1)]
    g = pl.DataFrame(edges, schema={"src": pl.Int64, "dst": pl.Int64, "w": pl.Int64}, orient="row")
    got = grow_cobrowse(g, 0, 4)
    assert set(got.tolist()) == {0, 1, 2, 3}
    assert set(grow_cobrowse(g, 0, 6).tolist()) >= {0, 1, 2, 3, 4}


def test_grow_taxonomy_stays_inside_one_subtree() -> None:
    cats = pl.DataFrame(
        {"category": [0, 1, 2, 3], "parent": [-1, 0, 0, 1], "name": ["root", "a", "b", "a1"]}
    )
    items = pl.DataFrame({"item": np.arange(30), "category": np.repeat([3, 1, 2], 10)})
    rng = np.random.default_rng(0)
    for _ in range(20):
        got = grow_taxonomy(items, cats, 12, rng, np.arange(30))
        c = items.filter(pl.col("item").is_in(pl.Series(got).implode()))["category"]
        # 12 items need the 'a' subtree (cats 1 and 3 = 20 items) unless the root was reached
        assert got.size == 12
        assert set(c.to_list()) <= {1, 3} or set(c.to_list()) <= {1, 2, 3}


def test_grow_taxonomy_never_mixes_departments() -> None:
    """Two top-level departments of 10 items each and a 15-item target: the store is one whole
    department (smaller than the target), never items from both."""
    cats = pl.DataFrame({"category": [0, 1], "parent": [-1, -1], "name": ["tea", "pans"]})
    items = pl.DataFrame({"item": np.arange(20), "category": np.repeat([0, 1], 10)})
    rng = np.random.default_rng(0)
    for _ in range(10):
        got = grow_taxonomy(items, cats, 15, rng, np.arange(20))
        c = set(items.filter(pl.col("item").is_in(pl.Series(got).implode()))["category"])
        assert got.size == 10 and len(c) == 1


def test_sample_store_hits_target_and_is_deterministic(tmp_path) -> None:
    from bench.simulator import cobrowse_graph

    ds = marketplace(seed=1)
    g = cobrowse_graph(ds, tmp_path / "g.parquet")
    a = sample_store(ds, 3000, 100, seed=4, graph=g)
    b = sample_store(ds, 3000, 100, seed=4, graph=g)
    assert a is not None and b is not None
    assert a.stats == b.stats and np.array_equal(a.log.item, b.log.item)
    pre = int((a.log.ts < ds.t_test).sum())
    assert 3000 <= pre < 3000 + 20  # at most one customer's events over target
    assert a.log.ts.max() < ds.t_end
    assert a.stats["n_items_requested"] == 100
    assert a.stats["n_items_target"] in widened_targets(100)


def test_widened_targets_follow_the_sampler() -> None:
    assert widened_targets(3000) == [3000, 4500, 6750]
