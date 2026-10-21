import numpy as np
import pytest

from mitate.complements import CategoryGraph, Edge, EdgeStatus, parse_proposals
from mitate.constraints import Constraints, Context, ItemState, Placement, Rules
from mitate.data import Catalog, Event, Log
from mitate.engine import Engine, EngineConfig, Shopper, repurchase_first
from mitate.explain import Evidence, _co_units
from tests.test_leakage import random_log

N = 12
CATS = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], np.int32)


def _catalog() -> Catalog:
    return Catalog(
        n_items=N,
        title=[f"item {i}" for i in range(N)],
        category=CATS,
        category_names=["shoes", "socks", "hats"],
        price=np.array([10.0, 12, 11, 100, 2, 3, 2, 4, 20, 22, 25, 21]),
    )


def _vectors() -> np.ndarray:
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(3, 8))
    V = centres[CATS] + 0.1 * rng.normal(size=(N, 8))
    return (V / np.linalg.norm(V, axis=1, keepdims=True)).astype(np.float32)


def _state(**kw) -> ItemState:
    s = ItemState(
        stock=np.full(N, np.nan), category=CATS, price=_catalog().price, regions=[None] * N
    )
    return ItemState(**{**s.__dict__, **kw})


def _empty_log() -> Log:
    z: list[int] = []
    return Log.build(z, z, z, z, z, 0, N)


def _co_purchase_log(pairs: int) -> Log:
    """`pairs` shoppers each order items 0 and 4 together; 3 shoppers order 0 and 8."""
    u, i, t, e, s = [], [], [], [], []
    for k in range(pairs + 3):
        other = 4 if k < pairs else 8
        for it in (0, other):
            u.append(k)
            i.append(it)
            t.append(1000 * k + it)
            e.append(Event.ORDER)
            s.append(k)
    return Log.build(u, i, t, e, s, pairs + 3, N)


# ---------------------------------------------------------------------------- constraints
def test_hard_filters_pins_and_diversity() -> None:
    regions = [None] * N
    regions[1] = frozenset({"56"})  # ships only to PIN prefix 56
    st = _state(stock=np.array([np.nan, 5, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1.0]), regions=regions)
    rules = Rules(
        blocked_items=frozenset({5}),
        blocked_categories=frozenset({2}),
        pins={"next": [7, 2]},  # 2 is out of stock: a pin never overrides a hard filter
        max_per_category=2,
    )
    c = Constraints(st, rules)
    S = np.linspace(1, 0, N)
    got = c.slate(S, 4, Placement.NEXT, Context(region="400001"))  # 5 eligible items
    assert got.items[0] == 7 and got.pinned[0] and not any(got.pinned[1:])
    assert 1 not in got.items  # not deliverable to 400001
    assert 2 not in got.items and 5 not in got.items and not set(got.items) & {8, 9, 10, 11}
    for cat in (0, 1):  # diversity cap holds while alternatives exist
        assert sum(CATS[i] == cat for i in got.items) <= 2
    near = c.slate(S, 5, Placement.NEXT, Context(region="560034"))
    assert 1 in near.items


def test_price_sanity_relative_to_anchor() -> None:
    c = Constraints(_state(), Rules(similar_price_ratio=2.0, complement_max_price_ratio=1.5))
    S = np.ones(N)
    sim = c.slate(S, N, Placement.SIMILAR, Context(anchor=0)).items
    assert 3 not in sim and 0 not in sim  # 100 is not "similar" to 10; anchor never shown
    comp = c.slate(S, N, Placement.COMPLEMENTS, Context(anchor=4)).items  # price 2 -> max 3
    assert all(_catalog().price[i] <= 3.0 for i in comp)


# ---------------------------------------------------------------------------- evidence
def test_co_units_matches_brute_force() -> None:
    log = random_log(seed=3)
    buy = log.event == Event.ORDER
    unit = np.where(log.user >= 0, log.user, 10_000 + log.session).astype(np.int64)
    M = _co_units(unit[buy], log.item[buy].astype(np.int64), log.n_items).toarray()
    sets: dict[int, set[int]] = {}
    for un, it in zip(unit[buy], log.item[buy], strict=True):
        sets.setdefault(int(un), set()).add(int(it))
    want = np.zeros_like(M)
    for s in sets.values():
        for a in s:
            for b in s:
                want[a, b] += 1
    np.testing.assert_array_equal(M, want)


@pytest.mark.parametrize(("pairs", "shown"), [(4, False), (5, True)])
def test_reason_requires_min_support(pairs: int, shown: bool) -> None:
    ev = Evidence(min_support=5).fit(_co_purchase_log(pairs), _catalog())
    r = ev.reason(Placement.COMPLEMENTS, 4, anchor=0)
    if shown:
        assert r is not None and r.code == "bought_with" and r.evidence["shoppers"] == pairs
        assert r.text == f"{pairs} shoppers who bought item 0 also bought this"
    else:
        assert r is None or r.code != "bought_with"


def test_similar_reason_states_only_true_attributes() -> None:
    ev = Evidence().fit(_empty_log(), _catalog())
    r = ev.reason(Placement.SIMILAR, 1, anchor=0)
    assert r is not None and r.code == "similar" and "same category (shoes)" in r.text
    assert "brand" not in r.text  # the catalog has no brands
    assert ev.reason(Placement.SIMILAR, 8, anchor=4) is None or "same category" not in (
        ev.reason(Placement.SIMILAR, 8, anchor=4).text  # type: ignore[union-attr]
    )


# ---------------------------------------------------------------------------- complements
def test_llm_proposals_are_filtered_to_the_taxonomy() -> None:
    paths = ["Shoes", "Socks", "Hats"]
    text = '```json\n{"pairs": [{"category": "Shoes", "goes_with": ["socks", "Laces", "Shoes"]}]}'
    edges = parse_proposals(text, paths)
    assert [(e.src, e.dst) for e in edges] == [(0, 1)]
    assert parse_proposals("not json", paths) == []


def test_only_approved_graph_edges_are_served() -> None:
    g = CategoryGraph()
    g.merge_proposals([Edge(0, 1), Edge(0, 2)])
    g.review(0, 1, EdgeStatus.APPROVED)
    g.review(0, 2, EdgeStatus.REJECTED)
    eng = Engine(EngineConfig(min_behaviour_events=10**9))
    eng.fit(_empty_log(), _catalog(), _state(), graph=g, vectors=_vectors())
    recs = eng.recommend(Placement.COMPLEMENTS, 4, anchors=[0])
    assert recs and all(CATS[r.item] == 1 for r in recs)
    assert all(r.reason is not None and r.reason.code == "pairs_with" for r in recs)
    assert recs[0].reason.text == "Pairs with shoes (store pairing)"  # type: ignore[union-attr]
    # round trip keeps review decisions
    assert CategoryGraph.from_json(g.to_json()).approved() == {0: [1]}


def test_data_backed_complements_rank_above_graph_only() -> None:
    g = CategoryGraph([Edge(0, 2, EdgeStatus.APPROVED)])
    eng = Engine(EngineConfig(min_behaviour_events=10**9, min_support=5))
    eng.fit(_co_purchase_log(6), _catalog(), _state(), graph=g, vectors=_vectors())
    recs = eng.recommend(Placement.COMPLEMENTS, 3, anchors=[0])
    assert recs[0].item == 4 and recs[0].reason.code == "bought_with"  # type: ignore[union-attr]
    assert {r.item for r in recs[1:]} <= {8, 9, 10, 11}


# ---------------------------------------------------------------------------- engine stages
def test_catalog_only_stage_serves_every_placement() -> None:
    eng = Engine(EngineConfig(min_behaviour_events=1))
    info = eng.fit(_empty_log(), _catalog(), _state(), vectors=_vectors())
    assert info.stage == "catalog_only" and info.active_sources == ()
    sim = eng.recommend(Placement.SIMILAR, 3, anchors=[4])
    assert [CATS[r.item] for r in sim] == [1, 1, 1]
    nxt = eng.recommend(Placement.NEXT, 3, shopper=Shopper(items=[8], ts=[0], events=[0]))
    assert len(nxt) == 3 and CATS[nxt[0].item] == 2
    # nothing approved and no co-purchases yet: related items from other categories, no reason
    comp = eng.recommend(Placement.COMPLEMENTS, 3, anchors=[0])
    assert len(comp) == 3 and all(CATS[r.item] != CATS[0] and r.reason is None for r in comp)


def test_co_bought_substitutes_stay_out_of_complements() -> None:
    # six shoppers each bought shoe 0 with shoe 1 (a substitute) and with sock 4 (a complement)
    shopper = [k for k in range(6) for _ in range(3)]
    log = Log.build(shopper, [0, 1, 4] * 6, list(range(18)), [Event.ORDER] * 18, shopper, 6, N)
    eng = Engine(EngineConfig(min_behaviour_events=10**9, min_support=5))
    eng.fit(log, _catalog(), _state(), vectors=_vectors())
    recs = eng.recommend(Placement.COMPLEMENTS, 4, anchors=[0])
    assert recs[0].item == 4 and all(CATS[r.item] != CATS[0] for r in recs)


def test_related_items_never_outrank_an_approved_pairing() -> None:
    g = CategoryGraph([Edge(0, 2, EdgeStatus.APPROVED)])
    eng = Engine(EngineConfig(min_behaviour_events=10**9))
    eng.fit(_empty_log(), _catalog(), _state(), graph=g, vectors=_vectors())
    recs = eng.recommend(Placement.COMPLEMENTS, 6, anchors=[0])
    assert [CATS[r.item] for r in recs[:4]] == [2, 2, 2, 2]  # the approved category first
    assert all(CATS[r.item] == 1 and r.reason is None for r in recs[4:])


def test_behavioural_stage_uses_the_blend_and_respects_thresholds() -> None:
    log = random_log(seed=5, n_items=N)
    cfg = EngineConfig(
        min_behaviour_events=100,
        sources=("recpop", "itemknn", "pctm"),
        source_min_events={"pctm": 10**9},
    )
    eng = Engine(cfg)
    info = eng.fit(log, _catalog(), _state(), vectors=_vectors())
    assert info.stage == "behavioural" and info.active_sources == ("recpop", "itemknn")
    assert info.ranker == "rrf"
    sh = Shopper(items=[1, 2, 3], ts=[1, 2, 3], events=[0, 0, 0], now=10**8)
    recs = eng.recommend(Placement.NEXT, 5, shopper=sh)
    assert len(recs) == 5 and len({r.item for r in recs}) == 5


def test_returning_customers_see_repurchases_first() -> None:
    log = random_log(seed=6, n_items=N)
    eng = Engine(EngineConfig(min_behaviour_events=100, sources=("recpop",)))
    eng.fit(log, _catalog(), _state(), vectors=_vectors())
    O = int(Event.ORDER)
    sh = Shopper(items=[9, 3, 9, 3, 9], ts=[1, 2, 3, 4, 5], events=[O] * 5, known=True, now=10**8)
    recs = eng.recommend(Placement.RETURNING, 4, shopper=sh)
    assert [r.item for r in recs[:2]] == [9, 3]
    assert recs[0].reason is not None and recs[0].reason.text == "You bought this 3 times before"


def test_diversity_cap_backfills_rather_than_returning_a_short_list() -> None:
    c = Constraints(_state(), Rules(max_per_category=1, blocked_categories=frozenset({1, 2})))
    got = c.slate(np.linspace(1, 0, N), 3, Placement.NEXT, Context())
    assert got.items == [0, 1, 2]  # only category 0 is left: best items in score order


def test_product_on_screen_drives_history_based_sources() -> None:
    """An anonymous visitor on item 0's page: transition sources must condition on item 0."""
    u, i, t, e, s = [], [], [], [], []
    for k in range(30):  # 30 sessions: 0 -> 4 ; 10 sessions of the popular item 9 alone
        for j, it in enumerate((0, 4)):
            u.append(-1), i.append(it), t.append(100 * k + j), e.append(0), s.append(k)
    for k in range(30, 40):
        for j in range(3):
            u.append(-1), i.append(9), t.append(100 * k + j), e.append(0), s.append(k)
    log = Log.build(u, i, t, e, s, 0, N)
    eng = Engine(EngineConfig(min_behaviour_events=10, sources=("pctm",)))
    eng.fit(log, _catalog(), _state(), vectors=_vectors())
    recs = eng.recommend(Placement.NEXT, 3, anchors=[0], shopper=Shopper(now=10**6))
    assert recs[0].item == 4


def test_co_purchase_reason_never_cites_the_item_itself() -> None:
    ev = Evidence(min_support=5).fit(_co_purchase_log(6), _catalog())
    r = ev.reason(Placement.RETURNING, 0, history=[0, 4], orders=[0, 4])
    assert r is not None and r.code == "bought_before"
    r = ev.reason(Placement.NEXT, 0, history=[0])
    assert r is None or r.code != "bought_with"


def test_audit_attributes_each_hidden_item_to_its_first_failing_rule() -> None:
    plain = Engine(EngineConfig(min_behaviour_events=10**9))
    plain.fit(_empty_log(), _catalog(), _state(), vectors=_vectors())
    S = plain._similar(4)
    S[4] = -np.inf
    first, second = (int(i) for i in np.argsort(-S)[:2])
    stock = np.full(N, np.nan)
    stock[first] = 0.0  # the best match is sold out, the runner-up is blocked
    eng = Engine(EngineConfig(min_behaviour_events=10**9))
    rules = Rules(blocked_items=frozenset({second}))
    eng.fit(_empty_log(), _catalog(), _state(stock=stock), rules=rules, vectors=_vectors())
    ranked = eng.rank(Placement.SIMILAR, 1, anchors=[4], audit=True)
    assert ranked.items[0].item not in (first, second)
    assert ranked.hidden == {"out_of_stock": 1, "blocked": 1}
    assert eng.rank(Placement.SIMILAR, 1, anchors=[4]).hidden == {}  # not audited by default


def test_engine_uses_the_ranker_trained_on_exactly_its_active_sources(tmp_path) -> None:
    """A shipped ranker serves a store only if it was trained on the store's active sources; a
    store whose source set no ranker covers falls back to rank fusion."""
    from bench.tasks import next_item
    from mitate.blend import Blend, Variant, ltr_groups
    from mitate.ltr import train_isolated
    from mitate.models.registry import MODELS
    from tests.test_leakage import T0, T1

    names = ("recpop", "itemknn", "pctm")
    log = random_log(seed=8, n_sess=1500)
    cat = Catalog(n_items=log.n_items)
    val = next_item(log, T0, T1, target="successive")
    fr = Blend(sources=[MODELS[n]() for n in names], n_cand=10).fit(log.until(T0), cat)
    frame = fr.candidates(val.queries)
    for variant in (Variant("full"), Variant("no_pctm", drop_sources=("pctm",))):
        X, y, g = ltr_groups([frame], val.targets, variant)
        train_isolated(
            tmp_path / "job.npz",
            tmp_path / f"{variant.name}.json",
            X,
            y,
            g,
            list(frame.feature_names),
        )
    (tmp_path / "MANIFEST.json").write_text("{}")
    catalog = Catalog(n_items=log.n_items)
    state = ItemState.unconstrained(log.n_items)

    def ranker(**kw) -> str:
        cfg = EngineConfig(min_behaviour_events=100, embedder="none", booster_dir=tmp_path, **kw)
        return Engine(cfg).fit(log, catalog, state).ranker

    assert ranker(sources=names) == "full"
    assert ranker(sources=names, source_min_events={"pctm": 10**9}) == "no_pctm"
    assert ranker(sources=("recpop", "itemknn", "seqrules")) == "rrf"


def test_repurchases_lead_by_order_count_then_recency() -> None:
    S = np.array([5.0, 4.0, 3.0, 2.0, 1.0, -np.inf])
    # counts: item 3 three times, item 1 twice, item 4 once
    out = repurchase_first(S, [3, 1, 3, 4, 1, 3], n_rep=2)
    assert np.argsort(-out, kind="stable").tolist() == [3, 1, 0, 2, 4, 5]
    # equal counts: the more recently ordered item first
    assert np.argsort(-repurchase_first(S, [2, 4], n_rep=2))[:2].tolist() == [4, 2]
    # a repurchase lifts an item the ranker would never show
    assert np.argmax(repurchase_first(S, [5], n_rep=1)) == 5
    assert np.array_equal(repurchase_first(S, [], n_rep=5), S)
    assert np.array_equal(S[:5], [5.0, 4.0, 3.0, 2.0, 1.0])  # the input is not modified


def test_benchmark_rail_matches_the_engine() -> None:
    from bench.run import _repurchase_first
    from mitate.data import Queries

    rng = np.random.default_rng(0)
    S = rng.random((2, 12)).astype(np.float32)
    items = np.array([4, 7, 4, 1, 9, 9, 2], np.int32)
    event = np.array(
        [Event.ORDER, Event.VIEW, Event.ORDER, Event.ORDER, Event.ORDER, Event.CART, Event.ORDER],
        np.int8,
    )
    q = Queries(
        indptr=np.array([0, 4, 7], np.int64),
        items=items,
        ts=np.arange(7, dtype=np.int64),
        event=event,
        now=np.array([10, 10], np.int64),
        user=np.array([0, 1], np.int32),
        session_start=np.zeros(2, np.int64),
        anchor=np.full(2, -1, np.int32),
    )
    out = _repurchase_first(S, q, n_rep=2)
    shoppers = [
        Shopper(items=items[:4].tolist(), ts=list(range(4)), events=event[:4].tolist()),
        Shopper(items=items[4:].tolist(), ts=list(range(3)), events=event[4:].tolist()),
    ]
    for r, sh in enumerate(shoppers):
        expected = repurchase_first(S[r], sh.orders(), 2)
        assert np.array_equal(np.argsort(-out[r]), np.argsort(-expected))
