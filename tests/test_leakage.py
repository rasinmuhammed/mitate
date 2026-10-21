"""Leakage guards: these must pass for CI. See bench/leakage.py for the rationale."""

import numpy as np
import pytest

from bench.gru4rec import GRU4Rec
from bench.leakage import assert_future_invariant, perturb_after
from bench.tasks import next_basket, next_item
from mitate.data import Catalog, Event, Log
from mitate.models.baselines import EASE, GPTopFreq, ItemKNN, RecencyPopularity, RP3beta
from mitate.models.mf import IALS
from mitate.models.sasrec import GSASRec, SASRecCE
from mitate.models.sequential import PCTM, SeqRules
from mitate.models.sknn import VSTAN

pytestmark = pytest.mark.leakage


def random_log(seed: int = 0, n_users: int = 60, n_items: int = 40, n_sess: int = 900) -> Log:
    """Time-local sessions (minutes long) spread over 100 days; 20% anonymous."""
    rng = np.random.default_rng(seed)
    u, i, t, e, s = [], [], [], [], []
    for sess in range(n_sess):
        user = -1 if rng.random() < 0.2 else int(rng.integers(n_users))
        start = int(rng.integers(0, 100 * 86_400))
        k = int(rng.integers(1, 8))
        for j in range(k):
            u.append(user)
            i.append(int(rng.integers(n_items)))
            t.append(start + 60 * j)
            e.append(int(rng.choice([Event.VIEW, Event.CART, Event.ORDER], p=[0.6, 0.2, 0.2])))
            s.append(sess)
    return Log.build(u, i, t, e, s, n_users, n_items)


T0, T1 = 70 * 86_400, 85 * 86_400


@pytest.mark.parametrize("task", ["basket", "item"])
def test_histories_strictly_precede_query_time(task: str) -> None:
    log = random_log()
    ev = next_basket(log, T0, T1) if task == "basket" else next_item(log, T0, T1, seed=1)
    q = ev.queries
    rows = np.repeat(np.arange(len(q)), q.lengths())
    assert np.all(q.ts < q.now[rows]), "a history event is at/after its query time"
    assert np.all(q.now >= T0) and np.all(q.now < T1)
    assert ev.targets.nnz > 0


def test_log_until_refuses_future() -> None:
    log = random_log()
    past = log.until(T0)
    assert past.ts.max() < T0 and past.horizon == T0
    with pytest.raises(ValueError):
        Log(past.user, past.item, past.ts + T0, past.event, past.session, 60, 40, horizon=T0)


def test_perturbation_keeps_past_and_changes_future() -> None:
    log = random_log()
    p = perturb_after(log, T0, seed=3)
    a, b = log.until(T0), p.until(T0)
    for col in ("user", "item", "ts", "event", "session"):
        np.testing.assert_array_equal(getattr(a, col), getattr(b, col))
    assert len(p) != len(log) or not np.array_equal(p.item, log.item)


MODELS = [GPTopFreq, RecencyPopularity, ItemKNN, RP3beta, EASE, IALS, PCTM, SeqRules, VSTAN]


@pytest.mark.parametrize("cls", MODELS, ids=lambda c: c.name)
def test_model_predictions_invariant_to_future(cls) -> None:
    """Fit on the pre-cutoff log, score validation-window queries built from the same log.
    Changing anything at/after T0 must not change a single score."""
    base = random_log()
    queries = next_basket(base, T0, T1).queries  # fixed query set (histories < T0)

    def pipeline(log: Log) -> np.ndarray:
        m = cls(seed=0).fit(log.until(T0), Catalog(n_items=log.n_items))
        return m.score(queries)

    assert_future_invariant(pipeline, base, T0)


@pytest.mark.parametrize("target", ["successive", "last", "random"])
def test_basket_targets_have_strictly_earlier_history(target: str) -> None:
    log = random_log(seed=4)
    ev = next_basket(log, T0, T1, target=target, seed=0)
    q = ev.queries
    rows = np.repeat(np.arange(len(q)), q.lengths())
    assert np.all(q.ts < q.now[rows])
    # each target is an in-window order of that user, from a basket that started at query time
    for r in range(min(len(q), 50)):
        m = (log.user == q.user[r]) & (log.event == Event.ORDER) & (log.ts >= T0) & (log.ts < T1)
        first = (log.user == q.user[r]) & (log.ts == q.now[r]) & (log.event == Event.ORDER)
        sess = set(log.session[first].tolist())
        assert sess, "no order at query time"
        in_basket = m & np.isin(log.session, list(sess))
        assert set(ev.targets[r].indices) <= set(log.item[in_basket].tolist())
        # the basket has no events before the cutoff
        assert not np.any(
            (log.user == q.user[r]) & np.isin(log.session, list(sess)) & (log.ts < T0)
        )
    if target != "successive":
        assert np.unique(ev.unit).size == len(ev)


def test_successive_covers_every_window_order() -> None:
    log = random_log(seed=5)
    ev = next_basket(log, T0, T1, target="successive")
    win = log.between(T0, T1)
    m = (win.event == Event.ORDER) & (win.user >= 0)
    key = win.user[m].astype(np.int64) * 10_000 + win.session[m]
    past = log.until(T0)
    pkey = past.user[past.user >= 0].astype(np.int64) * 10_000 + past.session[past.user >= 0]
    n_baskets = np.unique(key[~np.isin(key, pkey)]).size
    assert len(ev) == n_baskets


def test_no_future_items_are_recommended() -> None:
    """Checklist #8: an item cannot be recommended before it first exists."""
    from bench.tasks import available_from

    log = random_log(seed=9)
    # make items 30..39 appear only late in the window
    late = log.item >= 30
    ts = np.where(late, np.maximum(log.ts, T1 - 3 * 86_400), log.ts)
    log = Log.build(log.user, log.item, ts, log.event, log.session, log.n_users, log.n_items)
    ev = next_item(log, T0, T1, target="successive")
    avail = available_from(log)
    m = RecencyPopularity().fit(log.until(T0), Catalog(n_items=log.n_items))
    R = m.recommend(ev.queries, 20, available_from=avail)
    first = avail[np.maximum(R, 0)]
    assert np.all((R < 0) | (first <= ev.queries.now[:, None]))


@pytest.mark.parametrize("cls", MODELS, ids=lambda c: c.name)
def test_scoring_never_mutates_the_model(cls) -> None:
    """Checklist #6: model state is frozen after fit; scoring test queries must not update it."""
    import pickle

    base = random_log()
    m = cls(seed=0).fit(base.until(T0), Catalog(n_items=base.n_items))
    before = pickle.dumps(m.__dict__)
    m.score(next_basket(base, T0, T1).queries)
    assert pickle.dumps(m.__dict__) == before


NEURAL = [  # small CPU configurations: CPU training is deterministic given the seed
    lambda: SASRecCE(d=16, n_blocks=1, max_len=8, epochs=2, threads=1, seed=0),
    lambda: GSASRec(d=16, n_blocks=1, max_len=8, epochs=2, negatives=4, threads=1, seed=0),
    lambda: GRU4Rec(layers=16, batch_size=16, n_sample=8, epochs=2, threads=1, seed=0),
]


@pytest.mark.network
@pytest.mark.parametrize("make", NEURAL, ids=lambda f: f().name)
def test_neural_predictions_invariant_to_future(make) -> None:
    base = random_log()
    queries = next_basket(base, T0, T1).queries

    def pipeline(log: Log) -> np.ndarray:
        return make().fit(log.until(T0), Catalog(n_items=log.n_items)).score(queries)

    assert_future_invariant(pipeline, base, T0)
