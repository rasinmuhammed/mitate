import numpy as np
import pytest

from mitate import guard
from mitate.blend import DAY, Blend
from mitate.constraints import Placement
from mitate.data import Catalog, Event, Log
from mitate.engine import Engine, EngineConfig, Shopper
from mitate.models.baselines import RecencyPopularity
from mitate.models.sequential import PCTM

N = 20


def cycle_log(days: int = 30, events: tuple[Event, ...] = (Event.VIEW,)) -> Log:
    """Anonymous sessions walking the cycle 0 -> 1 -> ... -> 19 -> 0, spread over ``days``."""
    rng = np.random.default_rng(0)
    u, i, t, e, s = [], [], [], [], []
    for sess in range(600):
        start, t0 = int(rng.integers(N)), int(rng.integers(days * DAY))
        for k in range(int(rng.integers(2, 8))):
            u.append(-1)
            i.append((start + k) % N)
            t.append(t0 + 60 * k)
            e.append(int(rng.choice(events)))
            s.append(sess)
    return Log.build(u, i, t, e, s, 0, N)


class Noise:
    """A ranker that has learned nothing about this store."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.random.default_rng(1).random(X.shape[0])


def test_choose_takes_the_best_mean_and_gives_ties_to_the_blend() -> None:
    a, b = np.array([0.2, 0.4]), np.array([0.4, 0.2])
    assert guard.choose({guard.BLEND: a, "recpop": b}) == guard.BLEND
    assert guard.choose({guard.BLEND: a, "recpop": b + 0.1}) == "recpop"


def test_the_replay_task_follows_what_the_log_records() -> None:
    assert guard.holdout_task(cycle_log()) == "next_item"
    assert guard.holdout_task(cycle_log(events=(Event.ORDER,))) == "next_basket"


def test_a_ranker_that_ranks_this_store_badly_is_replaced() -> None:
    log = cycle_log()
    blend = Blend(sources=[RecencyPopularity(), PCTM()], n_cand=10, booster=Noise())
    report = guard.run(blend, log, Catalog(N))
    assert report is not None and report.choice == "pctm"
    assert report.ndcg["pctm"] > report.ndcg[guard.BLEND] and report.n_shoppers > 0


def test_nothing_to_replay_means_no_report() -> None:
    log = cycle_log(days=3)  # the whole log falls inside the replay week
    assert guard.run(Blend(sources=[RecencyPopularity()]), log, Catalog(N)) is None


def test_the_engine_serves_the_source_the_guard_chose(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guard, "choose", lambda scores: "recpop")
    cfg = EngineConfig(min_behaviour_events=100, sources=("recpop", "pctm"), embedder="none")
    eng = Engine(cfg)
    info = eng.fit(cycle_log(), Catalog(N))
    assert info.guard is not None and info.guard.choice == "recpop"
    assert isinstance(eng.ranking, RecencyPopularity)
    sh = Shopper(items=[3], ts=[0], events=[int(Event.VIEW)], now=40 * DAY)
    got = [r.item for r in eng.recommend(Placement.NEXT, 5, shopper=sh)]
    S = eng.ranking.score(sh.queries())[0]
    assert got == [int(x) for x in np.argsort(-S, kind="stable")[:5]]


def test_the_guard_can_be_turned_off() -> None:
    cfg = EngineConfig(min_behaviour_events=100, sources=("recpop", "pctm"), guard_days=0)
    info = Engine(cfg).fit(cycle_log(), Catalog(N))
    assert info.guard is None and info.stage == "behavioural"
