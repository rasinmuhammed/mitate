"""Every model fits on a tiny log, scores the right shape, and ranks an obvious signal."""

import numpy as np
import pytest

from mitate.data import Catalog, Event, Log, Queries
from mitate.models.baselines import EASE, GPTopFreq, ItemKNN, Popularity, RecencyPopularity, RP3beta
from mitate.models.mf import BPR, IALS


def toy_log(seed: int = 0) -> Log:
    """Two disjoint communities: users 0-49 use items 0-9, users 50-99 use items 10-19."""
    rng = np.random.default_rng(seed)
    u, i, t = [], [], []
    for user in range(100):
        block = 0 if user < 50 else 10
        for k, item in enumerate(rng.choice(10, 6, replace=False) + block):
            u.append(user)
            i.append(item)
            t.append(1_000 + user * 10 + k)
    n = len(u)
    return Log.build(u, i, t, np.full(n, Event.ORDER), np.array(u), 100, 20)


MODELS = [
    Popularity(),
    RecencyPopularity(),
    GPTopFreq(),
    ItemKNN(neighbours=10),
    RP3beta(neighbours=10),
    EASE(lam=1.0),
    IALS(factors=2, regularization=1.0, iterations=10),
    BPR(factors=8),
]


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.name)
def test_fit_score_shape(model) -> None:
    log = toy_log()
    model.fit(log, Catalog(n_items=20))
    q = Queries.from_lists([[0, 1], [12, 13], []])
    S = model.score(q)
    assert S.shape == (3, 20) and S.dtype == np.float32


@pytest.mark.parametrize(
    "model",
    [m for m in MODELS if m.name not in {"pop", "recpop", "gp_topfreq"}],
    ids=lambda m: m.name,
)
def test_personalised_models_respect_communities(model) -> None:
    model.fit(toy_log(), Catalog(n_items=20))
    q = Queries.from_lists([[0, 1, 2], [12, 13, 14]])
    q.history  # noqa: B018
    R = model.recommend(q, 5, exclude=q.matrix(20, binary=True))
    assert (R[0] < 10).mean() >= 0.8, R
    assert (R[1] >= 10).mean() >= 0.8, R


from mitate.models.sequential import PCTM, MarkovChain, SeqRules, pair_counts, training_sequences  # noqa: E402


def chain_log() -> Log:
    """Anonymous sessions walking a cycle 0->1->...->19->0 from random starts."""
    rng = np.random.default_rng(0)
    u, i, t, s = [], [], [], []
    ts = 0
    for sess in range(300):
        start = rng.integers(20)
        for k in range(rng.integers(2, 8)):
            u.append(-1)
            i.append((start + k) % 20)
            t.append(ts)
            s.append(sess)
            ts += 1
    return Log.build(u, i, t, np.zeros(len(u)), s, 0, 20)


def test_pair_counts_distance_and_sequence_boundaries() -> None:
    seq = np.array([0, 0, 0, 1, 1])
    items = np.array([3, 4, 5, 6, 7])
    C = pair_counts(seq, items, 10, 2, "u").toarray()
    assert C[3, 4] == 1 and C[3, 5] == 1 and C[4, 5] == 1 and C[6, 7] == 1
    assert C[5, 6] == 0  # no transition across sequences
    C2 = pair_counts(seq, items, 10, 2, "div").toarray()
    assert C2[3, 5] == 0.5


def test_training_sequences_collapses_repeats() -> None:
    log = Log.build([0, 0, 0, -1], [1, 1, 2, 1], [1, 2, 3, 4], [0, 0, 0, 0], [9, 9, 9, 5], 1, 3)
    seq, items = training_sequences(log)
    assert items.tolist() == [1, 1, 2] or items.tolist() == [1, 1, 2][::1]
    assert (seq < 0).sum() == 1  # the anonymous session is its own sequence


from mitate.models.sknn import VSTAN  # noqa: E402


@pytest.mark.parametrize(
    "model",
    [
        MarkovChain(),
        PCTM(window=3),
        SeqRules(window=3),
        VSTAN(),
        ItemKNN(neighbours=5),
        RP3beta(neighbours=5),
    ],
    ids=lambda m: m.name,
)
def test_sequence_models_predict_next_in_chain(model) -> None:
    log = chain_log()
    model.fit(log, Catalog(n_items=20))
    now = int(log.ts.max()) + 1
    q = Queries.from_lists([[3, 4, 5], [17, 18]], now=[now, now])
    R = model.recommend(q, 1, exclude=q.matrix(20, binary=True))
    if model.name in {"markov", "pctm", "seqrules", "vstan"}:
        assert R[:, 0].tolist() == [6, 19]
    else:  # order-agnostic co-occurrence still lands in the neighbourhood
        assert abs(int(R[0, 0]) - 4) <= 3 and abs(int(R[1, 0]) - 17) <= 3


from mitate.models.sasrec import SASRec  # noqa: E402


@pytest.mark.parametrize("loss,blocks", [("ce", "std"), ("gbce", "std"), ("ssm", "ligr")])
def test_sasrec_learns_chain(loss: str, blocks: str) -> None:
    log = chain_log()
    m = SASRec(
        loss=loss,
        blocks=blocks,
        d=32,
        n_blocks=1,
        max_len=8,
        epochs=40,
        negatives=8,
        lr=5e-3,
        dropout=0.0,
        threads=1,
    ).fit(log, Catalog(n_items=20))
    now = int(log.ts.max()) + 1
    q = Queries.from_lists([[3, 4, 5], [17, 18], []], now=[now] * 3)
    S = m.score(q)
    assert np.isfinite(S).all()
    R = m.recommend(q, 1, exclude=q.matrix(20, binary=True))
    assert R[:2, 0].tolist() == [6, 19]


from bench.gru4rec import GRU4Rec  # noqa: E402


@pytest.mark.network
@pytest.mark.parametrize("loss", ["cross-entropy", "bpr-max"])
def test_official_gru4rec_learns_chain(loss: str) -> None:
    """The pinned official GRU4Rec code, driven by the benchmark adapter, learns a cycle."""
    log = chain_log()
    m = GRU4Rec(
        loss=loss, layers=32, batch_size=16, learning_rate=0.1, n_sample=8, epochs=30, threads=1
    ).fit(log, Catalog(n_items=20))
    now = int(log.ts.max()) + 1
    q = Queries.from_lists([[3, 4, 5], [17, 18], []], now=[now] * 3)
    S = m.score(q)
    assert S.shape == (3, 20) and S.dtype == np.float32
    assert S[:2].argmax(axis=1).tolist() == [6, 19]
    assert not S[2].any()  # no history: no opinion


@pytest.mark.network
def test_official_gru4rec_batch_larger_than_the_store() -> None:
    """A tuned batch size above the number of training sessions trains on every session."""
    log = chain_log()
    m = GRU4Rec(layers=16, batch_size=512, n_sample=8, epochs=2, threads=1).fit(
        log, Catalog(n_items=20)
    )
    q = Queries.from_lists([[3, 4, 5]], now=[int(log.ts.max()) + 1])
    assert np.isfinite(m.score(q)).all()
