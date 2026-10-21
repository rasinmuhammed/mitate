import numpy as np

from bench.tasks import next_item
from mitate.blend import Blend, Variant, ablate, ltr_groups
from mitate.data import Catalog
from mitate.ltr import train_isolated
from mitate.models.baselines import ItemKNN, RecencyPopularity
from mitate.models.sequential import PCTM
from tests.test_leakage import T0, T1, random_log


def _sources():
    return [RecencyPopularity(), ItemKNN(neighbours=10), PCTM(window=3)]


def test_source_ablation_is_exact() -> None:
    log = random_log(seed=7)
    q = next_item(log, T0, T1, target="last").queries
    cat = Catalog(n_items=log.n_items)
    full = Blend(sources=_sources(), n_cand=8).fit(log.until(T0), cat)
    without = Blend(sources=[s for s in _sources() if s.name != "itemknn"], n_cand=8)
    without.fit(log.until(T0), cat)
    fa = ablate(full.candidates(q), Variant("no_itemknn", drop_sources=("itemknn",)))
    fb = without.candidates(q)
    for r in range(len(q)):
        a = set(fa.cand[r][fa.cand[r] >= 0].tolist())
        b = set(fb.cand[r][fb.cand[r] >= 0].tolist())
        assert a == b
    # shared feature columns agree on the shared candidates
    shared = [n for n in fb.feature_names if n in fa.feature_names]
    ia = [fa.feature_names.index(n) for n in shared]
    ib = [fb.feature_names.index(n) for n in shared]
    for r in range(len(q)):
        for c in fb.cand[r][fb.cand[r] >= 0]:
            ra = fa.X[r, list(fa.cand[r]).index(c)][ia]
            rb = fb.X[r, list(fb.cand[r]).index(c)][ib]
            np.testing.assert_allclose(ra, rb, equal_nan=True)


def test_lambdarank_round_trip(tmp_path) -> None:
    log = random_log(seed=8, n_sess=1500)
    cat = Catalog(n_items=log.n_items)
    val = next_item(log, T0, T1, target="successive")
    b = Blend(sources=_sources(), n_cand=10).fit(log.until(T0), cat)
    frames = [b.candidates(val.queries)]
    X, y, g = ltr_groups(frames, val.targets)
    assert X.shape[0] == g.sum() and y.sum() >= len(g)
    b.booster = train_isolated(
        tmp_path / "job.npz", tmp_path / "m.json", X, y, g, list(frames[0].feature_names)
    )
    S = b.score(val.queries)
    assert S.shape == (len(val), log.n_items) and np.isfinite(S).all()


def test_variant_booster_serves_a_blend_without_the_source(tmp_path) -> None:
    """Serving a store below a source's threshold: a blend built without the source, scored by
    the booster trained on the matching ablation, equals the ablated full blend."""
    log = random_log(seed=9, n_sess=1500)
    cat = Catalog(n_items=log.n_items)
    val = next_item(log, T0, T1, target="successive")
    v = Variant("no_itemknn", drop_sources=("itemknn",))
    full = Blend(sources=_sources(), n_cand=10).fit(log.until(T0), cat)
    fr = full.candidates(val.queries)
    X, y, g = ltr_groups([fr], val.targets, v)
    booster = train_isolated(
        tmp_path / "job.npz", tmp_path / "m.json", X, y, g, list(fr.feature_names)
    )
    full.booster = booster
    want = full.scores_from_frame(ablate(fr, v), full.background())
    small = Blend(
        sources=[s for s in _sources() if s.name != "itemknn"], n_cand=10, booster=booster
    )
    got = small.fit(log.until(T0), cat).score(val.queries)
    np.testing.assert_allclose(got, want, rtol=1e-6)
