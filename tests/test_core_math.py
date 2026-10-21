import numpy as np
import pytest
import scipy.sparse as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from mitate.linalg import spd_inverse_inplace
from mitate.metrics import gini_exposure, hit_matrix, ranking_metrics, split_repeat_explore
from mitate.models.ease import ease_weights
from mitate.stats import benjamini_yekutieli, holm, paired_test, per_user_mean


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n", [1, 7, 300])
def test_spd_inverse_matches_numpy(n: int, dtype: type) -> None:
    rng = np.random.default_rng(n)
    A = rng.standard_normal((n + 5, n))
    G = (A.T @ A + n * np.eye(n)).astype(dtype)
    ref = np.linalg.inv(G.astype(np.float64))
    P = spd_inverse_inplace(G.copy(), block=64)
    tol = 1e-4 if dtype == np.float32 else 1e-10
    np.testing.assert_allclose(P, ref, rtol=tol, atol=tol)
    np.testing.assert_array_equal(P, P.T)


def test_ease_matches_direct_formula() -> None:
    rng = np.random.default_rng(0)
    X = sp.random(200, 50, density=0.1, format="csr", random_state=1)
    X.data[:] = 1.0
    lam = 3.0
    G = (X.T @ X).toarray() + lam * np.eye(50)
    P = np.linalg.inv(G)
    ref = -P / np.diag(P)[None, :]
    np.fill_diagonal(ref, 0.0)
    np.testing.assert_allclose(ease_weights(X, lam), ref, atol=1e-10)
    assert rng is not None


def test_ranking_metrics_hand_computed() -> None:
    # query 0: targets {2, 5}; recs 5,1,2 -> hits at ranks 1 and 3
    # query 1: targets {0};    recs 3,4,9 -> no hit
    # query 2: no targets      -> NaN
    targets = sp.csr_matrix((np.ones(3), ([0, 0, 1], [2, 5, 0])), shape=(3, 10))
    recs = np.array([[5, 1, 2], [3, 4, 9], [0, 1, 2]])
    m = ranking_metrics(recs, targets, ks=[1, 3])
    d3, d1 = m[3], m[1]
    idcg = 1 + 1 / np.log2(3)
    assert d3.ndcg[0] == pytest.approx((1 + 1 / np.log2(4)) / idcg)
    assert d3.recall[0] == pytest.approx(1.0)
    assert d1.recall[0] == pytest.approx(0.5)
    assert d1.recall_capped[0] == pytest.approx(1.0)
    assert d3.mrr[0] == 1.0 and d3.mrr[1] == 0.0
    assert d3.hit[1] == 0.0 and np.isnan(d3.hit[2])


@settings(max_examples=50, deadline=None)
@given(st.integers(1, 30), st.integers(5, 40), st.integers(0, 10_000))
def test_hit_matrix_matches_bruteforce(n_q: int, n_items: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    T = sp.random(n_q, n_items, density=0.2, format="csr", random_state=seed)
    k = min(5, n_items)
    recs = np.stack([rng.permutation(n_items)[:k] for _ in range(n_q)])
    recs[rng.random(recs.shape) < 0.1] = -1
    got = hit_matrix(recs, T)
    dense = T.toarray() > 0
    want = np.array([[r >= 0 and dense[q, r] for r in row] for q, row in enumerate(recs)])
    np.testing.assert_array_equal(got, want)


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 10_000))
def test_metrics_bounded_and_monotone_recall(seed: int) -> None:
    rng = np.random.default_rng(seed)
    T = sp.random(20, 30, density=0.15, format="csr", random_state=seed)
    recs = np.stack([rng.permutation(30)[:20] for _ in range(20)])
    m = ranking_metrics(recs, T, ks=[5, 10, 20])
    for k in (5, 10, 20):
        for v in (m[k].ndcg, m[k].recall, m[k].mrr, m[k].hit):
            v = v[~np.isnan(v)]
            assert np.all((v >= 0) & (v <= 1 + 1e-12))
    r5, r20 = m[5].recall, m[20].recall
    ok = ~np.isnan(r5)
    assert np.all(r20[ok] >= r5[ok])


def test_repeat_explore_partition() -> None:
    T = sp.csr_matrix(np.array([[1, 1, 0, 1], [0, 1, 1, 0]]))
    H = sp.csr_matrix(np.array([[1, 0, 0, 1], [0, 0, 0, 0]]))
    rep, exp = split_repeat_explore(T, H)
    np.testing.assert_array_equal(rep.toarray(), [[1, 0, 0, 1], [0, 0, 0, 0]])
    np.testing.assert_array_equal(exp.toarray(), [[0, 1, 0, 0], [0, 1, 1, 0]])


def test_gini_extremes() -> None:
    assert gini_exposure(np.arange(10)[None, :], 10, 10) == pytest.approx(0.0)
    assert gini_exposure(np.zeros((5, 1), dtype=int), 10, 1) == pytest.approx(0.9)


def test_holm_known_values() -> None:
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.isnan(holm([np.nan, 0.5])[0])


def test_paired_test_detects_shift_and_null() -> None:
    rng = np.random.default_rng(0)
    a = rng.random(500)
    r = paired_test(a + 0.05 + rng.normal(0, 0.01, 500), a, n_boot=500, n_perm=500)
    assert r.diff == pytest.approx(0.05, abs=0.002)
    assert r.p_ttest < 1e-10 and r.p_perm < 0.01 and r.ci_low > 0 and r.d_z > 1
    r0 = paired_test(a, a, n_boot=200, n_perm=200)
    assert r0.p_ttest == 1.0 and r0.diff == 0.0


def test_paired_test_type1_error_is_nominal() -> None:
    """Under a true null (zero-mean skewed differences) the t-test rejects ~5% of the time."""
    rng = np.random.default_rng(1)
    rej = sum(
        paired_test(rng.exponential(1.0, 80) - 1.0, np.zeros(80), n_boot=50, n_perm=50).p_ttest
        < 0.05
        for _ in range(400)
    )
    assert 0.025 < rej / 400 < 0.085


def test_benjamini_yekutieli_matches_reference() -> None:
    # m=3, c(3)=1.8333; sorted p = .01,.02,.04 -> .055,.055,.0733
    got = benjamini_yekutieli([0.04, 0.01, 0.02])
    np.testing.assert_allclose(got, [0.0733333, 0.055, 0.055], rtol=1e-5)


def test_per_user_mean() -> None:
    u, m = per_user_mean(np.array([1.0, 0.0, np.nan, 0.5]), np.array([3, 3, 4, 5]))
    assert u.tolist() == [3, 5] and m.tolist() == [0.5, 0.5]
