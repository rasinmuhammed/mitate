import random

import numpy as np
import pytest

from mitate.online.abtest import estimate, plan, required_visitors, simulate_power
from mitate.online.assign import assign, unit_hash
from mitate.online.interleave import analyse, team_draft
from mitate.online.simulate import ab_validation, interleaving_validation


def test_assignment_is_deterministic_and_balanced() -> None:
    arms = [("holdout", 0.1), ("mitate", 0.9)]
    got = [assign("exp1", f"v{i}", arms) for i in range(20_000)]
    assert got == [assign("exp1", f"v{i}", arms) for i in range(20_000)]
    share = np.mean([g == "holdout" for g in got])
    assert abs(share - 0.1) < 0.01
    assert 0 <= unit_hash("s", "x") < 1


def test_team_draft_properties() -> None:
    rng = random.Random(0)
    a, b = list(range(10)), list(range(9, -1, -1))
    il = team_draft(a, b, 10, rng)
    assert sorted(il.items) == list(range(10))  # no duplicates, everything placed
    na, nb = il.team.count(0), il.team.count(1)
    assert abs(na - nb) <= 1
    # identical rankers must give identical lists and balanced credit in expectation
    il2 = team_draft(a, a, 6, rng)
    assert il2.items == a[:6]


def test_power_formula_matches_textbook() -> None:
    # sigma=1, mean=1, rel MDE=0.1 -> delta=0.1; 50/50 => n = 4 * 7.849 / 0.01 = 3140
    n = required_visitors(1.0, 1.0, 0.1)
    assert n == pytest.approx(3140, abs=2)
    p = plan(daily_visitors=100, mean=1.0, std=1.0, rel_mde=0.1)
    assert not p.conclusive_within_horizon and p.days_needed == pytest.approx(n / 100)


def test_simulated_power_close_to_formula_for_gaussian() -> None:
    rng = np.random.default_rng(0)
    v = rng.normal(10, 2, 50_000)
    n = required_visitors(10, 2, 0.02)
    pw = simulate_power(v, 0.02, n, n_sims=300)
    assert 0.72 < pw < 0.88


@pytest.mark.parametrize("regime", ["rare_buyers", "repeat_grocery"])
def test_estimate_recovers_known_lift(regime: str) -> None:
    res = ab_validation(n=20_000, lift=0.10, sims=80, seed=1, regime=regime)
    for arm in ("raw", "cuped"):
        se = res[arm]["sd_estimate"] / np.sqrt(80)
        assert abs(res[arm]["bias"]) < 4 * se + 0.005
        assert res[arm]["ci95_coverage"] >= 0.85
    if regime == "repeat_grocery":
        assert res["cuped"]["mean_variance_reduction"] > 0.15
        assert res["cuped"]["sd_estimate"] < res["raw"]["sd_estimate"]


def test_effect_never_uses_clicks_and_reports_counts() -> None:
    e = estimate("rpv", np.array([0.0, 10, 0, 5]), np.array([0.0, 12, 0, 6, 1]))
    assert e.n_control == 4 and e.n_treatment == 5 and e.ci_low < e.abs_effect < e.ci_high


def test_interleaving_delta_ignores_unengaged_impressions() -> None:
    # 3 engaged impressions (A, A, B) plus 97 without any credited event
    vis = list(range(100))
    ca = [1, 1, 0] + [0] * 97
    cb = [0, 0, 1] + [0] * 97
    r = analyse(vis, ca, cb, winsor_q=None)
    assert r.n_impressions == 100 and r.n_engaged == 3 and r.n_visitors == 3
    # Chapelle: (wins_A + ties/2) / N - 1/2 = 2/3 - 1/2
    assert r.delta == pytest.approx(2 / 3 - 0.5)


def test_interleaving_se_is_clustered_by_visitor() -> None:
    rng = np.random.default_rng(0)
    # 200 visitors with a fixed personal preference, 20 impressions each: impressions are not
    # independent, so the cluster SE must be far larger than the naive per-impression SE.
    pref = rng.choice([-1, 1], 200)
    vis = np.repeat(np.arange(200), 20)
    out = np.repeat(pref, 20)
    r = analyse(vis, (out > 0).astype(int), (out < 0).astype(int), winsor_q=None)
    naive = 0.5 * out.std(ddof=1) / np.sqrt(out.size)
    assert r.se > 3 * naive


def test_interleaving_winsorises_heavy_visitors() -> None:
    vis = [0] * 5000 + list(range(1, 1001))
    ca = [1] * 5000 + [1, 0] * 500
    cb = [0] * 5000 + [0, 1] * 500
    raw = analyse(vis, ca, cb, winsor_q=None)
    capped = analyse(vis, ca, cb, winsor_q=0.99)
    assert raw.delta > 0.4  # one bot dominates
    assert capped.winsorised == 1 and abs(capped.delta) < 0.01


def test_interleaving_validation_detects_better_ranker_and_holds_size() -> None:
    res = interleaving_validation(n_visitors=1000, quality_gap=0.6, sims=30, seed=3)
    assert res["power_detect_better_A"] > 0.8
    assert res["wrong_direction_significant"] == 0.0
    # 30 null runs: P(>= 5 false positives | size 0.05) < 0.02
    assert res["null_false_positive_rate"] <= 4 / 30


def test_rare_heavy_tailed_buyers() -> None:
    """Few buyers with lognormal order values: the ratio estimate stays unbiased, the Fieller
    interval covers, the test keeps its nominal error rate, and the CLT rule refuses to call
    such a small, skewed test reliable."""
    res = ab_validation(n=5_000, lift=0.10, sims=400, seed=2, regime="rare_buyers")
    raw = res["raw"]
    se = raw["sd_estimate"] / np.sqrt(400)
    assert abs(raw["bias"]) < 3 * se
    assert raw["ci95_coverage"] >= 0.92
    null = res["null_false_positive_rate"]
    assert null["rate"] <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 400)
    assert null["share_reported_reliable"] < 0.05
