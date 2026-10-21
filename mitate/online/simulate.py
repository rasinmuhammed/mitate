"""Known-truth simulations that validate the online measurement module.

Visitors have a latent spend propensity; pre-period and experiment-period revenue share it, so
CUPED has something to exploit. Conversion is Bernoulli and order value lognormal, giving the heavy
right tail typical of small-store revenue. The treatment multiplies conversion probability by
(1 + lift), so the true relative RPV effect equals ``lift`` exactly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from mitate.online.abtest import estimate
from mitate.online.interleave import analyse, team_draft


@dataclass(frozen=True)
class Population:
    pre_revenue: NDArray[np.float64]
    base_conv: NDArray[np.float64]
    order_value: NDArray[np.float64]


REGIMES = {
    # (per-visit conversion rate, visits per period): rare one-off buyers vs repeat grocery shoppers
    "rare_buyers": (0.03, 1),
    "repeat_grocery": (0.25, 4),
}


def population(
    n: int, rng: np.random.Generator, base_rate: float = 0.03, visits: int = 1
) -> Population:
    propensity = rng.lognormal(0.0, 1.0, n)
    p = np.clip(base_rate * propensity / propensity.mean(), 0, 0.9)
    value = rng.lognormal(np.log(40.0), 0.8, n)
    pre = rng.binomial(visits, p) * value * rng.lognormal(0, 0.3, n)
    return Population(pre, p, value)


class Draw(NamedTuple):
    rel: float  # estimated relative lift
    rel_ci_low: float
    rel_ci_high: float
    variance_reduction: float
    true_rel: float  # E[y | treat] / E[y | control] - 1 for this population
    p_value: float
    clt_ok: bool


def run_ab(
    n: int,
    lift: float,
    rng: np.random.Generator,
    cuped: bool,
    regime: str = "rare_buyers",
    holdout_share: float = 0.5,
) -> Draw:
    """One simulated holdout test. The true lift equals ``lift`` except where clipping the
    conversion probability at 1 binds."""
    rate, visits = REGIMES[regime]
    pop = population(n, rng, rate, visits)
    treat = rng.random(n) >= holdout_share
    p = np.where(treat, np.clip(pop.base_conv * (1 + lift), 0, 1), pop.base_conv)
    y = rng.binomial(visits, p) * pop.order_value
    e = estimate(
        "rpv",
        y[~treat],
        y[treat],
        pop.pre_revenue[~treat] if cuped else None,
        pop.pre_revenue[treat] if cuped else None,
    )
    p_t = np.clip(pop.base_conv * (1 + lift), 0, 1)
    true = float((p_t * pop.order_value).sum() / (pop.base_conv * pop.order_value).sum() - 1)
    return Draw(
        e.rel_effect, e.rel_ci_low, e.rel_ci_high, e.variance_reduction, true, e.p_value, e.clt_ok
    )


def ab_validation(
    n: int = 20_000,
    lift: float = 0.10,
    sims: int = 400,
    seed: int = 0,
    regime: str = "rare_buyers",
) -> dict[str, dict[str, float]]:
    """Bias and coverage are measured against each draw's exact population lift."""
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, float]] = {}
    for cuped in (False, True):
        est = np.array([run_ab(n, lift, rng, cuped, regime) for _ in range(sims)])
        truth = est[:, 4]
        cover = np.mean((est[:, 1] <= truth) & (truth <= est[:, 2]))
        out["cuped" if cuped else "raw"] = {
            "true_rel_lift": float(truth.mean()),
            "mean_estimate": float(est[:, 0].mean()),
            "bias": float((est[:, 0] - truth).mean()),
            "sd_estimate": float(est[:, 0].std(ddof=1)),
            "ci95_coverage": float(cover),
            "mean_variance_reduction": float(est[:, 3].mean()),
        }
    rng0 = np.random.default_rng(seed + 1)
    null = [run_ab(n, 0.0, rng0, True, regime) for _ in range(sims)]
    rejected = np.array([d.p_value < 0.05 for d in null])
    reliable = np.array([d.clt_ok for d in null])
    out["null_false_positive_rate"] = {
        "rate": float(rejected.mean()),
        "share_reported_reliable": float(reliable.mean()),
        "rate_when_reliable": float(rejected[reliable].mean()) if reliable.any() else float("nan"),
    }
    return out


def interleaving_validation(
    n_visitors: int = 2000,
    mean_impressions: float = 3.0,
    n_items: int = 200,
    k: int = 10,
    quality_gap: float = 0.3,
    bot_share: float = 0.01,
    sims: int = 100,
    seed: int = 0,
) -> dict[str, float]:
    """Cascade-click shoppers with heterogeneous engagement.

    Each visitor has their own item attractiveness and a clickiness multiplier (log-normal), a
    geometric number of impressions, and abandons the list with probability 0.1 per slot. Ranker A
    sees attractiveness through less noise than B, so A is truly better; under the null both see
    the same noise level. A `bot_share` of visitors produce 30x the impressions and click a
    uniformly random slot half the time: they carry no preference and stress the per-visitor
    weighting.
    """
    rng = np.random.default_rng(seed)

    def one(gap: float) -> tuple[float, float, float, bool]:
        r = random.Random(int(rng.integers(1 << 30)))
        vis: list[int] = []
        ca: list[int] = []
        cb: list[int] = []
        n_imp = rng.geometric(1.0 / mean_impressions, n_visitors)  # mean = mean_impressions
        bot = rng.random(n_visitors) < bot_share
        n_imp[bot] *= 30
        noise_b = 1.0
        noise_a = noise_b - gap
        for v in range(n_visitors):
            attract = np.minimum(rng.beta(0.5, 4.0, n_items) * rng.lognormal(0.0, 0.5), 1.0)
            sd = attract.std()
            for _ in range(int(n_imp[v])):
                ra = np.argsort(-(attract + rng.normal(0, noise_a * sd, n_items)))[:k].tolist()
                rb = np.argsort(-(attract + rng.normal(0, noise_b * sd, n_items)))[:k].tolist()
                il = team_draft(ra, rb, k, r)
                clicks = []
                if bot[v]:
                    if rng.random() < 0.5:
                        clicks.append(il.items[int(rng.integers(len(il.items)))])
                else:
                    for it in il.items:
                        if rng.random() < attract[cast(int, it)]:  # ids from argsort
                            clicks.append(it)
                            break
                        if rng.random() < 0.1:
                            break
                a, b = il.credit(clicks)
                vis.append(v)
                ca.append(a)
                cb.append(b)
        res = analyse(vis, ca, cb)
        return res.delta, res.p_value, float(res.ci_low <= 0 <= res.ci_high), res.clt_ok

    better = np.array([one(quality_gap) for _ in range(sims)])
    null = np.array([one(0.0) for _ in range(sims)])
    return {
        "power_detect_better_A": float(np.mean((better[:, 0] > 0) & (better[:, 1] < 0.05))),
        "wrong_direction_significant": float(np.mean((better[:, 0] < 0) & (better[:, 1] < 0.05))),
        "mean_delta_when_A_better": float(better[:, 0].mean()),
        "null_false_positive_rate": float(np.mean(null[:, 1] < 0.05)),
        "null_ci95_coverage": float(null[:, 2].mean()),
        "clt_rule_satisfied": float(np.concatenate([better[:, 3], null[:, 3]]).mean()),
    }
