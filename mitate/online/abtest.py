"""Holdout / A-B analysis with CUPED and an honest power calculator.

Randomisation is per visitor, so revenue per visitor (RPV) and conversion are per-visitor means and
need no delta method. CUPED (Deng, Xu, Kohavi & Walker, WSDM 2013) subtracts theta * (X - mean X)
from the outcome, with X the visitor's pre-period value of the same metric and
theta = cov(Y, X) / var(X) estimated on the pooled sample. Variance shrinks by a factor (1 - rho^2)
and the estimate stays unbiased because X is independent of assignment.

Only purchase outcomes (revenue, orders) are reported as lift. Attributed clicks on widgets are a
diagnostic, never an effect estimate: a recommender can shift clicks without changing sales.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sps

Arr = NDArray[np.float64]


@dataclass(frozen=True)
class Effect:
    metric: str
    control_mean: float
    treatment_mean: float
    abs_effect: float  # treatment - control, per visitor
    ci_low: float
    ci_high: float
    rel_effect: float  # abs_effect / control_mean
    rel_ci_low: float
    rel_ci_high: float
    p_value: float
    n_control: int
    n_treatment: int
    cuped: bool
    variance_reduction: float  # 1 - var_cuped / var_raw (0 when CUPED is off)
    skewness: float  # largest absolute sample skewness of the two arms' outcomes
    clt_ok: bool  # Kohavi et al. (KDD 2014): each arm has n >= 355 * skewness^2


def skewness(y: Arr) -> float:
    """Sample skewness (biased moment estimator; 0 for constant samples)."""
    sd = float(y.std())
    return float(np.mean((y - y.mean()) ** 3) / sd**3) if sd > 0 else 0.0


def fieller(mt: float, mc: float, vt: float, vc: float, tq: float) -> tuple[float, float]:
    """Fieller's interval for mu_t / mu_c from independent arm means and their variances.

    It contains r exactly when |mt - r * mc| <= tq * sqrt(vt + r^2 vc), so it excludes 1 exactly
    when the difference test rejects. Unbounded when mc is not distinguishable from zero.
    """
    a = mc**2 - tq**2 * vc
    if a <= 0:
        return -math.inf, math.inf
    d = tq**2 * (vt * a + vc * mt**2)
    return (mt * mc - math.sqrt(d)) / a, (mt * mc + math.sqrt(d)) / a


def cuped_adjust(y: Arr, x: Arr) -> tuple[Arr, float]:
    """CUPED-adjusted outcome and the achieved variance reduction."""
    vx = float(np.var(x, ddof=1))
    if vx <= 0 or y.size < 3:
        return y, 0.0
    theta = float(np.cov(y, x, ddof=1)[0, 1] / vx)
    adj = y - theta * (x - x.mean())
    vr = 1.0 - float(np.var(adj, ddof=1) / max(np.var(y, ddof=1), 1e-300))
    return adj, vr


def estimate(
    metric: str,
    y_control: Arr | Sequence[float],
    y_treatment: Arr | Sequence[float],
    x_control: Arr | Sequence[float] | None = None,
    x_treatment: Arr | Sequence[float] | None = None,
    alpha: float = 0.05,
) -> Effect:
    """Difference in means with a Welch CI and p-value; relative CI by Fieller's theorem, with
    the same critical value, so the two intervals always agree on significance."""
    yc, yt = np.asarray(y_control, float), np.asarray(y_treatment, float)
    vr = 0.0
    use_cuped = x_control is not None and x_treatment is not None
    if use_cuped:
        assert x_control is not None and x_treatment is not None
        y = np.concatenate([yc, yt])
        x = np.concatenate([np.asarray(x_control, float), np.asarray(x_treatment, float)])
        adj, vr = cuped_adjust(y, x)
        yc, yt = adj[: yc.size], adj[yc.size :]
    mc, mt = float(yc.mean()), float(yt.mean())
    vc, vt = float(yc.var(ddof=1)) / yc.size, float(yt.var(ddof=1)) / yt.size
    se = math.sqrt(vc + vt)
    dof = (vc + vt) ** 2 / (vc**2 / (yc.size - 1) + vt**2 / (yt.size - 1)) if se > 0 else 1.0
    tq = float(sps.t.ppf(1 - alpha / 2, dof))
    diff = mt - mc
    p = float(2 * sps.t.sf(abs(diff) / se, dof)) if se > 0 else 1.0
    # Relative effect uses the (adjusted) control mean: CUPED leaves arm means unbiased, and using
    # the same quantities as the SE keeps the delta-method interval coherent.
    base = mc
    # mt / mc overestimates mu_t / mu_c in small samples (E[1/mc] > 1/E[mc], strongest for rare,
    # heavy-tailed revenue). Second-order delta-method correction for independent arms
    # (Deng, Knoblich & Lu, KDD 2018): E[mt/mc] ~ (mu_t/mu_c)(1 + Var(mc)/mu_c^2).
    rel = (mt / mc) / (1.0 + vc / mc**2) - 1.0 if base else float("nan")
    lo, hi = fieller(mt, mc, vt, vc, tq) if base else (float("nan"), float("nan"))
    skew = max(abs(skewness(yc)), abs(skewness(yt)))
    return Effect(
        metric=metric,
        control_mean=float(np.mean(y_control)),
        treatment_mean=float(np.mean(y_treatment)),
        abs_effect=diff,
        ci_low=diff - tq * se,
        ci_high=diff + tq * se,
        rel_effect=rel,
        rel_ci_low=lo - 1.0,
        rel_ci_high=hi - 1.0,
        p_value=p,
        n_control=int(yc.size),
        n_treatment=int(yt.size),
        cuped=use_cuped,
        variance_reduction=vr,
        skewness=skew,
        clt_ok=min(yc.size, yt.size) >= 355.0 * skew**2,
    )


# --------------------------------------------------------------------------- power


@dataclass(frozen=True)
class PowerPlan:
    visitors_needed_total: int
    days_needed: float
    conclusive_within_horizon: bool
    mde_at_horizon: float  # relative MDE reachable with the traffic available in the horizon
    assumptions: str


def required_visitors(
    mean: float,
    std: float,
    rel_mde: float,
    treat_share: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.8,
    variance_reduction: float = 0.0,
) -> int:
    """Total visitors for a two-sided test of a relative lift ``rel_mde`` in a per-visitor mean.

    n_total = (z_{1-a/2} + z_{power})^2 * sigma^2 (1 - VR) / (p (1-p) * delta^2), with delta the
    absolute MDE and p the treatment share.
    """
    z: float = sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power)
    var = std**2 * (1.0 - variance_reduction)
    delta = rel_mde * mean
    if delta <= 0:
        raise ValueError("MDE must be positive")
    p = treat_share
    return math.ceil(z**2 * var / (p * (1 - p) * delta**2))


def plan(
    daily_visitors: float,
    mean: float,
    std: float,
    rel_mde: float,
    horizon_days: float = 28,
    treat_share: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.8,
    variance_reduction: float = 0.0,
) -> PowerPlan:
    """Can this store detect ``rel_mde`` within ``horizon_days``? And what can it detect?"""
    n = required_visitors(mean, std, rel_mde, treat_share, alpha, power, variance_reduction)
    days = n / daily_visitors if daily_visitors > 0 else math.inf
    z = sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power)
    n_h = daily_visitors * horizon_days
    var = std**2 * (1 - variance_reduction)
    p = treat_share
    mde_h = z * math.sqrt(var / (p * (1 - p) * n_h)) / mean if n_h > 0 and mean else math.inf
    return PowerPlan(
        visitors_needed_total=n,
        days_needed=days,
        conclusive_within_horizon=days <= horizon_days,
        mde_at_horizon=mde_h,
        assumptions=(
            "Normal approximation on per-visitor means; std should come from the store's own "
            "visitor-level revenue (heavy tails inflate it). Use simulate_power() to check."
        ),
    )


def simulate_power(
    pre_period_values: Arr,
    rel_lift: float,
    n_total: int,
    treat_share: float = 0.5,
    alpha: float = 0.05,
    n_sims: int = 500,
    seed: int = 0,
) -> float:
    """Empirical power by resampling the store's own visitor-level outcomes and applying a
    multiplicative lift to treatment. Honest under heavy tails, where the normal formula is not."""
    rng = np.random.default_rng(seed)
    v = np.asarray(pre_period_values, float)
    n_t = int(n_total * treat_share)
    n_c = n_total - n_t
    hits = 0
    for _ in range(n_sims):
        yc = rng.choice(v, n_c, replace=True)
        yt = rng.choice(v, n_t, replace=True) * (1 + rel_lift)
        _, p = sps.ttest_ind(yt, yc, equal_var=False)
        hits += p < alpha
    return hits / n_sims
