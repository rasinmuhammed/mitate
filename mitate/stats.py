"""Paired significance testing for per-user metric vectors.

Every comparison is paired over the same users (per-user metrics are averaged over a user's
targets first, so users are the independent unit). For each pair we report the mean difference,
the relative change, a percentile paired-bootstrap 95% CI, the paired effect size
d_z = mean(d) / sd(d) (Sakai, SIGIR 2016), a two-sided paired t-test p-value (primary), and a
two-sided sign-flip randomisation p-value (robustness check).

Wilcoxon and sign tests are deliberately absent: they are unreliable for hypotheses about mean
effectiveness at RecSys sample sizes (Urbano et al., SIGIR 2019; Ihemelandu & Ekstrand, SIGIR
2023). See docs/research_ledger.md.

Families of comparisons are adjusted with Holm (FWER; confirmatory "does Mitate beat X") or
Benjamini-Yekutieli (FDR under dependence; exploratory leaderboards).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sps

FloatArr = NDArray[np.float64]


@dataclass(frozen=True)
class PairedResult:
    n: int
    mean_a: float
    mean_b: float
    diff: float  # mean(a - b)
    ci_low: float
    ci_high: float
    d_z: float
    rel_change: float  # mean(d) / mean(b)
    p_ttest: float
    p_perm: float
    p_adj: float = float("nan")

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _clean(a: FloatArr, b: FloatArr) -> tuple[FloatArr, FloatArr]:
    if a.shape != b.shape:
        raise ValueError("paired vectors must have the same shape")
    keep = ~(np.isnan(a) | np.isnan(b))
    return a[keep], b[keep]


def paired_test(
    a: FloatArr,
    b: FloatArr,
    *,
    n_boot: int = 10_000,
    n_perm: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedResult:
    """Compare system ``a`` against reference ``b`` on paired per-user values."""
    a, b = _clean(np.asarray(a, float), np.asarray(b, float))
    n = a.size
    d = a - b
    nan = float("nan")
    if n == 0:
        return PairedResult(0, nan, nan, nan, nan, nan, nan, nan, nan, nan)
    rng = np.random.default_rng(seed)
    mean_d = float(d.mean())
    boots = _resample_means(d, n_boot, rng, flip=False)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    flips = _resample_means(d, n_perm, rng, flip=True)
    p_perm = float((1 + np.sum(np.abs(flips) >= abs(mean_d) - 1e-15)) / (n_perm + 1))
    sd = float(d.std(ddof=1)) if n > 1 else nan
    if n > 1 and sd > 0:
        d_z = mean_d / sd
        p_t = float(2 * sps.t.sf(abs(mean_d) / (sd / np.sqrt(n)), n - 1))
    else:
        d_z, p_t = 0.0, (1.0 if mean_d == 0 else 0.0)
    mb = float(b.mean())
    return PairedResult(
        n=n,
        mean_a=float(a.mean()),
        mean_b=mb,
        diff=mean_d,
        ci_low=float(lo),
        ci_high=float(hi),
        d_z=float(d_z),
        rel_change=mean_d / mb if mb else nan,
        p_ttest=p_t,
        p_perm=p_perm,
    )


def _resample_means(d: FloatArr, reps: int, rng: np.random.Generator, flip: bool) -> FloatArr:
    """Bootstrap means (resample with replacement) or sign-flip means, chunked to bound memory."""
    n = d.size
    out = np.empty(reps)
    chunk = max(1, 4_000_000 // max(n, 1))
    for s in range(0, reps, chunk):
        e = min(reps, s + chunk)
        if flip:
            signs = rng.integers(0, 2, size=(e - s, n), dtype=np.int8) * 2 - 1
            out[s:e] = (signs * d).mean(axis=1)
        else:
            out[s:e] = d[rng.integers(0, n, size=(e - s, n))].mean(axis=1)
    return out


def holm(pvalues: Sequence[float]) -> FloatArr:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1). NaNs pass through."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = pv.size
    if m == 0:
        return out
    order = np.argsort(pv)
    adj = np.maximum.accumulate((m - np.arange(m)) * pv[order])
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def benjamini_yekutieli(pvalues: Sequence[float]) -> FloatArr:
    """BY step-up adjusted p-values (FDR under arbitrary dependence). NaNs pass through."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = pv.size
    if m == 0:
        return out
    c = np.sum(1.0 / np.arange(1, m + 1))
    order = np.argsort(pv)
    ranked = pv[order] * m * c / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def per_user_mean(
    values: FloatArr, user: NDArray[np.integer]
) -> tuple[NDArray[np.integer], FloatArr]:
    """Average a per-target metric within user (Successive targets), NaNs ignored."""
    v = np.asarray(values, float)
    ok = ~np.isnan(v)
    u, inv = np.unique(user[ok], return_inverse=True)
    sums = np.bincount(inv, weights=v[ok])
    cnt = np.bincount(inv)
    return u, sums / cnt


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def t_interval(values: Sequence[float], alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean and Student-t CI across independent replicates (e.g. simulated stores)."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if v.size < 2:
        m = float(v.mean()) if v.size else float("nan")
        return m, m, m
    m = float(v.mean())
    half = float(sps.t.ppf(1 - alpha / 2, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size))
    return m, m - half, m + half
