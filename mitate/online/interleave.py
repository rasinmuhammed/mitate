"""Team-draft interleaving (Radlinski, Kurup & Joachims, CIKM 2008).

Two rankers take turns, in a coin-flip order each round, adding their highest-ranked item not yet
shown. Every shown slot remembers which team placed it, so a click (or add-to-cart, or purchase)
is credited to exactly one ranker. One impression's outcome is a win for whichever team collected
more credited events, else a tie. The preference statistic is Chapelle et al.'s Delta_AB over
engaged impressions, tested with visitors as the independent unit.
"""

from __future__ import annotations

import random
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sps

A, B = 0, 1


@dataclass(frozen=True)
class Interleaved:
    items: list[Hashable]
    team: list[int]  # A or B for each slot

    def credit(self, engaged: Iterable[Hashable]) -> tuple[int, int]:
        """Credited events (A, B) for the engaged items on this impression."""
        pos = {it: k for k, it in enumerate(self.items)}
        a = b = 0
        for it in engaged:
            k = pos.get(it)
            if k is None:
                continue
            if self.team[k] == A:
                a += 1
            else:
                b += 1
        return a, b


def team_draft(
    ranking_a: Sequence[Hashable], ranking_b: Sequence[Hashable], k: int, rng: random.Random
) -> Interleaved:
    items: list[Hashable] = []
    team: list[int] = []
    shown: set[Hashable] = set()
    ia = ib = 0
    count_a = count_b = 0

    def next_new(r: Sequence[Hashable], i: int) -> int:
        while i < len(r) and r[i] in shown:
            i += 1
        return i

    while len(items) < k:
        ia, ib = next_new(ranking_a, ia), next_new(ranking_b, ib)
        a_left, b_left = ia < len(ranking_a), ib < len(ranking_b)
        if not (a_left or b_left):
            break
        pick_a = count_a < count_b or (count_a == count_b and rng.random() < 0.5)
        if not a_left:
            pick_a = False
        elif not b_left:
            pick_a = True
        if pick_a:
            it, ia, count_a = ranking_a[ia], ia + 1, count_a + 1
            team.append(A)
        else:
            it, ib, count_b = ranking_b[ib], ib + 1, count_b + 1
            team.append(B)
        items.append(it)
        shown.add(it)
    return Interleaved(items, team)


@dataclass(frozen=True)
class InterleavingResult:
    n_impressions: int  # all logged impressions
    n_engaged: int  # impressions with at least one credited event: the only ones analysed
    n_visitors: int  # visitors with an engaged impression (the independent units)
    wins_a: int
    wins_b: int
    ties: int
    delta: float  # Chapelle et al. Delta_AB in [-0.5, 0.5]; > 0 favours A
    se: float  # cluster-robust over visitors
    ci_low: float
    ci_high: float
    p_value: float
    skewness: float  # of the per-visitor linearised contributions
    clt_ok: bool  # n_visitors > 355 * skewness^2 (Kohavi et al., KDD 2014)
    winsorised: int  # visitors whose impression weight was capped


def analyse(
    visitor: Sequence[Hashable],
    credit_a: Sequence[int],
    credit_b: Sequence[int],
    winsor_q: float | None = 0.999,
    alpha: float = 0.05,
) -> InterleavingResult:
    """Delta_AB = (wins_A + ties/2) / N - 1/2 over the N impressions with a credited event
    (Chapelle, Joachims, Radlinski & Yue, TOIS 2012). Impressions without any engagement say
    nothing about the rankers and only dilute the effect, so they are dropped.

    Impressions of one visitor are correlated, so Delta is treated as a ratio of per-visitor sums
    and its standard error comes from the delta method over visitors (cluster-robust). Visitors
    with extreme impression counts (bots, crawlers, power users) have their weight capped at the
    `winsor_q` quantile, keeping their average outcome.
    """
    ca, cb = np.asarray(credit_a), np.asarray(credit_b)
    vis = np.asarray(visitor, dtype=object).astype(str)
    n_imp = int(ca.size)
    engaged = (ca + cb) > 0
    ca, cb, vis = ca[engaged], cb[engaged], vis[engaged]
    outcome = 0.5 * np.sign(ca - cb).astype(float)
    _, vid = np.unique(vis, return_inverse=True)
    m = int(vid.max()) + 1 if vid.size else 0
    # numpy's stub types bincount as intp; with float weights it returns float64
    s_v = cast(NDArray[np.float64], np.bincount(vid, weights=outcome, minlength=m))
    n_v = np.bincount(vid, minlength=m).astype(float)
    capped = 0
    if winsor_q is not None and m:
        cap = max(float(np.quantile(n_v, winsor_q)), 1.0)
        over = n_v > cap
        capped = int(over.sum())
        s_v[over] *= cap / n_v[over]
        n_v[over] = cap
    nan = float("nan")
    delta, se, skew = nan, float("inf"), nan
    if m:
        delta = float(s_v.sum() / n_v.sum())
    if m > 1:
        r = s_v - delta * n_v
        se = float(np.sqrt(m / (m - 1) * (r @ r)) / n_v.sum())
        skew = float(sps.skew(r, bias=False)) if r.std() > 0 else 0.0
    if np.isfinite(se) and se > 0:
        z = float(sps.norm.ppf(1 - alpha / 2))
        lo, hi, p = delta - z * se, delta + z * se, float(2 * sps.norm.sf(abs(delta) / se))
    else:
        lo, hi, p = nan, nan, 1.0
    return InterleavingResult(
        n_impressions=n_imp,
        n_engaged=int(outcome.size),
        n_visitors=m,
        wins_a=int((outcome > 0).sum()),
        wins_b=int((outcome < 0).sum()),
        ties=int((outcome == 0).sum()),
        delta=delta,
        se=se,
        ci_low=lo,
        ci_high=hi,
        p_value=p,
        skewness=skew,
        clt_ok=bool(m > 355 * skew**2) if np.isfinite(skew) else False,
        winsorised=capped,
    )
