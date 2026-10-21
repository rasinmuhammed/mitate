"""Mitate's blend: union candidates from every source, describe each (query, item) pair with
source, item, shopper and store-size features, and rank with LightGBM LambdaMART.

The ranker is trained on validation-window labels pooled across stores of many sizes. The
store-level data-volume features are constant inside one request, so they can only matter through
splits that treat stores of different sizes differently; the ``no_volume`` ablation measures
whether they do (docs/results.md).
Ablations switch off feature groups or sources without touching anything else.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self, cast

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from mitate.data import Catalog, Event, Log, Queries
from mitate.models.base import F32, Recommender

DAY = 86_400

STORE_FEATURES = [
    "store_log_interactions",
    "store_log_items",
    "store_log_users",
    "store_log_sessions",
    "store_density",
    "store_anon_share",
]
QUERY_FEATURES = [
    "q_log_hist_len",
    "q_session_len",
    "q_known_user",
    "q_hours_since_last",
    "q_distinct_items",
]
ITEM_FEATURES = [
    "i_log_pop",
    "i_log_pop_7d",
    "i_log_pop_30d",
    "i_age_days",
    "i_is_cold",
    "i_price_band",
    "i_buyers",
]
PAIR_FEATURES = ["p_hist_count", "p_hours_since_seen", "p_same_cat_share", "p_rel_price"]


GROUPS = {
    "store": STORE_FEATURES,
    "query": QUERY_FEATURES,
    "item": ITEM_FEATURES,
    "pair": PAIR_FEATURES,
}


@dataclass(frozen=True)
class CandidateFrame:
    """Candidates for a batch of queries. ``nominated[b, c, s]`` says whether source ``s`` put
    candidate ``c`` in its top-n (the last slot is the popularity filler)."""

    cand: NDArray[np.int32]  # (B, C), -1 padded
    X: NDArray[np.float32]  # (B, C, F)
    nominated: NDArray[np.bool_]  # (B, C, S + 1)
    feature_names: tuple[str, ...]
    source_names: tuple[str, ...]


@dataclass(frozen=True)
class Variant:
    """An ablation: sources removed from candidate generation and features, and feature groups
    blanked. Removing a source this way is exact: other sources' scores never depend on it."""

    name: str = "full"
    drop_sources: tuple[str, ...] = ()
    drop_groups: tuple[str, ...] = ()


def ablate(f: CandidateFrame, v: Variant) -> CandidateFrame:
    if not v.drop_sources and not v.drop_groups:
        return f
    keep_src = np.array([s not in v.drop_sources for s in f.source_names] + [True])
    still = (f.nominated & keep_src[None, None, :]).any(axis=2) & (f.cand >= 0)
    X = f.X.copy()
    cols = set()
    for g in v.drop_groups:
        cols |= set(GROUPS[g])
    for s in v.drop_sources:
        cols |= {f"s_{s}_score", f"s_{s}_rank", f"s_{s}_norm"}
    for j, nm in enumerate(f.feature_names):
        if nm in cols:
            X[:, :, j] = np.nan
    cand = np.where(still, f.cand, -1).astype(np.int32)
    X[~still] = np.nan
    return CandidateFrame(cand, X, f.nominated & still[..., None], f.feature_names, f.source_names)


@dataclass
class Blend(Recommender):
    """``sources`` are configured (unfitted) recommenders. ``booster`` is a fitted
    ``mitate.ltr.TreeEnsemble`` (anything with ``predict(X)``), or ``None`` to fall back to
    reciprocal-rank fusion, which needs no labels (used before a store has any)."""

    name: ClassVar[str] = "mitate"
    sources: Sequence[Recommender] = ()
    n_cand: int = 50
    variant: Variant = field(default_factory=Variant)
    booster: Any = field(default=None, repr=False)
    batch: int = 256
    _fitted: bool = field(default=False, repr=False)

    # ------------------------------------------------------------------ fitting
    def fit(self, log: Log, catalog: Catalog) -> Self:
        for s in self.sources:
            s.fit(log, catalog)
        n = log.n_items
        horizon = int(log.ts.max()) + 1 if len(log) else 0
        self._now_fit = horizon
        ev_w = np.array([1.0, 3.0, 6.0])
        w = ev_w[log.event]
        self._pop = np.bincount(log.item, weights=w, minlength=n)
        recent7 = log.ts >= horizon - 7 * DAY
        recent30 = log.ts >= horizon - 30 * DAY
        self._pop7 = np.bincount(log.item[recent7], weights=w[recent7], minlength=n)
        self._pop30 = np.bincount(log.item[recent30], weights=w[recent30], minlength=n)
        orders = log.event == Event.ORDER
        self._buyers = (
            log.where(orders).item_counts(distinct_users=True) if orders.any() else (np.zeros(n))
        )
        first = np.full(n, np.iinfo(np.int64).max)
        np.minimum.at(first, log.item, log.ts)
        self._age = np.where(first < np.iinfo(np.int64).max, (horizon - first) / DAY, np.nan)
        self._cold = self._pop == 0
        self._band = catalog.price_band().astype(np.float64)
        self._band[self._band < 0] = np.nan
        self._price = catalog.price if catalog.price is not None else np.full(n, np.nan)
        self._cat = catalog.category if catalog.category is not None else np.full(n, -1)
        known = log.user >= 0
        n_sess = np.unique(log.session).size
        n_users = np.unique(log.user[known]).size
        self._store = np.array(
            [
                np.log10(1 + len(log)),
                np.log10(1 + n),
                np.log10(1 + n_users),
                np.log10(1 + n_sess),
                len(log) / max(1.0, (n_users + n_sess) * n),
                float((~known).mean()) if len(log) else 0.0,
            ]
        )
        self._fitted = True
        return self

    # ------------------------------------------------------------------ features
    def feature_names(self) -> list[str]:
        src = []
        for s in self.sources:
            src += [f"s_{s.name}_score", f"s_{s.name}_rank", f"s_{s.name}_norm"]
        names = STORE_FEATURES + QUERY_FEATURES + ITEM_FEATURES + PAIR_FEATURES + src
        return names

    def candidates(self, q: Queries) -> CandidateFrame:
        """Union of every source's top-n plus popular fillers, with features."""
        B, n = len(q), self._pop.size
        S_all = [s.score(q) for s in self.sources]
        k = min(self.n_cand, n)
        tops = []
        for S in S_all:
            fin = np.where(np.isfinite(S), S, -np.inf)
            idx = (
                np.argpartition(-fin, k - 1, axis=1)[:, :k]
                if k < n
                else np.tile(np.arange(n), (B, 1))
            )
            order = np.argsort(-np.take_along_axis(fin, idx, 1), axis=1, kind="stable")
            tops.append(np.take_along_axis(idx, order, 1))
        # popularity candidates guarantee a full list for empty histories
        pop_top = np.argsort(-self._pop, kind="stable")[:k]
        allc = np.concatenate([*tops, np.tile(pop_top, (B, 1))], axis=1)
        C = allc.shape[1]
        cand = np.full((B, C), -1, np.int32)
        for r in range(B):
            u = np.unique(allc[r])
            cand[r, : u.size] = u
        valid = cand >= 0
        S_n = len(self.sources)
        nominated = np.zeros((B, C, S_n + 1), bool)
        for j, top in enumerate([*tops, np.tile(pop_top, (B, 1))]):
            member = np.zeros((B, n), bool)
            np.put_along_axis(member, top, True, 1)
            nominated[:, :, j] = np.take_along_axis(member, np.maximum(cand, 0), 1) & valid
        ci = np.maximum(cand, 0)
        names = self.feature_names()
        X = np.full((B, C, len(names)), np.nan, np.float32)
        f = 0
        X[:, :, f : f + len(STORE_FEATURES)] = self._store
        f += len(STORE_FEATURES)
        # query features
        lens = q.lengths()
        rows = np.repeat(np.arange(B), lens)
        last_ts: NDArray[np.float64] = np.full(B, np.nan)
        if lens.sum():
            np.maximum.at(last_ts_tmp := np.full(B, -np.inf), rows, q.ts.astype(np.float64))
            last_ts = np.where(np.isfinite(last_ts_tmp), last_ts_tmp, np.nan)
        H = q.matrix(n, binary=False, weights={e: 1.0 for e in Event})
        distinct = np.diff(H.indptr)
        qf = np.stack(
            [
                np.log1p(lens),
                lens - q.session_start,
                (q.user >= 0).astype(float),
                (q.now - last_ts) / 3600.0,
                distinct,
            ],
            axis=1,
        )
        X[:, :, f : f + len(QUERY_FEATURES)] = qf[:, None, :]
        f += len(QUERY_FEATURES)
        # item features
        itf = np.stack(
            [
                np.log1p(self._pop),
                np.log1p(self._pop7),
                np.log1p(self._pop30),
                self._age,
                self._cold.astype(float),
                self._band,
                np.log1p(self._buyers),
            ],
            axis=1,
        )
        X[:, :, f : f + len(ITEM_FEATURES)] = itf[ci]
        f += len(ITEM_FEATURES)
        # pair features
        X[:, :, f] = _gather_sparse(H, ci)
        last_seen = _last_seen(q, n, ci)
        X[:, :, f + 1] = (q.now[:, None] - last_seen) / 3600.0
        cat_share, rel_price = self._cat_and_price(q, ci)
        X[:, :, f + 2] = cat_share
        X[:, :, f + 3] = rel_price
        f += len(PAIR_FEATURES)
        for S, top in zip(S_all, tops, strict=True):
            sc = np.take_along_axis(S, ci, 1)
            rank = np.full((B, C), k + 1, np.float32)
            pos = np.full((B, n), k + 1, np.float32)
            np.put_along_axis(pos, top, np.arange(1, k + 1, dtype=np.float32)[None, :], 1)
            rank = np.take_along_axis(pos, ci, 1)
            mx = np.max(np.where(np.isfinite(S), np.abs(S), 0), axis=1, keepdims=True)
            X[:, :, f] = np.where(np.isfinite(sc), sc, np.nan)
            X[:, :, f + 1] = rank
            X[:, :, f + 2] = np.where(np.isfinite(sc), sc / np.maximum(mx, 1e-12), np.nan)
            f += 3
        X[~valid] = np.nan
        return CandidateFrame(cand, X, nominated, tuple(names), tuple(s.name for s in self.sources))

    def _cat_and_price(self, q: Queries, ci: NDArray[np.int32]) -> tuple[F32, F32]:
        B = len(q)
        share = np.full(ci.shape, np.nan, np.float32)
        relp = np.full(ci.shape, np.nan, np.float32)
        for r in range(B):
            h = q.history(r)
            if h.size == 0:
                continue
            hc = self._cat[h]
            if (hc >= 0).any():
                cats, cnt = np.unique(hc[hc >= 0], return_counts=True)
                lut = dict(zip(cats.tolist(), (cnt / h.size).tolist(), strict=True))
                share[r] = [lut.get(int(c), 0.0) for c in self._cat[ci[r]]]
            hp = self._price[h]
            hp = hp[np.isfinite(hp) & (hp > 0)]
            if hp.size:
                relp[r] = self._price[ci[r]] / np.median(hp)
        return share, relp

    # ------------------------------------------------------------------ scoring
    def score(self, q: Queries) -> F32:
        assert self._fitted
        n = self._pop.size
        out = np.empty((len(q), n), np.float32)
        bg = self.background()
        for s in range(0, len(q), self.batch):
            e = min(len(q), s + self.batch)
            fr = ablate(self.candidates(q.slice(s, e)), self.variant)
            out[s:e] = self.scores_from_frame(fr, bg)
        return out

    def scores_from_frame(self, fr: CandidateFrame, bg: F32) -> F32:
        """Dense scores: ranked candidates on top, everything else below in popularity order."""
        Sb = np.tile(bg, (fr.cand.shape[0], 1))
        sc = self._rank_scores(fr.X, fr.cand)
        rows, cols = np.nonzero(fr.cand >= 0)
        Sb[rows, fr.cand[rows, cols]] = sc[rows, cols]
        return Sb

    def background(self) -> F32:
        return cast(F32, (-1e6 + self._pop / (self._pop.max() + 1.0)).astype(np.float32))

    def _rank_scores(self, X: F32, cand: NDArray[np.int32]) -> F32:
        B, C, F = X.shape
        if self.booster is not None:
            Xf = X.reshape(B * C, F)
            want = getattr(self.booster, "feature_names", None)
            have = self.feature_names()
            if want is not None and list(want) != have:
                # A booster trained with more sources (e.g. the no_<source> variant served when a
                # store is below that source's threshold): absent sources' features are NaN,
                # exactly as in the ablation the booster was trained under.
                pos = {nm: j for j, nm in enumerate(have)}
                unknown = [nm for nm in have if nm not in set(want)]
                if unknown:
                    raise ValueError(f"booster was not trained with features {unknown}")
                Z = np.full((B * C, len(want)), np.nan, np.float32)
                for j, nm in enumerate(want):
                    if nm in pos:
                        Z[:, j] = Xf[:, pos[nm]]
                Xf = Z
            return cast(F32, self.booster.predict(Xf).reshape(B, C).astype(np.float32))
        # reciprocal-rank fusion over sources (Cormack et al., SIGIR 2009), k = 60
        names = self.feature_names()
        ranks = [j for j, nm in enumerate(names) if nm.endswith("_rank")]
        rr = np.zeros((B, C))
        for j in ranks:
            r = X[:, :, j]
            rr += np.where(np.isfinite(r) & (r <= self.n_cand), 1.0 / (60.0 + r), 0.0)
        return rr.astype(np.float32)


def _gather_sparse(M: sp.csr_matrix, ci: NDArray[np.int32]) -> F32:
    out = np.zeros(ci.shape, np.float32)
    for r in range(ci.shape[0]):
        s, e = M.indptr[r], M.indptr[r + 1]
        if s == e:
            continue
        lut = dict(zip(M.indices[s:e].tolist(), M.data[s:e].tolist(), strict=True))
        out[r] = [lut.get(int(c), 0.0) for c in ci[r]]
    return out


def _last_seen(q: Queries, n: int, ci: NDArray[np.int32]) -> NDArray[np.float64]:
    out = np.full(ci.shape, np.nan)
    for r in range(len(q)):
        s, e = q.indptr[r], q.indptr[r + 1]
        if s == e:
            continue
        # ascending time, so later duplicates overwrite earlier ones: the last sighting wins
        lut = dict(zip(q.items[s:e].tolist(), q.ts[s:e].tolist(), strict=True))
        out[r] = [lut.get(int(c), np.nan) for c in ci[r]]
    return out


# ---------------------------------------------------------------------------- LTR training
def ltr_groups(
    frames: list[CandidateFrame], targets: sp.csr_matrix, v: Variant | None = None
) -> tuple[F32, NDArray[np.floating[Any]], NDArray[np.int64]]:
    """(X, y, group sizes) for queries with at least one positive among candidates. ``frames``
    are consecutive batches covering ``targets`` row by row."""
    Xs, ys, gs = [], [], []
    row = 0
    for fr0 in frames:
        fr = ablate(fr0, v or Variant())
        for r in range(fr.cand.shape[0]):
            m = fr.cand[r] >= 0
            c = fr.cand[r][m]
            t = targets.indices[targets.indptr[row] : targets.indptr[row + 1]]
            row += 1
            lab = np.isin(c, t).astype(np.float32)
            if lab.sum() == 0:
                continue
            Xs.append(fr.X[r][m])
            ys.append(lab)
            gs.append(c.size)
    if not Xs:
        F = frames[0].X.shape[2] if frames else 0
        return np.zeros((0, F), np.float32), np.zeros(0), np.zeros(0, int)
    return np.vstack(Xs), np.concatenate(ys), np.asarray(gs)
