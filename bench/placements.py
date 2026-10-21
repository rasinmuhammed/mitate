"""Product-page placements on Amazon Grocery: "similar items" and "goes well with".

    uv run python -m bench.placements experiments/configs/placements.yaml

Ground truth is external and static: Amazon 2018 ``also_view`` lists for *similar* (substitutes,
McAuley et al., KDD 2015) and ``also_buy`` lists for *complements*. Co-purchase mixes complements
with same-need substitutes (ledger D5), so complements are also scored against
``also_buy_cross``: the ``also_buy`` pairs whose two products sit in different leaf categories of
the store. Pairs are mapped from 2018 ASINs to 2023 parent items. No list is derived from the
stores' own logs.

For every simulated store (taxonomy-grown assortment, store_simulation.md), the anchors are the
store's products with at least one ground-truth partner inside its catalog. They are split once
into a **dev** half (baseline tuning and any product decision) and a **test** half, reported once.
Behavioural methods see only the store's events before ``t_test``; text methods see only the
catalog. Mitate is the shipped engine (``Engine.recommend``, default rules and thresholds) fitted
on the same events, and again on none at all (day one).
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import polars as pl
import scipy.sparse as sp
import yaml
from numpy.typing import NDArray

from bench.datasets import load
from bench.run import get_store
from bench.simulator import Store
from mitate.constraints import Placement
from mitate.content import embed, item_texts
from mitate.data import Catalog, Log, Queries
from mitate.engine import Engine, EngineConfig
from mitate.explain import Evidence
from mitate.metrics import ranking_metrics
from mitate.models.base import Recommender, topk
from mitate.models.baselines import EASE, ItemKNN
from mitate.stats import holm, paired_test, t_interval

DATA = Path("data/amazon")
K = 20
PLACEMENT = {
    "also_view": Placement.SIMILAR,
    "also_buy": Placement.COMPLEMENTS,
    "also_buy_cross": Placement.COMPLEMENTS,
}
Method = Callable[[NDArray[np.int64]], NDArray[np.int32]]  # anchors -> top-K items (-1 padded)


# --------------------------------------------------------------------------- ground truth
def ground_truth(kind: str) -> pl.DataFrame:
    """(item, target) pairs in the benchmark's dense item space, cached next to the data."""
    if kind == "also_buy_cross":
        return ground_truth("also_buy")  # the category filter needs the store's catalog
    out = DATA / f"gt_{kind}.parquet"
    if out.exists():
        return pl.read_parquet(out)
    variant = (
        pl.scan_parquet(DATA / "reviews.parquet")
        .select(pl.col("asin").alias("a"), pl.col("parent_asin").alias("p"))
        .unique()
        .collect()
    )
    items = pl.read_parquet(DATA / "items.parquet").select("item", "parent_asin")
    raw = pl.read_parquet(DATA / f"{kind}.parquet").rename({"asin": "src", kind: "dst"})

    def to_item(col: str) -> pl.DataFrame:
        """2018 ASIN -> 2023 parent item: through a reviewed variant, else the ASIN itself."""
        a = raw.select(col).unique()
        via_variant = a.join(variant, left_on=col, right_on="a").select(col, "p")
        direct = a.join(items, left_on=col, right_on="parent_asin").select(
            col, pl.col(col).alias("p")
        )
        m = pl.concat([via_variant, direct]).unique(col, keep="first", maintain_order=True)
        return m.join(items, left_on="p", right_on="parent_asin").select(col, "item")

    gt = (
        raw.join(to_item("src").rename({"item": "a"}), on="src")
        .join(to_item("dst").rename({"item": "b"}), on="dst")
        .select(pl.col("a").alias("item"), pl.col("b").alias("target"))
        .filter(pl.col("item") != pl.col("target"))
        .unique()
        .sort("item", "target")
    )
    gt.write_parquet(out)
    return gt


def anchor_split(
    st: Store, gt: pl.DataFrame, max_anchors: int, cross_category: bool = False
) -> tuple[dict[str, NDArray[np.int64]], sp.csr_matrix]:
    """Dev/test anchors (dense store ids) and the anchor x item ground-truth matrix."""
    raw = np.asarray(st.item_raw, np.int64)
    g = gt.filter(pl.col("item").is_in(raw) & pl.col("target").is_in(raw))
    a = np.searchsorted(raw, g["item"].to_numpy())
    b = np.searchsorted(raw, g["target"].to_numpy())
    if cross_category:
        assert st.catalog.category is not None
        cat = np.asarray(st.catalog.category)
        keep = cat[a] != cat[b]
        a, b = a[keep], b[keep]
    n = raw.size
    T = sp.csr_matrix((np.ones(a.size), (a, b)), shape=(n, n))
    T.data[:] = 1.0
    anchors = np.unique(a)
    rng = np.random.default_rng(st.seed)
    anchors = rng.permutation(anchors)[: 2 * max_anchors]
    half = anchors.size // 2
    return {"dev": np.sort(anchors[:half]), "test": np.sort(anchors[half:])}, T


# --------------------------------------------------------------------------- methods
def _ranked(S: NDArray[np.float64], anchors: NDArray[np.int64]) -> NDArray[np.int32]:
    """Top-K per anchor; items scored -inf are not recommended (the list is padded with -1)."""
    S32 = S.astype(np.float32)
    S32[np.arange(anchors.size), anchors] = -np.inf  # never recommend the anchor itself
    recs = topk(S32, K)
    recs[~np.isfinite(np.take_along_axis(S32, recs.astype(np.int64), axis=1))] = -1
    return recs


def _content(V: NDArray[np.float32], other_than: NDArray[np.int32] | None = None) -> Method:
    """Text similarity; with ``other_than`` (item categories), only items of another category."""

    def run(anchors: NDArray[np.int64]) -> NDArray[np.int32]:
        S = (V[anchors] @ V.T).astype(np.float64)
        if other_than is not None:
            S[other_than[None, :] == other_than[anchors, None]] = -np.inf
        return _ranked(S, anchors)

    return run


def _item_item(model: Recommender) -> Method:
    def run(anchors: NDArray[np.int64]) -> NDArray[np.int32]:
        S = model.score(Queries.from_lists([[int(a)] for a in anchors]))
        return _ranked(S.astype(np.float64), anchors)

    return run


def ndcg10(recs: NDArray[np.int32], T: sp.csr_matrix, anchors: NDArray[np.int64]) -> float:
    return float(np.nanmean(ranking_metrics(recs, T[anchors], (10,))[10].ndcg))


def tune_item_item(
    cls: type[Recommender],
    log: Log,
    cat: Catalog,
    T: sp.csr_matrix,
    dev: NDArray[np.int64],
    trials: int,
    seed: int,
) -> tuple[Method, dict[str, Any]]:
    """Equal-budget TPE on the dev anchors (the placement's own objective)."""

    def objective(trial: optuna.Trial) -> float:
        m = cls(**cls.space(trial), seed=seed).fit(log, cat)
        return ndcg10(_item_item(m)(dev), T, dev)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True)
    )
    study.optimize(objective, n_trials=trials)
    best = dict(study.best_params)
    return _item_item(cls(**best, seed=seed).fit(log, cat)), best | {"dev": study.best_value}


def baselines(
    st: Store,
    vecs: dict[str, NDArray[np.float32]],
    T: sp.csr_matrix,
    dev: NDArray[np.int64],
    trials: int,
) -> tuple[dict[str, Method], dict[str, Any]]:
    log, cat = st.log.until(st.t_test), st.catalog
    n = cat.n_items
    pop = np.log1p(log.item_counts(distinct_users=True))
    category = np.asarray(cat.category) if cat.category is not None else np.full(n, -1)
    same = category[None, :] == category[:, None]  # n x n bool; stores stay below ~10K items

    def by_category(sign: int) -> Method:
        def run(anchors: NDArray[np.int64]) -> NDArray[np.int32]:
            S = pop[None, :] + 1e3 * (same[anchors] if sign > 0 else ~same[anchors])
            return _ranked(S, anchors)

        return run

    out: dict[str, Method] = {
        "pop": lambda anchors: _ranked(np.tile(pop, (anchors.size, 1)), anchors),
        "same_category_pop": by_category(+1),
        "other_category_pop": by_category(-1),
    }
    for name, V in vecs.items():
        out[f"content_{name}"] = _content(V)
        out[f"content_{name}_other_category"] = _content(V, category)
    params: dict[str, Any] = {}
    if len(log):
        B = Evidence(min_support=1).fit(log, cat).bought.astype(np.float64).tocsr()
        base = np.maximum(B.diagonal(), 1.0)

        def co_purchase(anchors: NDArray[np.int64]) -> NDArray[np.int32]:
            C = B[anchors].toarray() / np.sqrt(base[anchors, None] * base[None, :])
            return _ranked(C + 1e-6 * pop[None, :], anchors)  # popularity breaks ties

        out["co_purchase"] = co_purchase
        for cls in (ItemKNN, EASE):
            out[cls.name], params[cls.name] = tune_item_item(cls, log, cat, T, dev, trials, st.seed)
    return out, params


def mitate(st: Store, kind: str, V: NDArray[np.float32], day_one: bool) -> Method:
    """The shipped engine: product defaults, default rules, no merchant input."""
    log = st.log.until(st.t_test)
    if day_one:
        log = log.until(int(log.ts.min()) if len(log) else 0)
    eng = Engine(EngineConfig(min_behaviour_events=2**62))  # the next-item blend is not needed
    eng.fit(log, st.catalog, vectors=V)
    placement = PLACEMENT[kind]

    def run(anchors: NDArray[np.int64]) -> NDArray[np.int32]:
        recs = np.full((anchors.size, K), -1, np.int32)
        for r, a in enumerate(anchors.tolist()):
            items = [x.item for x in eng.recommend(placement, K, anchors=[a])]
            recs[r, : len(items)] = items
        return recs

    return run


# --------------------------------------------------------------------------- per store
def evaluate_store(
    st: Store, gts: dict[str, pl.DataFrame], cfg: dict[str, Any], out: Path
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    texts = item_texts(st.catalog)
    vecs = {m: embed(texts, m) for m in cfg["embedders"]}
    summary: dict[str, Any] = {"store": st.name, "stats": st.stats}
    for kind, gt in gts.items():
        split, T = anchor_split(st, gt, cfg["max_anchors"], kind == "also_buy_cross")
        if split["test"].size == 0:
            summary[kind] = {"anchors": 0}
            continue
        methods, params = baselines(st, vecs, T, split["dev"], cfg["trials"])
        V = vecs[cfg["mitate_embedder"]]
        methods["mitate"] = mitate(st, kind, V, day_one=False)
        methods["mitate_day_one"] = mitate(st, kind, V, day_one=True)
        arrays: dict[str, NDArray[np.float64]] = {}
        res: dict[str, dict[str, dict[str, float]]] = {}
        for part, anchors in split.items():
            res[part] = {}
            for name, f in methods.items():
                recs = f(anchors)
                m = ranking_metrics(recs, T[anchors], (10, 20))
                per = {
                    "ndcg@10": m[10].ndcg,
                    "recall@10": m[10].recall,
                    "hit@10": m[10].hit,
                    "recall@20": m[20].recall,
                }
                res[part][name] = {k: float(np.nanmean(v)) for k, v in per.items()}
                res[part][name]["filled@10"] = float((recs[:, :10] >= 0).mean())
                arrays[f"{part}:{name}"] = per["ndcg@10"]
        # numpy's stub cannot tell these keys from its bool ``allow_pickle`` keyword
        np.savez_compressed(out / f"{kind}.npz", **split, **arrays)  # type: ignore[arg-type]
        summary[kind] = {
            "anchors": {p: int(a.size) for p, a in split.items()},
            "targets_per_anchor": float(T[split["test"]].sum(axis=1).mean()),
            "params": params,
            **res,
        }
    summary["seconds"] = time.monotonic() - t0
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


# --------------------------------------------------------------------------- analysis
def dev_choice(z: Any, methods: list[str]) -> str:
    """The baseline a merchant could deploy: the best on this store's dev anchors."""
    return max(methods, key=lambda m: float(np.nanmean(z[f"dev:{m}"])))


def analyse(root: Path, kinds: list[str]) -> dict[str, Any]:
    """Per size: test NDCG@10 per method (mean and t-interval across stores). Mitate is compared,
    paired over pooled test anchors, with (a) the baseline each store would pick on its dev
    anchors and (b) the best baseline on test (oracle, a harsher bar). Holm across sizes."""
    report: dict[str, Any] = {}
    stores = sorted(p for p in root.iterdir() if (p / "summary.json").exists())

    def size(p: Path) -> int:
        return int(p.name.split("-n")[1].split("-")[0])

    for kind in kinds:
        rows: dict[str, Any] = {}
        for n in sorted({size(p) for p in stores}):
            files = [p / f"{kind}.npz" for p in stores if size(p) == n]
            group = [np.load(f) for f in files if f.exists()]
            if not group:
                continue
            names = [f[5:] for f in group[0].files if f.startswith("test:")]
            store_means = {m: [float(np.nanmean(z[f"test:{m}"])) for z in group] for m in names}
            table = {
                m: dict(zip(("mean", "lo", "hi"), t_interval(v), strict=True)) | {"stores": len(v)}
                for m, v in store_means.items()
            }
            base = [m for m in names if not m.startswith("mitate")]
            selected = np.concatenate([z[f"test:{dev_choice(z, base)}"] for z in group])
            oracle = max(base, key=lambda m: table[m]["mean"])
            mine = np.concatenate([z["test:mitate"] for z in group])
            rows[str(n)] = {
                "methods": table,
                "anchors": int(mine.size),
                "vs_dev_selected": paired_test(mine, selected).as_dict(),
                "vs_oracle": {"model": oracle}
                | paired_test(mine, np.concatenate([z[f"test:{oracle}"] for z in group])).as_dict(),
            }
        for key in ("vs_dev_selected", "vs_oracle"):
            adj = holm([r[key]["p_ttest"] for r in rows.values()])
            for r, p in zip(rows.values(), adj, strict=True):
                r[key]["p_adj"] = float(p)
        report[kind] = rows
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--buckets", type=int, nargs="*")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    ds = load(cfg["dataset"])
    root = Path(cfg["results_dir"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text(Path(a.config).read_text())
    gts = {k: ground_truth(k) for k in cfg["kinds"]}
    for b in cfg["buckets"]:
        if a.buckets and b["n"] not in a.buckets:
            continue
        for seed in cfg["seeds"]:
            st = get_store(ds, b["n"], b["items"], seed, None)
            if st is None:
                print(f"{ds.name}-n{b['n']}-s{seed}: infeasible", flush=True)
                continue
            out = root / st.name
            if (out / "summary.json").exists():
                continue
            s = evaluate_store(st, gts, cfg, out)
            print(
                f"[{st.name}] {st.catalog.n_items} items, anchors "
                f"{ {k: s[k]['anchors'] for k in cfg['kinds']} } ({s['seconds']:.0f}s)",
                flush=True,
            )
    rep = analyse(root, cfg["kinds"])
    (root / "analysis.json").write_text(json.dumps(rep, indent=1))
    for kind, rows in rep.items():
        for n, row in rows.items():
            top = sorted(row["methods"].items(), key=lambda kv: -kv[1]["mean"])[:5]
            print(kind, n, [(m, round(d["mean"], 4)) for m, d in top])


if __name__ == "__main__":
    main()
