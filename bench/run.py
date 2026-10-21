"""Benchmark runner: the one entry point behind every reported number.

    python -m bench.run experiments/configs/benchmark.yaml [--datasets otto] [--stage 1|2|all]

Stage 1, per store (independent, resumable):
  1. sample (or load) the store; build validation (Last targets in [t_val, t_test)) and test
     (Successive targets in [t_test, t_end)) sets;
  2. tune every model with the same Optuna budget on validation;
  3. refit with the best parameters on everything before t_test and score the test set;
  4. fit the blend's sources (best parameters) before t_val and save LTR groups from validation.
Stage 2, per dataset:
  5. pool the LTR groups of every store, tune and train one LambdaMART per variant;
  6. per store, refit sources before t_test and evaluate every Mitate variant on the test set.

Everything is seeded; outputs go to experiments/results/<dataset>/<store>/.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from bench import tasks
from bench.datasets import Dataset, load
from bench.evaluate import evaluate
from bench.registry import MODELS, NEURAL
from bench.simulator import (
    Store,
    _catalog,
    cobrowse_graph,
    sample_store,
    save_manifest,
    widened_targets,
)
from bench.tuning import evaluate_config, tune
from mitate.blend import Blend, CandidateFrame, Variant, ablate
from mitate.data import I64, Log
from mitate.ltr import TreeEnsemble, train_isolated
from mitate.models.base import F32, topk

RESULTS = Path("experiments/results")
STORES = Path("data/stores")


# --------------------------------------------------------------------------- stores
def store_path(ds: Dataset, n: int, seed: int) -> Path:
    # keyed by the split: a store's assortment is chosen with data before t_test only
    return STORES / ds.name / f"t{ds.t_test}" / f"{ds.name}-n{n}-s{seed}.npz"


def get_store(ds: Dataset, n: int, n_items: int, seed: int, graph: Any) -> Store | None:
    p = store_path(ds, n, seed)
    if p.exists():
        z = np.load(p, allow_pickle=False)
        log = Log(
            z["user"],
            z["item"],
            z["ts"],
            z["event"],
            z["session"],
            int(z["n_users"]),
            int(z["n_items"]),
        )
        stats = json.loads(str(z["stats"]))
        # stores cached before the requested size was recorded: any size widening could reach
        built_for = stats.get("n_items_requested")
        ok = (
            built_for == n_items
            if built_for
            else stats["n_items_target"] in widened_targets(n_items)
        )
        if not ok:
            raise ValueError(
                f"{p} was built for {built_for or stats['n_items_target']} items, "
                f"not {n_items}; delete it to rebuild"
            )
        return Store(
            ds.name,
            n,
            seed,
            log,
            _catalog(ds, z["item_raw"]),
            ds.t_val,
            ds.t_test,
            ds.t_end,
            z["item_raw"],
            stats,
        )
    st = sample_store(ds, n, n_items, seed, graph)
    if st is None:
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    lg = st.log
    np.savez_compressed(
        p,
        user=lg.user,
        item=lg.item,
        ts=lg.ts,
        event=lg.event,
        session=lg.session,
        n_users=lg.n_users,
        n_items=lg.n_items,
        item_raw=st.item_raw,
        stats=json.dumps(st.stats),
    )
    return st


def eval_sets(
    st: Store, task: str, cfg: dict[str, Any], exclude_seen: bool
) -> tuple[tasks.EvalSet, tasks.EvalSet]:
    """Validation and test queries; raises ``tasks.EmptyWindowError`` if either window has none."""
    make = tasks.next_item if task == "next_item" else tasks.next_basket
    val = make(
        st.log,
        st.t_val,
        st.t_test,
        target="last",
        exclude_seen=exclude_seen,
        seed=st.seed,
        max_units=cfg["tuning"]["val_units"],
    )
    test = make(
        st.log,
        st.t_test,
        st.t_end,
        target="successive",
        exclude_seen=exclude_seen,
        seed=st.seed,
        max_units=cfg["test"]["units"],
    )
    for name, es in (("validation", val), ("test", test)):
        if len(es) == 0:
            raise tasks.EmptyWindowError(f"{st.name}: no {name} query")
    return val, test


# --------------------------------------------------------------------------- stage 1
@dataclass
class StoreRun:
    st: Store
    out: Path
    cfg: dict[str, Any]
    dcfg: dict[str, Any]
    trials: int = 0
    shared: Path | None = None  # store whose tuned configurations are reused (ledger D18)
    val: tasks.EvalSet | None = None
    test: tasks.EvalSet | None = None

    def __post_init__(self) -> None:
        if self.val is None or self.test is None:
            self.val, self.test = eval_sets(
                self.st, self.dcfg["task"], self.cfg, self.dcfg.get("exclude_seen", False)
            )

    @property
    def avail(self) -> I64:
        return tasks.available_from(self.st.log)

    def best_params(self, name: str) -> dict[str, Any]:
        p = self.out / "tuning" / f"{name}.json"
        return json.loads(p.read_text())["best_params"] if p.exists() else {}

    def run_model(self, name: str) -> None:
        dest = self.out / "test" / f"{name}.npz"
        if dest.exists():
            return
        assert self.val is not None and self.test is not None  # filled by __post_init__
        cls = MODELS[name]
        cfg = self.cfg
        train_v = self.st.log.until(self.st.t_val)
        fixed: dict[str, Any] = {}
        if name in NEURAL:
            fixed["time_budget_s"] = cfg["tuning"]["neural_trial_budget_s"]
            fixed["device"] = cfg["tuning"].get("neural_device", "cpu")
        t0 = time.monotonic()
        source = self.shared / "tuning" / f"{name}.json" if self.shared else None
        if source is not None and source.exists():
            tr = evaluate_config(
                cls,
                fixed,
                json.loads(source.read_text())["best_params"],
                train_v,
                self.st.catalog,
                self.val,
                seed=self.st.seed,
                available_from=self.avail,
            )
        else:
            source = None
            tr = tune(
                cls,
                fixed,
                train_v,
                self.st.catalog,
                self.val,
                self.trials or cfg["tuning"]["trials"],
                seed=self.st.seed,
                available_from=self.avail,
            )
        (self.out / "tuning").mkdir(parents=True, exist_ok=True)
        (self.out / "tuning" / f"{name}.json").write_text(
            json.dumps(
                {
                    "best_params": tr.best_params,
                    "best_value": tr.best_value,
                    "n_trials": tr.n_trials,
                    "seconds": tr.seconds,
                    "trials": tr.trials,
                    "shared_from": source.parent.parent.name if source else None,
                },
                indent=1,
                default=str,
            )
        )
        params = dict(tr.best_params)
        if name in NEURAL:
            params.pop("epochs", None)
            params["epochs"] = max(int(tr.best_params.get("epochs", 1)), 1)
        train_t = self.st.log.until(self.st.t_test)
        m = cls(**fixed, **params, seed=self.st.seed)
        t1 = time.monotonic()
        try:
            m.fit(train_t, self.st.catalog)
        except MemoryError as e:
            self._save_skip(name, str(e))
            return
        fit_s = time.monotonic() - t1
        t2 = time.monotonic()
        recs = m.recommend(
            self.test.queries,
            self.cfg["test"]["k"],
            exclude=self.test.exclude,
            available_from=self.avail,
        )
        pred_s = time.monotonic() - t2
        self._save(
            name,
            recs,
            train_t,
            {
                "params": params,
                "tune_s": tr.seconds,
                "fit_s": fit_s,
                "predict_s": pred_s,
                "total_s": time.monotonic() - t0,
                "val_ndcg@10": tr.best_value,
            },
        )

    def _save_skip(self, name: str, why: str) -> None:
        (self.out / "test").mkdir(parents=True, exist_ok=True)
        (self.out / "test" / f"{name}.json").write_text(json.dumps({"skipped": why}))

    def _save(self, name: str, recs: NDArray[np.int32], train: Log, meta: dict[str, Any]) -> None:
        assert self.test is not None  # filled by __post_init__
        per, store_level = evaluate(recs, self.test, train)
        (self.out / "test").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.out / "test" / f"{name}.npz",
            recs=recs,
            unit=self.test.unit,
            # numpy's stub cannot tell these keys from its bool ``allow_pickle`` keyword
            **{f"m:{k}": v for k, v in per.items()},  # type: ignore[arg-type]
            **{f"seg:{k}": v for k, v in self.test.segments.items()},  # type: ignore[arg-type]
        )
        with warnings.catch_warnings():  # metrics without eligible queries are NaN by design
            warnings.simplefilter("ignore", RuntimeWarning)
            summary = {k: float(np.nanmean(v)) for k, v in per.items()}
        (self.out / "test" / f"{name}.json").write_text(
            json.dumps(
                {"summary": summary, "store_level": store_level, **meta}, indent=1, default=str
            )
        )

    def blend_sources(self) -> list[Any]:
        out = []
        for name in self.dcfg["blend_sources"]:
            p = self.best_params(name)
            if name in NEURAL:
                p = {k: v for k, v in p.items() if k != "epochs"} | {
                    "epochs": max(int(p.get("epochs", 1)), 1),
                    "device": self.cfg["tuning"].get("neural_device", "cpu"),
                }
            out.append(MODELS[name](**p, seed=self.st.seed))
        return out

    def ltr_groups(self) -> None:
        dest = self.out / "ltr_val.npz"
        if dest.exists():
            return
        assert self.val is not None  # filled by __post_init__
        cfg = self.cfg["blend"]
        rng = np.random.default_rng(self.st.seed)
        units = np.unique(self.val.unit)
        pick = rng.choice(units, min(cfg["ltr_val_units"], units.size), replace=False)
        sub = self.val.subset(np.flatnonzero(np.isin(self.val.unit, pick)))
        b = Blend(sources=self.blend_sources(), n_cand=cfg["n_cand"])
        b.fit(self.st.log.until(self.st.t_val), self.st.catalog)
        rows_X, rows_y, rows_nom, groups = [], [], [], []
        for s in range(0, len(sub), b.batch):
            e = min(len(sub), s + b.batch)
            fr = b.candidates(sub.queries.slice(s, e))
            T = sub.targets[s:e]
            for r in range(e - s):
                m = fr.cand[r] >= 0
                c = fr.cand[r][m]
                lab = np.isin(c, T.indices[T.indptr[r] : T.indptr[r + 1]])
                if not lab.any():
                    continue
                rows_X.append(fr.X[r][m].astype(np.float16))
                rows_y.append(lab)
                rows_nom.append(fr.nominated[r][m])
                groups.append(c.size)
        np.savez_compressed(
            dest,
            X=np.vstack(rows_X),
            y=np.concatenate(rows_y),
            nom=np.vstack(rows_nom),
            groups=np.asarray(groups),
            feature_names=np.array(fr.feature_names),
            source_names=np.array(fr.source_names),
            n_val_queries=len(sub),
            store_features=np.array(b._store),
        )


def scale_rule(cfg: dict[str, Any], n: int) -> dict[str, Any]:
    """The compute rule for a bucket of ``n`` interactions: the last ``tuning.scale`` entry with
    ``from_n <= n``, or no rule. Rules are part of the protocol (ledger D18)."""
    rules = [r for r in cfg["tuning"].get("scale", ()) if r["from_n"] <= n]
    return max(rules, key=lambda r: r["from_n"]) if rules else {}


def stage1(
    ds: Dataset,
    cfg: dict[str, Any],
    dcfg: dict[str, Any],
    buckets: list[dict[str, int]],
    seeds: list[int],
    spare: int = 0,
) -> list[Store]:
    """Stage 1 on every bucket. A bucket's stores are the first ``len(seeds)`` of ``seeds``, then
    up to ``spare`` further seeds, whose store can be sampled and has validation and test
    queries; every seed passed over is recorded in the manifest with its reason (ledger D24)."""
    graph = (
        cobrowse_graph(ds, Path("data") / ds.name / f"cobrowse-t{ds.t_test}.parquet")
        if (ds.subcatalog == "cobrowse")
        else None
    )
    manifest = RESULTS / ds.name / "stores.json"
    include_items = ds.license.startswith("CC BY")
    stores = []
    for b in buckets:
        rule = scale_rule(cfg, b["n"])
        models = [m for m in dcfg["models"] if m not in rule.get("skip", ())]
        shared: Path | None = None
        candidates = [*seeds, *range(max(seeds) + 1, max(seeds) + 1 + spare)]
        found = 0
        for seed in candidates:
            if found == len(seeds):
                break
            st = get_store(ds, b["n"], b["items"], seed, graph)
            sets, reason = None, "the sampler cannot reach this size"
            if st is not None:
                try:
                    sets = eval_sets(st, dcfg["task"], cfg, dcfg.get("exclude_seen", False))
                except tasks.EmptyWindowError as e:
                    reason = str(e)
            if st is None or sets is None:
                print(f"[{ds.name}] bucket {b['n']} seed {seed}: {reason}; passed over", flush=True)
                name = f"{ds.name}-n{b['n']}-s{seed}"
                row = {"store": name, "dataset": ds.name, "bucket": b["n"], "seed": seed}
                save_manifest([], manifest, include_items, skipped=[row | {"skipped": reason}])
                continue
            found += 1
            stores.append(st)
            out = RESULTS / ds.name / st.name
            out.mkdir(parents=True, exist_ok=True)
            (out / "store.json").write_text(json.dumps(st.stats, indent=1))
            save_manifest([st], manifest, include_items)
            trials = rule.get("trials", cfg["tuning"]["trials"])
            run = StoreRun(st, out, cfg, dcfg, trials, shared, *sets)
            if rule.get("shared_tuning") and shared is None:
                shared = out  # the bucket's first feasible store tunes; the rest reuse
            for name in models:
                t = time.monotonic()
                run.run_model(name)
                print(f"[{st.name}] {name} done ({time.monotonic() - t:.0f}s)", flush=True)
            run.ltr_groups()
        if found < len(seeds):
            print(f"[{ds.name}] bucket {b['n']}: {found} of {len(seeds)} stores", flush=True)
    return stores


def recorded_stores(ds: Dataset, buckets: list[dict[str, int]]) -> list[Store]:
    """The stores stage 1 evaluated, read from its manifest."""
    rows = json.loads((RESULTS / ds.name / "stores.json").read_text())
    items = {b["n"]: b["items"] for b in buckets}
    out = []
    for r in rows:
        if r["bucket"] in items and "skipped" not in r:
            st = get_store(ds, r["bucket"], items[r["bucket"]], r["seed"], None)
            assert st is not None, f"{r['store']} is in the manifest but cannot be loaded"
            out.append(st)
    return out


# --------------------------------------------------------------------------- stage 2
def variants(cfg: dict[str, Any], sources: list[str]) -> list[tuple[Variant, bool]]:
    out = [
        (Variant(v["name"], drop_groups=tuple(v.get("drop_groups", ()))), bool(v.get("rrf")))
        for v in cfg["blend"]["variants"]
    ]
    out += [(Variant(f"no_{s}", drop_sources=(s,)), False) for s in sources]
    return out


def _ablate_rows(
    X: NDArray[np.float16],
    nom: NDArray[np.bool_],
    y: NDArray[np.bool_],
    groups: NDArray[np.int64],
    fnames: list[str],
    snames: list[str],
    v: Variant,
) -> tuple[F32, F32, NDArray[np.intp]]:
    """Row-wise version of ``mitate.blend.ablate`` for stored LTR groups."""
    fr = CandidateFrame(
        np.zeros((1, 1), np.int32),
        X[None].astype(np.float32),
        nom[None],
        tuple(fnames),
        tuple(snames),
    )
    keep_src = np.array([s not in v.drop_sources for s in snames] + [True])
    still = (nom & keep_src[None, :]).any(axis=1)
    Xa = (
        ablate(fr, Variant(v.name, (), v.drop_groups)).X[0]
        if v.drop_groups
        else X.astype(np.float32)
    )
    for s in v.drop_sources:
        for j, nm in enumerate(fnames):
            if nm.startswith(f"s_{s}_"):
                Xa[:, j] = np.nan
    gid = np.repeat(np.arange(groups.size), groups)
    Xa, ya, gid = Xa[still], y[still], gid[still]
    g_new = np.bincount(gid, minlength=groups.size)
    has_pos = np.bincount(gid, weights=ya.astype(float), minlength=groups.size) > 0
    keep_rows = has_pos[gid]
    return Xa[keep_rows], ya[keep_rows].astype(np.float32), g_new[has_pos]


def stage2(ds: Dataset, cfg: dict[str, Any], dcfg: dict[str, Any], stores: list[Store]) -> None:
    pools = []
    for st in stores:
        p = RESULTS / ds.name / st.name / "ltr_val.npz"
        if p.exists():
            pools.append((st, np.load(p)))
    if not pools:
        return
    snames = [str(x) for x in pools[0][1]["source_names"]]
    boosters = train_rankers(ds, cfg, pools)
    todo = variants(cfg, snames)
    # the ranker the product ships, trained on another dataset's stores (ledger D25)
    if src := dcfg.get("transfer_from"):
        boosters[f"{src}_full"] = TreeEnsemble.load(RESULTS / src / "_ltr" / "full.json")
        todo.append((Variant(f"{src}_full"), False))
    for st in stores:
        run = StoreRun(st, RESULTS / ds.name / st.name, cfg, dcfg)
        score_mitate(run, boosters, todo)


def train_rankers(
    ds: Dataset, cfg: dict[str, Any], pools: list[tuple[Store, Any]]
) -> dict[str, TreeEnsemble | None]:
    """One LambdaMART per variant, trained on the validation groups of every store together;
    saved under ``<results>/<dataset>/_ltr`` and reloaded on resume. Rank fusion has none."""
    fnames = [str(x) for x in pools[0][1]["feature_names"]]
    snames = [str(x) for x in pools[0][1]["source_names"]]
    model_dir = RESULTS / ds.name / "_ltr"
    model_dir.mkdir(parents=True, exist_ok=True)
    boosters: dict[str, TreeEnsemble | None] = {}
    for v, rrf in variants(cfg, snames):
        if rrf:
            boosters[v.name] = None
            continue
        path = model_dir / f"{v.name}.json"
        if path.exists():
            boosters[v.name] = TreeEnsemble.load(path)
            continue
        parts = [
            _ablate_rows(z["X"], z["nom"], z["y"], z["groups"], fnames, snames, v) for _, z in pools
        ]
        boosters[v.name] = _fit_ltr(parts, fnames, cfg, model_dir / f"{v.name}.job.npz", path)
    return boosters


def score_mitate(
    run: StoreRun,
    boosters: dict[str, TreeEnsemble | None],
    todo_variants: list[tuple[Variant, bool]],
) -> None:
    """Every Mitate variant on one store's test queries: the sources are fitted once on the
    pre-test log and each variant re-ranks the same candidates."""
    st, out, cfg = run.st, run.out, run.cfg
    assert run.test is not None  # filled by __post_init__
    todo = [
        (v, rrf) for v, rrf in todo_variants if not (out / "test" / f"mitate_{v.name}.npz").exists()
    ]
    if not todo:
        return
    b = Blend(sources=run.blend_sources(), n_cand=cfg["blend"]["n_cand"])
    train_t = st.log.until(st.t_test)
    t0 = time.monotonic()
    b.fit(train_t, st.catalog)
    fit_s = time.monotonic() - t0
    bg = b.background()
    avail = run.avail
    k = cfg["test"]["k"]
    recs = {v.name: np.full((len(run.test), k), -1, np.int32) for v, _ in todo}
    pred_s = {v.name: 0.0 for v, _ in todo}
    for s in range(0, len(run.test), b.batch):
        e = min(len(run.test), s + b.batch)
        t1 = time.monotonic()
        fr_full = b.candidates(run.test.queries.slice(s, e))
        cand_s = time.monotonic() - t1
        for v, _ in todo:
            t2 = time.monotonic()
            b.booster = boosters[v.name]
            S = b.scores_from_frame(ablate(fr_full, v), bg)
            if run.test.exclude is not None:
                ex = run.test.exclude[s:e].tocoo()
                S[ex.row, ex.col] = -np.inf
            S[avail[None, :] > run.test.queries.now[s:e, None]] = -np.inf
            recs[v.name][s:e] = topk(S, k)
            pred_s[v.name] += cand_s + time.monotonic() - t2
    for v, _ in todo:
        run._save(
            f"mitate_{v.name}",
            recs[v.name],
            train_t,
            {"variant": v.name, "fit_s": fit_s, "predict_s": pred_s[v.name]},
        )
    print(f"[{st.name}] mitate variants done", flush=True)


def _fit_ltr(
    parts: list[tuple[F32, F32, NDArray[np.intp]]],
    fnames: list[str],
    cfg: dict[str, Any],
    job: Path,
    out: Path,
    seed: int = 0,
) -> TreeEnsemble:
    """Tune LambdaMART on a store-level 80/20 split of the pooled groups (same trial budget as
    every model), refit on all groups, in an isolated process (see mitate.ltr)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(parts))
    n_val = max(1, len(parts) // 5) if len(parts) > 1 else 0
    va, tr = idx[:n_val], idx[n_val:]

    def cat(
        ids: NDArray[np.integer[Any]],
    ) -> tuple[F32, NDArray[np.floating[Any]], NDArray[np.integer[Any]]]:
        sel = [parts[i] for i in ids if parts[i][2].size]
        if not sel:
            return np.zeros((0, len(fnames)), np.float32), np.zeros(0), np.zeros(0, np.int64)
        return (
            np.vstack([p[0] for p in sel]),
            np.concatenate([p[1] for p in sel]),
            np.concatenate([p[2] for p in sel]),
        )

    X, y, g = cat(np.arange(len(parts)))
    split = (*cat(tr), *cat(va)) if n_val else None
    ens = train_isolated(
        job,
        out,
        X,
        y,
        g,
        fnames,
        seed=seed,
        trials=cfg["blend"]["ltr_trials"] if split else 0,
        valid_split=split,
    )
    job.unlink(missing_ok=True)
    return ens


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--stage", default="all", choices=["1", "2", "all"])
    ap.add_argument("--buckets", nargs="*", type=int, help="restrict to these bucket sizes")
    ap.add_argument("--seeds", nargs="*", type=int)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    # Results are resumable (finished models are skipped), so every config writes to its own
    # directory: a probe run can never be mistaken for a results-of-record run.
    global RESULTS
    RESULTS = Path(cfg.get("results_dir", "experiments/results"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "config.yaml").write_text(Path(a.config).read_text())
    for name, dcfg in cfg["datasets"].items():
        if a.datasets and name not in a.datasets:
            continue
        ds = load(name)
        buckets = [b for b in cfg["buckets"] if not a.buckets or b["n"] in a.buckets]
        # explicit --seeds run exactly those seeds; the configured set may draw spare seeds
        seeds = a.seeds if a.seeds is not None else cfg["seeds"]
        spare = 0 if a.seeds is not None else cfg.get("spare_seeds", 0)
        stores = (
            stage1(ds, cfg, dcfg, buckets, seeds, spare)
            if a.stage in ("1", "all")
            else recorded_stores(ds, buckets)
        )
        if a.stage in ("2", "all"):
            stage2(ds, cfg, dcfg, stores)


if __name__ == "__main__":
    main()
