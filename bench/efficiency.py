"""Efficiency benchmark (docs/protocol.md section 9): measured, not estimated.

    uv run python -m bench.efficiency experiments/configs/benchmark.yaml --dataset otto --seed 0
    uv run python -m bench.efficiency experiments/configs/benchmark.yaml --buckets 1000 3000

For every (store size, model) pair a **fresh, single-threaded process** fits the model with the
store's tuned hyperparameters on all data before ``t_test`` and reports:

* ``fit_s``: wall-clock fit time;
* ``peak_rss_mb``: peak resident memory of the process, and ``rss_delta_mb`` above the baseline
  after imports and loading the store (the parent resolves the dataset, so the child never
  touches the full source data);
* ``size_mb``: serialised (pickled) model size;
* ``p50_ms`` / ``p95_ms``: single-request latency (one shopper, full-catalog scoring, top-20,
  seen/availability masks) over up to 500 test queries, after 20 warm-up requests;
* ``qps_batch``: throughput when scoring the test queries in batches of 256.

Single-threaded means OMP/BLAS/numba/torch threads are pinned to 1 before any import, matching
the serving target (one CPU core, no GPU). The machine is recorded with every result.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
}
OUT = Path("experiments/results/efficiency")


def _rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10  # bytes on macOS, KiB on Linux


def machine() -> dict[str, Any]:
    info: dict[str, Any] = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        info["cpu"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        info["ram_gb"] = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 2**30
    except (OSError, subprocess.CalledProcessError):
        info["cpu"] = platform.processor()
    return info


def _serialised_mb(model: Any) -> float:
    """Pickled size of a fitted model. Code fetched at run time (the official GRU4Rec) cannot be
    pickled by reference, so its objects are stored as their attributes and torch modules as their
    weights, which is what a deployment would save."""
    import importlib
    import io

    import torch

    try:
        return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) / 2**20
    except pickle.PicklingError:
        pass

    def importable(name: str) -> bool:
        try:
            importlib.import_module(name)
        except ImportError:
            return False
        return True

    class Pickler(pickle.Pickler):
        def reducer_override(self, obj: Any) -> Any:
            if isinstance(obj, torch.nn.Module):
                return dict, (obj.state_dict(),)
            if not isinstance(obj, type) and not importable(type(obj).__module__):
                return dict, (vars(obj),)
            return NotImplemented

    buf = io.BytesIO()
    Pickler(buf, protocol=pickle.HIGHEST_PROTOCOL).dump(model)
    return len(buf.getvalue()) / 2**20


def one(job: dict[str, Any]) -> dict[str, Any]:
    """Runs inside the fresh process."""
    import numpy as np
    import torch

    torch.set_num_threads(1)
    from bench import tasks
    from bench.registry import MODELS
    from mitate.blend import Blend
    from mitate.data import Catalog, Log
    from mitate.ltr import TreeEnsemble
    from mitate.models.base import topk

    z = np.load(job["store"], allow_pickle=False)
    n_users, n_items = int(z["n_users"]), int(z["n_items"])
    log = Log(z["user"], z["item"], z["ts"], z["event"], z["session"], n_users, n_items)
    catalog = Catalog(n_items=log.n_items)
    train = log.until(job["t_test"])
    ev = tasks.next_item(log, job["t_test"], job["t_end"], seed=job["seed"], max_units=job["units"])
    avail = tasks.available_from(log)
    base = _rss_mb()

    def make(name: str) -> Any:
        params = job["params"].get(name, {})
        cls = MODELS[name]
        p = {k: v for k, v in params.items() if k != "epochs" or name in job["neural"]}
        return cls(**p, seed=job["seed"])

    if job["model"] == "mitate":
        booster = TreeEnsemble.load(Path(job["booster"])) if job.get("booster") else None
        sources = [make(s) for s in job["sources"]]
        model: Any = Blend(sources=sources, n_cand=job["n_cand"], booster=booster)
    else:
        model = make(job["model"])
    t0 = time.perf_counter()
    model.fit(train, catalog)
    fit_s = time.perf_counter() - t0
    peak = _rss_mb()
    size_mb = _serialised_mb(model)
    q, ex = ev.queries, ev.exclude
    n = min(len(q), 500)
    lat = []
    for i in range(n + 20):
        j = i % len(q)
        t = time.perf_counter()
        exj = ex[j : j + 1] if ex is not None else None  # OTTO keeps seen items eligible
        model.recommend(q.slice(j, j + 1), 20, exclude=exj, available_from=avail)
        if i >= 20:
            lat.append((time.perf_counter() - t) * 1e3)
    t = time.perf_counter()
    for s in range(0, len(q), 256):
        e = min(len(q), s + 256)
        S = model.score(q.slice(s, e))
        topk(S, 20)
    qps = len(q) / (time.perf_counter() - t)
    a = np.asarray(lat)
    peak = max(peak, _rss_mb())
    return {
        "fit_s": fit_s,
        "peak_rss_mb": peak,
        "rss_delta_mb": peak - base,
        "size_mb": size_mb,
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "qps_batch": qps,
        "n_latency_queries": int(a.size),
        "n_items": log.n_items,
        "n_train_events": len(train),
    }


def run(
    cfg: dict[str, Any],
    dataset: str,
    seed: int,
    results: Path,
    buckets: list[int] | None = None,
) -> None:
    from bench.datasets import load
    from bench.registry import NEURAL
    from bench.run import store_path

    ds = load(dataset)
    dcfg = cfg["datasets"][dataset]
    models = [*dcfg["models"], "mitate"]
    OUT.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **THREAD_ENV}
    for b in cfg["buckets"]:
        if buckets and b["n"] not in buckets:
            continue
        store = f"{dataset}-n{b['n']}-s{seed}"
        sdir = results / dataset / store
        if not (sdir / "tuning").exists():
            print(f"[{store}] no tuning results; run the benchmark first", flush=True)
            continue
        params = {
            p.stem: json.loads(p.read_text())["best_params"]
            for p in (sdir / "tuning").glob("*.json")
        }
        for m in models:
            dest = OUT / dataset / store / f"{m}.json"
            if dest.exists():
                continue
            booster = results / dataset / "_ltr" / "full.json"
            job = {
                "store": str(store_path(ds, b["n"], seed)),
                "t_test": ds.t_test,
                "t_end": ds.t_end,
                "seed": seed,
                "model": m,
                "params": params,
                "neural": sorted(NEURAL),
                "units": cfg["test"]["units"],
                "sources": dcfg["blend_sources"],
                "n_cand": cfg["blend"]["n_cand"],
                "booster": str(booster) if booster.exists() else None,
            }
            cmd = [sys.executable, "-m", "bench.efficiency", "--one", json.dumps(job)]
            p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=6 * 3600)
            if p.returncode != 0:
                print(f"[{store}] {m} failed:\n{p.stderr[-2000:]}", flush=True)
                continue
            res = json.loads(p.stdout.strip().splitlines()[-1])
            res |= {"model": m, "store": store, "threads": 1, "machine": machine()}
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(res, indent=1))
            print(
                f"[{store}] {m:10s} fit {res['fit_s']:8.1f}s  p50 {res['p50_ms']:7.2f}ms  "
                f"p95 {res['p95_ms']:7.2f}ms  rss {res['peak_rss_mb']:7.0f}MB  "
                f"size {res['size_mb']:7.1f}MB",
                flush=True,
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--dataset", default="otto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--buckets", nargs="*", type=int, help="restrict to these bucket sizes")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.one:
        print(json.dumps(one(json.loads(a.one))))
        return
    import yaml

    cfg = yaml.safe_load(Path(a.config).read_text())
    run(cfg, a.dataset, a.seed, Path(cfg.get("results_dir", "experiments/results")), a.buckets)


if __name__ == "__main__":
    main()
