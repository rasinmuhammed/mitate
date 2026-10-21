"""Content fingerprint of a benchmark store, to check two machines sample identical stores.

    uv run python -m bench.fingerprint experiments/configs/benchmark.yaml otto 1000 1

Loads the store from the cache, or samples it if it is not cached, exactly as ``bench.run``
does, then hashes the arrays that define it. The ``.npz`` bytes are not compared, because zip
metadata differs between machines even when the contents do not.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from bench.datasets import load
from bench.run import get_store
from bench.simulator import cobrowse_graph


def fingerprint(arrays: Sequence[NDArray[Any]]) -> str:
    h = hashlib.sha256()
    for x in arrays:
        a = np.ascontiguousarray(x)
        h.update(f"{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("dataset")
    ap.add_argument("n", type=int)
    ap.add_argument("seed", type=int)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    items = next(b["items"] for b in cfg["buckets"] if b["n"] == a.n)
    ds = load(a.dataset)
    graph = None
    if ds.subcatalog == "cobrowse":
        graph = cobrowse_graph(ds, Path("data") / ds.name / f"cobrowse-t{ds.t_test}.parquet")
    st = get_store(ds, a.n, items, a.seed, graph)
    if st is None:
        raise SystemExit("store is infeasible at this size")
    lg = st.log
    arrays: list[NDArray[Any]] = [lg.user, lg.item, lg.ts, lg.event, lg.session, st.item_raw]
    print(f"{ds.name}-n{a.n}-s{a.seed} {fingerprint(arrays)}")


if __name__ == "__main__":
    main()
