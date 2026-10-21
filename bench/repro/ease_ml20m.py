"""Gate 2: reproduce EASE on ML-20M under the VAE-CF strong-generalisation protocol.

Target (Steck, WWW 2019, Table 2): Recall@20 0.391, Recall@50 0.521, NDCG@100 0.420.
Preprocessing replicates Liang et al. (WWW 2018) exactly, including their numpy RNG calls:
ratings > 3.5, users with >= 5 ratings, 10k validation and 10k test users drawn with
np.random.seed(98765), items restricted to those seen by training users, and a random 80/20
fold-in/held-out split per evaluation user (again seeded 98765, users in ascending id order,
each user's rows in file order).

The regularisation strength is chosen on the validation users only; the test users are scored
once with the chosen value.

Usage: python -m bench.repro.ease_ml20m data/raw/ml-20m/ml-20m/ratings.csv [--lambda L]

``--lambda`` skips the validation sweep and runs only the test with a lambda already chosen on
validation (the sweep's log is kept in experiments/results/repro/).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import scipy.sparse as sp
from numpy.typing import NDArray

from mitate.metrics import ranking_metrics
from mitate.models.ease import ease_scores, ease_weights

SEED = 98765
N_HELDOUT = 10_000
LAMBDAS = (100.0, 200.0, 300.0, 500.0, 800.0, 1200.0, 2000.0)


def liang_split(path: Path) -> dict[str, sp.csr_matrix]:
    df = pl.read_csv(path, schema_overrides={"userId": pl.Int64, "movieId": pl.Int64})
    df = df.with_row_index("row").filter(pl.col("rating") > 3.5)
    counts = df.group_by("userId").len()
    keep = counts.filter(pl.col("len") >= 5)["userId"]
    df = df.filter(pl.col("userId").is_in(keep.implode()))

    unique_uid = np.sort(df["userId"].unique().to_numpy())
    np.random.seed(SEED)
    unique_uid = unique_uid[np.random.permutation(unique_uid.size)]
    n = unique_uid.size
    tr_users = unique_uid[: n - 2 * N_HELDOUT]
    vd_users = unique_uid[n - 2 * N_HELDOUT : n - N_HELDOUT]
    te_users = unique_uid[n - N_HELDOUT :]

    train = df.filter(pl.col("userId").is_in(pl.Series(tr_users).implode()))
    # pd.unique keeps order of first appearance in file order
    unique_sid = train.sort("row")["movieId"].unique(maintain_order=True).to_numpy()
    sid2id = {s: i for i, s in enumerate(unique_sid)}
    n_items = unique_sid.size

    def to_csr(frame: pl.DataFrame, users: NDArray[np.int64]) -> sp.csr_matrix:
        uid2row = {u: i for i, u in enumerate(users)}
        r = np.fromiter((uid2row[u] for u in frame["userId"].to_numpy()), np.int64)
        c = np.fromiter((sid2id[s] for s in frame["movieId"].to_numpy()), np.int64)
        return sp.csr_matrix((np.ones(r.size), (r, c)), shape=(len(users), n_items))

    def heldout(
        users: NDArray[np.int64],
    ) -> tuple[sp.csr_matrix, sp.csr_matrix, NDArray[np.int64]]:
        plays = df.filter(
            pl.col("userId").is_in(pl.Series(users).implode())
            & pl.col("movieId").is_in(pl.Series(unique_sid).implode())
        ).sort(["userId", "row"])
        uids = plays["userId"].to_numpy()
        is_te = np.zeros(uids.size, dtype=bool)
        np.random.seed(SEED)
        bounds = np.flatnonzero(np.diff(uids)) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [uids.size]])
        for s, e in zip(starts, ends, strict=True):
            m = e - s
            if m >= 5:
                pick = np.random.choice(m, size=int(0.2 * m), replace=False).astype("int64")
                is_te[s + pick] = True
        # evaluation users are those that kept at least one row (all of them after filtering)
        eval_users = np.unique(uids)
        mask = pl.Series(is_te)
        return (
            to_csr(plays.filter(~mask), eval_users),
            to_csr(plays.filter(mask), eval_users),
            eval_users,
        )

    vtr, vte, _ = heldout(vd_users)
    ttr, tte, _ = heldout(te_users)
    return {
        "train": to_csr(train, tr_users),
        "val_tr": vtr,
        "val_te": vte,
        "test_tr": ttr,
        "test_te": tte,
    }


def evaluate(
    B: NDArray[np.floating[Any]], fold_in: sp.csr_matrix, heldout: sp.csr_matrix
) -> dict[str, float]:
    keep = np.diff(heldout.indptr) > 0
    fold_in, heldout = fold_in[keep], heldout[keep]
    recs = []
    for s in range(0, fold_in.shape[0], 2000):
        S = ease_scores(fold_in[s : s + 2000], B)
        Xb = fold_in[s : s + 2000]
        S[Xb.nonzero()] = -np.inf  # fold-in items are never recommended
        top = np.argpartition(-S, 100, axis=1)[:, :100]
        order = np.argsort(-np.take_along_axis(S, top, 1), axis=1)
        recs.append(np.take_along_axis(top, order, 1))
    R = np.vstack(recs)
    m = ranking_metrics(R, heldout, ks=(20, 50, 100))
    return {
        "recall_capped@20": float(np.nanmean(m[20].recall_capped)),
        "recall_capped@50": float(np.nanmean(m[50].recall_capped)),
        "ndcg@100": float(np.nanmean(m[100].ndcg)),
        "n_users": int(R.shape[0]),
        "se_ndcg@100": float(np.nanstd(m[100].ndcg) / np.sqrt(R.shape[0])),
    }


def main(path: str, fixed_lambda: float | None = None) -> None:
    t0 = time.time()
    d = liang_split(Path(path))
    X = d["train"]
    print(
        f"train users {X.shape[0]}, items {X.shape[1]}, nnz {X.nnz}, "
        f"val {d['val_tr'].shape[0]}, test {d['test_tr'].shape[0]} ({time.time() - t0:.0f}s)"
    )
    val: dict[float, dict[str, float]] = {}
    for lam in () if fixed_lambda is not None else LAMBDAS:
        t = time.time()
        B = ease_weights(X, lam, dtype=np.float64)
        val[lam] = evaluate(B, d["val_tr"], d["val_te"])
        print(f"lambda={lam:>6}: {val[lam]}  ({time.time() - t:.0f}s)", flush=True)
        del B
    best = fixed_lambda if fixed_lambda is not None else max(val, key=lambda k: val[k]["ndcg@100"])
    B = ease_weights(X, best, dtype=np.float64)
    test = evaluate(B, d["test_tr"], d["test_te"])
    out = {
        "protocol": "Liang et al. 2018 strong generalisation, ML-20M, seed 98765",
        "data_stats": {"train_users": X.shape[0], "items": X.shape[1], "train_nnz": int(X.nnz)},
        "validation": {str(k): v for k, v in val.items()} or "see ease_ml20m_validation_sweep.log",
        "chosen_lambda": best,
        "test": test,
        "published": {"recall_capped@20": 0.391, "recall_capped@50": 0.521, "ndcg@100": 0.420},
        "seconds": round(time.time() - t0, 1),
    }
    dest = Path("experiments/results/repro/ease_ml20m.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["test"], indent=2))


if __name__ == "__main__":
    lam = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--lambda" else None
    main(sys.argv[1], lam)
