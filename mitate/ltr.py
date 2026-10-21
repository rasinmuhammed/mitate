"""LambdaMART training (LightGBM, in an isolated process) and dependency-free scoring.

Why two halves: on macOS, LightGBM and PyTorch each load their own OpenMP runtime, and the two
cannot coexist in one process (segfaults and deadlocks, reproduced in this repo). Training therefore
runs in a fresh interpreter that never imports torch, and scoring uses ``TreeEnsemble``, a
vectorised NumPy evaluator of LightGBM's own model dump. Serving then needs neither LightGBM nor
OpenMP, and its output is checked against ``lightgbm.Booster.predict`` in the test suite.

CLI (used by the benchmark runner):  python -m mitate.ltr <job.npz> <model_out.json>
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ZERO_EPS = 1e-35  # LightGBM's kZeroThreshold


@dataclass(frozen=True)
class _Tree:
    feature: NDArray[np.int32]
    threshold: NDArray[np.float64]
    left: NDArray[np.int32]  # child index; negative = ~leaf_index
    right: NDArray[np.int32]
    default_left: NDArray[np.bool_]
    missing: NDArray[np.int8]  # 0 None, 1 Zero, 2 NaN
    leaf_value: NDArray[np.float64]
    shrinkage_applied: bool


class TreeEnsemble:
    """Evaluates a LightGBM numerical-split model (``Booster.dump_model()``) with NumPy."""

    def __init__(self, dump: dict[str, Any]) -> None:
        self.feature_names: list[str] = dump["feature_names"]
        self.trees = [self._parse(t["tree_structure"]) for t in dump["tree_info"]]
        self.dump = dump

    @staticmethod
    def _parse(root: dict[str, Any]) -> _Tree:
        if "leaf_value" in root and "split_feature" not in root:  # single-leaf tree
            return _Tree(
                np.zeros(0, np.int32),
                np.zeros(0),
                np.zeros(0, np.int32),
                np.zeros(0, np.int32),
                np.zeros(0, bool),
                np.zeros(0, np.int8),
                np.array([root["leaf_value"]], np.float64),
                True,
            )
        feats: list[int]
        feats, thr, lc, rc, dl, mt = [], [], [], [], [], []
        leaves: list[float] = []
        mtype = {"None": 0, "Zero": 1, "NaN": 2}

        def visit(node: dict[str, Any]) -> int:
            if "split_feature" not in node:
                leaves.append(float(node["leaf_value"]))
                return ~(len(leaves) - 1)
            if node.get("decision_type", "<=") != "<=":
                raise ValueError("categorical splits are not supported")
            i = len(feats)
            feats.append(int(node["split_feature"]))
            thr.append(float(node["threshold"]))
            dl.append(bool(node["default_left"]))
            mt.append(mtype[node.get("missing_type", "None")])
            lc.append(0)
            rc.append(0)
            lc[i] = visit(node["left_child"])
            rc[i] = visit(node["right_child"])
            return i

        visit(root)
        return _Tree(
            np.asarray(feats, np.int32),
            np.asarray(thr),
            np.asarray(lc, np.int32),
            np.asarray(rc, np.int32),
            np.asarray(dl, bool),
            np.asarray(mt, np.int8),
            np.asarray(leaves),
            True,
        )

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.float64]:
        X = np.asarray(X, np.float64)
        out = np.zeros(X.shape[0])
        rows = np.arange(X.shape[0])
        for t in self.trees:
            if t.feature.size == 0:
                out += t.leaf_value[0]
                continue
            node = np.zeros(X.shape[0], np.int64)
            active = np.ones(X.shape[0], bool)
            while active.any():
                r = rows[active]
                nd = node[r]
                v = X[r, t.feature[nd]]
                mt = t.missing[nd]
                is_nan = np.isnan(v)
                # LightGBM: with missing type None, NaN is treated as 0.0
                v = np.where(is_nan & (mt != 2), 0.0, v)
                missing = (is_nan & (mt == 2)) | ((mt == 1) & (np.abs(v) <= ZERO_EPS))
                go_left = np.where(missing, t.default_left[nd], v <= t.threshold[nd])
                nxt = np.where(go_left, t.left[nd], t.right[nd])
                node[r] = nxt
                active[r] = nxt >= 0
            out += t.leaf_value[~node]
        return out

    @classmethod
    def load(cls, path: Path) -> TreeEnsemble:
        return cls(json.loads(Path(path).read_text()))


# --------------------------------------------------------------------------- training
def _train(job: dict[str, Any]) -> dict[str, Any]:
    """Runs inside the isolated process: optional Optuna tuning on a validation split, then a
    final fit on all groups. Returns the model dump."""
    import lightgbm as lgb
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seed = int(job["seed"])
    base = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [10],
        "verbosity": -1,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 4,
        "seed": seed,
        "lambdarank_truncation_level": 20,
    }
    names = list(job["feature_names"])

    def ds(
        X: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]],
        g: NDArray[np.integer[Any]],
        ref: Any = None,
    ) -> Any:
        return lgb.Dataset(X, y, group=g, feature_name=names, reference=ref, free_raw_data=False)

    best: dict[str, Any] = {"num_boost_round": 300}
    if int(job.get("trials", 0)) > 0 and job["Xva"].shape[0]:
        dtr = ds(job["Xtr"], job["ytr"], job["gtr"])
        dva = ds(job["Xva"], job["yva"], job["gva"], dtr)

        def objective(trial: Any) -> float:
            p = base | {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 500, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            }
            b = lgb.train(
                p,
                dtr,
                num_boost_round=1000,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            trial.set_user_attr("rounds", int(b.best_iteration or b.current_iteration()))
            return float(b.best_score["valid_0"]["ndcg@10"])

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True)
        )
        study.optimize(objective, n_trials=int(job["trials"]))
        best = dict(study.best_params) | {
            "num_boost_round": study.best_trial.user_attrs["rounds"],
            "val_ndcg@10": study.best_value,
        }
    params = base | {k: v for k, v in best.items() if k not in ("num_boost_round", "val_ndcg@10")}
    final = lgb.train(
        params, ds(job["X"], job["y"], job["g"]), num_boost_round=int(best["num_boost_round"])
    )
    dump = final.dump_model()
    dump["mitate_tuning"] = best
    return dump


def train_isolated(
    path_job: Path,
    path_model: Path,
    X: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    g: NDArray[np.integer[Any]],
    feature_names: list[str],
    seed: int = 0,
    trials: int = 0,
    valid_split: tuple[NDArray[Any], ...] | None = None,
) -> TreeEnsemble:
    """Write the job, train in a fresh interpreter without torch, load the dump for scoring.
    ``valid_split`` = (Xtr, ytr, gtr, Xva, yva, gva) enables tuning with ``trials`` trials."""
    empty = (np.zeros((0, X.shape[1]), np.float32), np.zeros(0), np.zeros(0, np.int64))
    tr = valid_split[:3] if valid_split else empty
    va = valid_split[3:] if valid_split else empty
    np.savez(
        path_job,
        X=X,
        y=y,
        g=g,
        Xtr=tr[0],
        ytr=tr[1],
        gtr=tr[2],
        Xva=va[0],
        yva=va[1],
        gva=va[2],
        feature_names=np.array(feature_names),
        seed=seed,
        trials=trials,
    )
    subprocess.run([sys.executable, "-m", "mitate.ltr", str(path_job), str(path_model)], check=True)
    return TreeEnsemble.load(path_model)


def _main(job_path: str, out: str) -> None:
    z = np.load(job_path, allow_pickle=False)
    job = {k: z[k] for k in z.files}
    job["feature_names"] = [str(s) for s in job["feature_names"]]
    dump = _train(job)
    Path(out).write_text(json.dumps(dump))


if __name__ == "__main__":
    if "torch" in sys.modules:
        raise RuntimeError("mitate.ltr must run in a process without torch")
    _main(sys.argv[1], sys.argv[2])
