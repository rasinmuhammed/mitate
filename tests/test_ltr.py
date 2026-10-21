"""The NumPy tree evaluator must reproduce LightGBM's predictions (incl. missing values)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from mitate.ltr import TreeEnsemble, train_isolated

SCRIPT = """
import json, sys, numpy as np, lightgbm as lgb
rng = np.random.default_rng(0)
X = rng.normal(size=(3000, 8)); X[rng.random(X.shape) < 0.2] = np.nan; X[:, 3] = 0.0 * X[:, 3]
y = ((np.nan_to_num(X[:, 0]) + rng.normal(0, .5, 3000)) > 0.5).astype(float)
b = lgb.train({"objective": "lambdarank", "verbosity": -1, "num_leaves": 15, "seed": 0,
               "min_data_in_leaf": 5}, lgb.Dataset(X, y, group=[30] * 100), 40)
np.save(sys.argv[1], X); np.save(sys.argv[2], b.predict(X))
open(sys.argv[3], "w").write(json.dumps(b.dump_model()))
"""


def test_numpy_evaluator_matches_lightgbm(tmp_path: Path) -> None:
    px, pp, pm = tmp_path / "X.npy", tmp_path / "p.npy", tmp_path / "m.json"
    subprocess.run([sys.executable, "-c", SCRIPT, str(px), str(pp), str(pm)], check=True)
    ens = TreeEnsemble(json.loads(pm.read_text()))
    np.testing.assert_allclose(ens.predict(np.load(px)), np.load(pp), rtol=1e-12, atol=1e-12)


def test_isolated_training_round_trip(tmp_path: Path) -> None:
    import torch  # noqa: F401  # the parent may hold torch; training must still work

    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 5)).astype(np.float32)
    y = (X[:, 1] > 0.8).astype(np.float32)
    g = np.full(100, 20)
    ens = train_isolated(
        tmp_path / "job.npz",
        tmp_path / "m.json",
        X,
        y,
        g,
        [f"f{i}" for i in range(5)],
        trials=2,
        valid_split=(X[:1600], y[:1600], g[:80], X[1600:], y[1600:], g[80:]),
    )
    s = ens.predict(X)
    assert np.corrcoef(s, X[:, 1])[0, 1] > 0.5
