from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Self

import numpy as np
from implicit.recommender_base import ModelFitError

from bench import tasks
from bench.tuning import tune
from mitate.data import Catalog, Log, Queries
from mitate.models.base import F32, Recommender
from tests.test_models_smoke import chain_log


@dataclass
class Fragile(Recommender):
    """Ranks by item id; training diverges whenever the learning rate is above one."""

    name: ClassVar[str] = "fragile"
    lr: float = 0.1
    n_items: int = 0

    def fit(self, log: Log, catalog: Catalog) -> Self:
        if self.lr > 1:
            raise ModelFitError("NaN encountered in factors")
        self.n_items = catalog.n_items
        return self

    def score(self, q: Queries) -> F32:
        return np.tile(-np.arange(self.n_items, dtype=np.float32), (len(q), 1))

    @classmethod
    def space(cls, trial: Any) -> dict[str, Any]:
        return {"lr": trial.suggest_categorical("lr", [0.1, 10.0])}


def test_a_diverging_configuration_is_a_failed_trial_not_a_crash() -> None:
    log = chain_log()
    t = int(log.ts.max()) // 2
    val = tasks.next_item(log, t, int(log.ts.max()) + 1, target="last")
    res = tune(Fragile, {}, log.until(t), Catalog(n_items=20), val, n_trials=8, seed=0)
    failed = [x for x in res.trials if x["params"]["lr"] > 1]
    assert failed and all(np.isnan(x["value"]) and "NaN" in x["error"] for x in failed)
    assert res.best_params == {"lr": 0.1} and np.isfinite(res.best_value)
