"""Equal-budget hyperparameter tuning.

Every model with a non-empty search space gets the same number of Optuna TPE trials with the same
seeded sampler, and the same objective (validation NDCG@10 on the temporal validation window).
Models with no hyperparameters are evaluated once. Models that train iteratively get a validation
``monitor`` so the epoch count is tuned within each trial instead of costing extra trials.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import optuna
from implicit.recommender_base import ModelFitError

from bench.tasks import EvalSet
from mitate.data import I64, Catalog, Log
from mitate.metrics import ranking_metrics
from mitate.models.base import Recommender

optuna.logging.set_verbosity(logging.WARNING)

Factory = Callable[..., Recommender]


@dataclass
class TuneResult:
    model: str
    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    trials: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0


def validation_score(
    model: Recommender,
    ev: EvalSet,
    k: int = 10,
    metric: str = "ndcg",
    available_from: I64 | None = None,
) -> float:
    recs = model.recommend(ev.queries, k, exclude=ev.exclude, available_from=available_from)
    m = ranking_metrics(recs, ev.targets, ks=(k,))[k]
    return float(np.nanmean(getattr(m, metric)))


def _trial(
    factory: Factory,
    fixed: dict[str, Any],
    params: dict[str, Any],
    train: Log,
    catalog: Catalog,
    val: EvalSet,
    seed: int,
    available_from: I64 | None,
) -> tuple[float, dict[str, Any]]:
    """Fit one configuration and score it on validation; iterative models also report the best
    epoch seen by their validation monitor."""
    m = factory(**fixed, **params, seed=seed)
    if hasattr(m, "monitor"):
        m.monitor = lambda mm: validation_score(mm, val, available_from=available_from)
    try:
        m.fit(train, catalog)
    except (MemoryError, ModelFitError) as e:  # dense EASE over the item cap; diverged factors
        return float("nan"), {"error": str(e)}
    v = validation_score(m, val, available_from=available_from)
    extra = {"epochs": cast(Any, m).best_epoch} if getattr(m, "best_epoch", 0) else {}
    return v, extra


def tune(
    factory: Factory,
    fixed: dict[str, Any],
    train: Log,
    catalog: Catalog,
    val: EvalSet,
    n_trials: int,
    seed: int,
    time_budget_s: float | None = None,
    available_from: I64 | None = None,
) -> TuneResult:
    cls: type[Recommender] = factory if isinstance(factory, type) else type(factory())
    t0 = time.monotonic()
    probe = optuna.trial.FixedTrial({})
    try:
        has_space = bool(cls.space(probe))
    except Exception:  # FixedTrial raises on any suggest -> the space is non-empty
        has_space = True

    def run(params: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        return _trial(factory, fixed, params, train, catalog, val, seed, available_from)

    if not has_space:
        v, extra = run({})
        return TuneResult(
            cls.name, dict(extra), v, 1, [{"params": {}, "value": v}], time.monotonic() - t0
        )

    trials: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = cls.space(trial)
        v, extra = run(params)
        trial.set_user_attr("extra", extra)
        trials.append({"params": params, "value": v, **extra})
        return -1.0 if np.isnan(v) else v

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True)
    )
    study.optimize(objective, n_trials=n_trials, timeout=time_budget_s, gc_after_trial=True)
    best = study.best_trial
    assert best.value is not None  # best_trial is a completed single-objective trial
    params = dict(best.params) | dict(best.user_attrs.get("extra", {}))
    params.pop("error", None)
    return TuneResult(
        cls.name, params, float(best.value), len(study.trials), trials, time.monotonic() - t0
    )


def evaluate_config(
    factory: Factory,
    fixed: dict[str, Any],
    params: dict[str, Any],
    train: Log,
    catalog: Catalog,
    val: EvalSet,
    seed: int,
    available_from: I64 | None = None,
) -> TuneResult:
    """Score a configuration tuned on another store against this store's own validation window.

    The epoch count is not shared: iterative models re-select it here with their monitor, so the
    result carries this store's validation score and epochs, as a one-trial ``tune`` would.
    """
    cls: type[Recommender] = factory if isinstance(factory, type) else type(factory())
    t0 = time.monotonic()
    shared = {k: v for k, v in params.items() if k != "epochs"}
    v, extra = _trial(factory, fixed, shared, train, catalog, val, seed, available_from)
    extra.pop("error", None)
    return TuneResult(
        cls.name,
        shared | extra,
        v,
        1,
        [{"params": shared, "value": v, **extra}],
        time.monotonic() - t0,
    )
