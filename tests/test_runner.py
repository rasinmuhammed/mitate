"""End-to-end runner tests on a synthetic marketplace (also the CI benchmark smoke run)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import bench.run as run
from bench import tasks
from bench.leakage import perturb_after
from bench.simulator import Store
from tests.synthetic import marketplace

CFG = {
    "seeds": [0],
    "buckets": [{"n": 2000, "items": 80}, {"n": 6000, "items": 200}],
    "tuning": {"trials": 2, "val_units": 150, "neural_trial_budget_s": 20},
    "test": {"units": 200, "k": 20},
    "blend": {
        "n_cand": 10,
        "ltr_val_units": 100,
        "ltr_trials": 2,
        "returning_rail": 10,
        "variants": [
            {"name": "full"},
            {"name": "rrf", "rrf": True},
            {"name": "no_volume", "drop_groups": ["store"]},
        ],
    },
}
DCFG = {
    "task": "next_item",
    "exclude_seen": False,
    "models": ["pop", "recpop", "itemknn", "pctm", "vstan"],
    "blend_sources": ["recpop", "itemknn", "pctm"],
    "returning": True,
}


@pytest.fixture()
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(run, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(run, "STORES", tmp_path / "stores")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_smoke_both_stages(isolated: Path) -> None:
    ds = marketplace()
    stores = run.stage1(ds, CFG, DCFG, CFG["buckets"], CFG["seeds"])
    assert len(stores) == 2
    run.stage2(ds, CFG, DCFG, stores)
    for st in stores:
        d = isolated / "results" / "synth" / st.name / "test"
        for name in [
            *DCFG["models"],
            "mitate_full",
            "mitate_rrf",
            "mitate_no_volume",
            "mitate_no_itemknn",
            "mitate_full_repeat",
        ]:
            summary = json.loads((d / f"{name}.json").read_text())["summary"]
            assert 0.0 <= summary["ndcg@10"] <= 1.0, name


def _perturbed(st: Store, t: int) -> Store:
    log = perturb_after(st.log, t, seed=11)
    return Store(
        st.dataset,
        st.bucket,
        st.seed,
        log,
        st.catalog,
        st.t_val,
        st.t_test,
        st.t_end,
        st.item_raw,
        st.stats,
    )


def test_runner_never_sees_the_future(isolated: Path) -> None:
    """Replace every event at/after t_test with noise; validation outputs and test predictions
    (for the same fixed test queries) must not change at all."""
    ds = marketplace(seed=3)
    st = run.get_store(ds, 4000, 120, 0, run.cobrowse_graph(ds, isolated / "g.parquet"))
    assert st is not None
    outs = []
    for k, store in enumerate((st, _perturbed(st, st.t_test))):
        out = isolated / f"r{k}"
        base = run.StoreRun(st, out, CFG, DCFG)  # eval sets from the clean log
        r = run.StoreRun(store, out, CFG, DCFG, val=base.val, test=base.test)
        for name in ("recpop", "itemknn", "pctm"):
            r.run_model(name)
        r.ltr_groups()
        outs.append(out)
    for name in ("recpop", "itemknn", "pctm"):
        a = json.loads((outs[0] / "tuning" / f"{name}.json").read_text())
        b = json.loads((outs[1] / "tuning" / f"{name}.json").read_text())
        assert a["trials"] == b["trials"], f"{name}: tuning saw the future"
        ra = np.load(outs[0] / "test" / f"{name}.npz")["recs"]
        rb = np.load(outs[1] / "test" / f"{name}.npz")["recs"]
        np.testing.assert_array_equal(ra, rb, err_msg=f"{name}: test model saw the future")
    za, zb = np.load(outs[0] / "ltr_val.npz"), np.load(outs[1] / "ltr_val.npz")
    for key in ("X", "y", "nom", "groups"):
        np.testing.assert_array_equal(za[key], zb[key])


def test_mitate_never_sees_the_future(isolated: Path) -> None:
    """Stage 2 end to end: with every event at/after t_test replaced by noise, every Mitate
    variant must rank the same fixed test queries identically (rankers trained on the clean
    store's validation groups, which the test above shows are themselves future-free)."""
    ds = marketplace(seed=3)
    st = run.get_store(ds, 4000, 120, 0, run.cobrowse_graph(ds, isolated / "g.parquet"))
    assert st is not None
    clean = run.StoreRun(st, isolated / "clean", CFG, DCFG)
    for name in DCFG["blend_sources"]:
        clean.run_model(name)
    clean.ltr_groups()
    pools = [(st, np.load(isolated / "clean" / "ltr_val.npz"))]
    snames = [str(x) for x in pools[0][1]["source_names"]]
    boosters = run.train_rankers(ds, CFG, pools)
    todo = run.variants(CFG, snames)
    boosters["full_repeat"] = boosters["full"]
    todo.append((run.Variant("full_repeat"), False))
    outs = []
    for k, store in enumerate((st, _perturbed(st, st.t_test))):
        out = isolated / f"m{k}"
        shutil.copytree(isolated / "clean" / "tuning", out / "tuning")
        r = run.StoreRun(store, out, CFG, DCFG, val=clean.val, test=clean.test)
        run.score_mitate(r, boosters, todo)
        outs.append(out)
    for v, _ in todo:
        a = np.load(outs[0] / "test" / f"mitate_{v.name}.npz")["recs"]
        b = np.load(outs[1] / "test" / f"mitate_{v.name}.npz")["recs"]
        np.testing.assert_array_equal(a, b, err_msg=f"mitate_{v.name} saw the future")


def test_scale_rules_share_tuning_and_skip_models(isolated: Path) -> None:
    """Large-bucket rules (ledger D18): the first store tunes with the rule's trial count, later
    stores reuse its configuration with one validation trial of their own, skipped models never
    run, and buckets below the rule keep the default protocol."""
    rule = {"from_n": 6000, "trials": 2, "shared_tuning": True, "skip": ["vstan"]}
    cfg = CFG | {"seeds": [0, 1], "tuning": CFG["tuning"] | {"trials": 3, "scale": [rule]}}
    dcfg = DCFG | {"models": ["itemknn", "pctm", "vstan"]}
    assert run.scale_rule(cfg, 2000) == {}
    assert run.scale_rule(cfg, 10**6)["trials"] == 2
    stores = run.stage1(marketplace(), cfg, dcfg, cfg["buckets"], cfg["seeds"])
    root = isolated / "results" / "synth"
    small = [st for st in stores if st.bucket == 2000]
    large = sorted((st for st in stores if st.bucket == 6000), key=lambda st: st.seed)
    assert len(large) == 2
    for st in small:
        t = json.loads((root / st.name / "tuning" / "itemknn.json").read_text())
        assert t["n_trials"] == 3 and t["shared_from"] is None
        assert (root / st.name / "test" / "vstan.json").exists()
    first, second = (
        json.loads((root / st.name / "tuning" / "itemknn.json").read_text()) for st in large
    )
    assert first["n_trials"] == 2 and first["shared_from"] is None
    assert second["n_trials"] == 1 and second["shared_from"] == large[0].name
    assert second["best_params"] == first["best_params"]
    for st in large:
        assert not (root / st.name / "test" / "vstan.json").exists()


def test_unevaluable_stores_are_replaced_and_recorded(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ledger D24: a seed whose store has no validation or test query is passed over for the
    next spare seed; the manifest records it, keeps entries across runs, and drives stage 2."""
    real = run.eval_sets

    def eval_sets(st: Store, *a: Any) -> tuple[tasks.EvalSet, tasks.EvalSet]:
        if st.seed == 0:
            raise tasks.EmptyWindowError(f"{st.name}: no validation query")
        return real(st, *a)

    monkeypatch.setattr(run, "eval_sets", eval_sets)
    ds, dcfg = marketplace(), DCFG | {"models": ["pop"]}
    small, large = CFG["buckets"]
    stores = run.stage1(ds, CFG, dcfg, [small], [0, 1], spare=2)
    assert [st.seed for st in stores] == [1, 2]
    run.stage1(ds, CFG, dcfg, [large], [1], spare=0)
    rows = json.loads((isolated / "results" / "synth" / "stores.json").read_text())
    assert [(r["bucket"], r["seed"], "skipped" in r) for r in rows] == [
        (2000, 0, True),
        (2000, 1, False),
        (2000, 2, False),
        (6000, 1, False),
    ]
    assert not (isolated / "results" / "synth" / "synth-n2000-s0").exists()
    again = run.recorded_stores(ds, CFG["buckets"])
    assert [(st.bucket, st.seed) for st in again] == [(2000, 1), (2000, 2), (6000, 1)]
