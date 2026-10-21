"""Aggregate stored per-query results into the tables and tests of docs/results.md.

    python -m bench.analyze experiments/results/<dataset> [--metric ndcg@10]

Per store: unit-averaged metric per model. Per bucket: mean, std and a t-interval across the
store seeds. Headline comparison per bucket: Mitate (full) against

* the **oracle best single method**: the single model with the highest mean test metric in that
  bucket, chosen after seeing test results (a harsher bar than any merchant could use), and
* the **validation-selected single method**: per store, the single model with the best
  validation NDCG@10 (what a merchant could actually deploy).

Paired tests run over shoppers pooled across the bucket's stores (each shopper belongs to one
store; pairs are the same shopper under both systems). Holm corrects across buckets
(confirmatory); Benjamini-Yekutieli corrects the full leaderboard and, separately, the ablations
(Mitate against each variant with one piece removed), both exploratory. Where the returning-
customer rail was scored (ledger D27), each version of it is compared with the oracle best single
method and Holm-corrected across buckets as its own family.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mitate.stats import benjamini_yekutieli, holm, paired_test, per_user_mean, t_interval

MITATE = "mitate_full"
# the returning-customer rail, repurchases first (ledger D27): with the ranker trained on this
# dataset's stores, and with the ranker the product ships
RETURNING = ("mitate_full_repeat", "mitate_otto_full_repeat")


def load_units(
    store_dir: Path, metric: str
) -> dict[str, tuple[NDArray[np.integer[Any]], NDArray[np.float64]]]:
    """model -> (unit ids, unit-averaged metric)."""
    out = {}
    for f in sorted((store_dir / "test").glob("*.npz")):
        z = np.load(f)
        key = f"m:{metric}"
        if key not in z.files:
            continue
        out[f.stem] = per_user_mean(z[key], z["unit"])
    return out


def val_scores(store_dir: Path) -> dict[str, float]:
    out = {}
    for f in (store_dir / "tuning").glob("*.json"):
        out[f.stem] = float(json.loads(f.read_text())["best_value"])
    return out


def evaluated_stores(root: Path) -> list[Path]:
    """Store directories with test results; where stage 1 wrote a manifest, only the stores it
    lists as evaluated (a seed passed over under ledger D24 is never analysed)."""
    manifest = root / "stores.json"
    listed = (
        {r["store"] for r in json.loads(manifest.read_text()) if "skipped" not in r}
        if manifest.exists()
        else None
    )
    return sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and (p / "test").exists() and (listed is None or p.name in listed)
    )


def analyse(root: Path, metric: str) -> dict[str, Any]:
    stores = evaluated_stores(root)
    by_bucket: dict[int, list[Path]] = defaultdict(list)
    for s in stores:
        by_bucket[int(s.name.split("-n")[1].split("-")[0])].append(s)
    table: dict[int, dict[str, Any]] = {}
    confirm_p: list[float] = []
    confirm_keys: list[tuple[int, str]] = []
    leaderboard: list[dict[str, Any]] = []
    ablations: list[dict[str, Any]] = []
    returning_p: dict[str, list[float]] = defaultdict(list)
    returning_keys: dict[str, list[int]] = defaultdict(list)
    for bucket in sorted(by_bucket):
        per_model: dict[str, list[float]] = defaultdict(list)
        units: dict[str, dict[str, tuple[NDArray[np.integer[Any]], NDArray[np.float64]]]] = {}
        val_pick: dict[str, str] = {}
        for s in by_bucket[bucket]:
            u = load_units(s, metric)
            units[s.name] = u
            for m, (_, v) in u.items():
                per_model[m].append(float(np.nanmean(v)))
            vs = {m: v for m, v in val_scores(s).items() if not m.startswith("mitate")}
            if vs:
                val_pick[s.name] = max(vs, key=lambda m: vs[m])
        summary = {
            m: dict(zip(("mean", "lo", "hi"), t_interval(v), strict=True))
            | {"std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n_stores": len(v)}
            for m, v in per_model.items()
        }
        singles = {m: d["mean"] for m, d in summary.items() if not m.startswith("mitate")}
        row: dict[str, Any] = {"models": summary, "n_stores": len(by_bucket[bucket])}
        if singles and MITATE in summary:
            oracle = max(singles, key=lambda m: singles[m])
            row["oracle_best_single"] = oracle

            def pooled(
                ref_of: dict[str, str],
                units: dict[str, dict[str, tuple[Any, Any]]] = units,
                system: str = MITATE,
            ) -> dict[str, Any]:
                wins: float
                a_all, b_all, wins = [], [], 0
                for sname, u in units.items():
                    ref = ref_of.get(sname)
                    if system not in u or ref not in u:
                        continue
                    ua, va = u[system]
                    ub, vb = u[ref]
                    _, ia, ib = np.intersect1d(ua, ub, return_indices=True)
                    a_all.append(va[ia])
                    b_all.append(vb[ib])
                    wins += float(np.nanmean(va[ia]) >= np.nanmean(vb[ib]))
                if not a_all:
                    return {}
                r = paired_test(np.concatenate(a_all), np.concatenate(b_all))
                return r.as_dict() | {"stores_mitate_ge": wins, "stores": len(a_all)}

            row["vs_oracle"] = pooled({s: oracle for s in units})
            row["vs_val_selected"] = pooled(val_pick)
            row["val_selected"] = val_pick
            if row["vs_oracle"]:
                confirm_p.append(row["vs_oracle"]["p_ttest"])
                confirm_keys.append((bucket, oracle))
            for m in singles:
                ref = {s: m for s in units}
                res = pooled(ref)
                if res:
                    leaderboard.append({"bucket": bucket, "baseline": m, **res})
            for system in RETURNING:
                res = pooled({s: oracle for s in units}, system=system)
                if res:
                    row.setdefault("returning", {})[system] = res
                    returning_p[system].append(res["p_ttest"])
                    returning_keys[system].append(bucket)
            # Mitate minus each reduced variant: what the removed piece contributes
            for m in summary:
                if m.startswith("mitate_") and m != MITATE and m not in RETURNING:
                    res = pooled({s: m for s in units})
                    if res:
                        ablations.append({"bucket": bucket, "variant": m, **res})
        table[bucket] = row
    for (bucket, _), p in zip(confirm_keys, holm(confirm_p), strict=True):
        table[bucket]["vs_oracle"]["p_adj_holm"] = float(p)
    # each returning-customer system is its own family across buckets, like the headline
    for system, ps in returning_p.items():
        for bucket, p in zip(returning_keys[system], holm(ps), strict=True):
            table[bucket]["returning"][system]["p_adj_holm"] = float(p)
    for row, p in zip(
        leaderboard, benjamini_yekutieli([r["p_ttest"] for r in leaderboard]), strict=True
    ):
        row["p_adj_by"] = float(p)
    for row, p in zip(
        ablations, benjamini_yekutieli([r["p_ttest"] for r in ablations]), strict=True
    ):
        row["p_adj_by"] = float(p)
    return {"metric": metric, "buckets": table, "leaderboard": leaderboard, "ablations": ablations}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--metric", default="ndcg@10")
    a = ap.parse_args()
    res = analyse(Path(a.root), a.metric)
    out = Path(a.root) / f"analysis_{a.metric.replace('@', '_')}.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    for b, row in res["buckets"].items():
        ms = row["models"]
        best = row.get("oracle_best_single")
        mit = ms.get(MITATE, {}).get("mean", float("nan"))
        print(
            f"n={b:>8}  stores={row['n_stores']}  mitate={mit:.4f}  "
            f"best={best}:{ms.get(best, {}).get('mean', float('nan')):.4f}  "
            f"d={row.get('vs_oracle', {}).get('diff', float('nan')):+.4f}  "
            f"p_holm={row.get('vs_oracle', {}).get('p_adj_holm', float('nan')):.3g}"
        )


if __name__ == "__main__":
    main()
