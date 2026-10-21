"""Tables and figures for docs/results.md, generated only from stored results.

    uv run python -m bench.report experiments/configs/benchmark.yaml

Reads ``<results>/<dataset>/analysis_<metric>.json`` (from ``bench.analyze``),
``<results>/placements/analysis.json`` (``bench.placements``), ``<results>/online/validation.json``
(``bench.online_validation``), ``experiments/results/efficiency``, the store manifests and the M1
replication (``otto-m1``, ledger D23); writes ``docs/generated/*.md`` and
``docs/figures/crossover_<dataset>.{png,svg}``, then renders ``docs/results.md`` from
``docs/results.template.md``. No number in results.md is typed by hand.

The crossover figure plots a set of models fixed in advance (one per method family, chosen before
results existed), not the per-bucket best of each family, which would bias the curves upward.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

FIGURE_MODELS = [  # (name, label); chosen before any result was seen, except as noted
    ("mitate_full", "Mitate (blend)"),
    ("sasrec_ce", "SASRec (full softmax)"),
    ("vstan", "Session kNN (VSTAN)"),
    ("pctm", "Transition probe (PCTM)"),
    ("ease", "EASE"),
    ("ials", "iALS"),
    ("pop", "Popularity"),
    # added after the H&M results: the strongest H&M baseline up to 30K, and leaving it out of
    # the figure would flatter Mitate there
    ("recpop", "Recent popularity"),
]
DATASET_NAMES = {
    "otto": "OTTO",
    "hm": "H&M",
    "dunnhumby": "Dunnhumby",
    "amazon_grocery": "Amazon Grocery",
}
LABELS = dict(FIGURE_MODELS) | {
    "mitate": "Mitate (blend)",
    "mitate_rrf": "Mitate, rank fusion (no learning)",
    "mitate_no_volume": "Mitate without store-size features",
    "recpop": "Recent popularity",
    "gp_topfreq": "GP-TopFreq",
    "itemknn": "ItemKNN",
    "rp3beta": "RP3beta",
    "bpr": "BPR",
    "markov": "Markov chain",
    "seqrules": "SeqRules",
    "gsasrec": "gSASRec (tuned k, t)",
    "gru4rec": "GRU4Rec",
    "esasrec": "eSASRec",
    "content_knn": "Content kNN",
}


NA = "\u2013"  # en dash for empty table cells

ABLATED = {  # mitate_<variant> -> what that variant removes
    "mitate_rrf": "Learned ranking (rank fusion instead)",
    "mitate_no_volume": "Store-size features",
    "mitate_otto_full": "In-domain ranker (the shipped OTTO ranker instead)",
} | {
    f"mitate_no_{m}": f"Source: {LABELS.get(m, m)}"
    for m in (
        "recpop",
        "itemknn",
        "rp3beta",
        "ease",
        "ials",
        "pctm",
        "seqrules",
        "vstan",
        "sasrec_ce",
    )
}


def num(x: float, spec: str) -> str:
    return format(x, spec) if math.isfinite(x) else NA


def fmt_p(p: float) -> str:
    """A p-value to three significant figures; one below double precision prints as a bound."""
    if not math.isfinite(p):
        return NA
    return "< 1e-300" if p < 1e-300 else f"{p:.3g}"


def size_label(n: int) -> str:
    return f"{n // 1_000_000}M" if n >= 1_000_000 else f"{n // 1000}K"


def fmt_ci(d: dict[str, float]) -> str:
    half = (d["hi"] - d["lo"]) / 2 if math.isfinite(d.get("hi", math.nan)) else math.nan
    return f"{d['mean']:.4f} ± {half:.4f}" if math.isfinite(half) else f"{d['mean']:.4f}"


def model_table(an: dict[str, Any]) -> str:
    buckets = sorted(an["buckets"], key=int)
    found = {m for b in buckets for m in an["buckets"][b]["models"]}
    # Mitate's reduced variants have their own table (ablation_table)
    found = {m for m in found if m == "mitate_full" or not m.startswith("mitate_")}
    models = sorted(found, key=lambda m: (not m.startswith("mitate"), LABELS.get(m, m)))
    head = "| Model | " + " | ".join(size_label(int(b)) for b in buckets) + " |"
    rows = [head, "|---|" + "---:|" * len(buckets)]
    for m in models:
        cells = []
        for b in buckets:
            row = an["buckets"][b]
            d = row["models"].get(m)
            if d is None:
                cells.append(NA)
                continue
            txt = fmt_ci(d)
            if m == row.get("oracle_best_single"):
                txt = f"**{txt}**"
            cells.append(txt)
        rows.append(f"| {LABELS.get(m, m)} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


RETURNING_LABELS = {
    "mitate_full_repeat": "ranker trained here",
    "mitate_otto_full_repeat": "shipped OTTO ranker",
}


def returning_table(an: dict[str, Any]) -> str | None:
    """The returning-customer rail (repurchases first, then the ranker) against the oracle best
    single method; each system is Holm-corrected across store sizes (ledger D27)."""
    buckets = sorted(an["buckets"], key=int)
    systems = [m for m in RETURNING_LABELS if any(m in an["buckets"][b]["models"] for b in buckets)]
    if not systems:
        return None
    rows = [
        "| Store size | Best single (chosen on test) | Ranker alone | Rail | Rail score "
        "| Δ vs best, 95% CI | p (Holm) |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for b in buckets:
        r = an["buckets"][b]
        best = r.get("oracle_best_single")
        bm = r["models"].get(best, {}).get("mean", math.nan) if best else math.nan
        alone = r["models"].get("mitate_full", {}).get("mean", math.nan)
        for m in systems:
            o = r.get("returning", {}).get(m)
            score = r["models"].get(m, {}).get("mean", math.nan)
            diff = (
                f"{o['diff']:+.4f} [{o['ci_low']:+.4f}, {o['ci_high']:+.4f}] | "
                f"{fmt_p(o.get('p_adj_holm', math.nan))}"
                if o
                else f"{NA} | {NA}"
            )
            rows.append(
                f"| {size_label(int(b))} | {LABELS.get(best, best)} {num(bm, '.4f')} | "
                f"{num(alone, '.4f')} | {RETURNING_LABELS[m]} | {num(score, '.4f')} | {diff} |"
            )
    return "\n".join(rows)


def headline_table(an: dict[str, Any]) -> str:
    rows = [
        "| Store size | Stores | Mitate | Best single (chosen on test) | Δ vs best, 95% CI | d_z "
        "| p (Holm) | Stores where Mitate ≥ best | Δ vs validation-selected single, 95% CI |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for b in sorted(an["buckets"], key=int):
        r = an["buckets"][b]
        o, v = r.get("vs_oracle") or {}, r.get("vs_val_selected") or {}
        best = r.get("oracle_best_single")
        mit = r["models"].get("mitate_full", {}).get("mean", math.nan)
        bm = r["models"].get(best, {}).get("mean", math.nan) if best else math.nan
        if not o:
            rows.append(f"| {size_label(int(b))} | {r['n_stores']} |" + f" {NA} |" * 7)
            continue
        rows.append(
            f"| {size_label(int(b))} | {r['n_stores']} | {mit:.4f} | {LABELS.get(best, best)} "
            f"{bm:.4f} | {o['diff']:+.4f} [{o['ci_low']:+.4f}, {o['ci_high']:+.4f}] | "
            f"{o['d_z']:+.2f} | {fmt_p(o.get('p_adj_holm', math.nan))} | "
            f"{int(o['stores_mitate_ge'])}/{o['stores']} | "
            + (f"{v['diff']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]" if v else NA)
            + " |"
        )
    return "\n".join(rows)


def ablation_table(an: dict[str, Any]) -> str:
    """Rows: what removing each piece costs (Mitate minus the reduced variant; positive means the
    piece helps), per store size, with Benjamini-Yekutieli adjusted p-values."""
    rows = an.get("ablations") or []
    if not rows:
        return ""
    buckets = sorted({int(r["bucket"]) for r in rows})
    cell = {(r["variant"], int(r["bucket"])): r for r in rows}
    out = [
        "| Removed | " + " | ".join(size_label(b) for b in buckets) + " |",
        "|---|" + "---:|" * len(buckets),
    ]
    order = list(ABLATED)
    found = {r["variant"] for r in rows}
    for v in sorted(found, key=lambda v: (order.index(v) if v in order else len(order), v)):
        cells = []
        for b in buckets:
            r = cell.get((v, b))
            if r is None:
                cells.append(NA)
                continue
            mark = "" if r["p_adj_by"] < 0.05 else " (n.s.)"
            cells.append(f"{r['diff']:+.4f}{mark}")
        out.append(f"| {ABLATED.get(v, v)} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def store_table(root: Path) -> str:
    """Per bucket: how big the evaluated stores are and how many test shoppers each holds, so the
    reader can see where a bucket's comparisons have little to go on."""
    if not (root / "stores.json").exists():
        return ""
    rows = json.loads((root / "stores.json").read_text())
    units: dict[int, list[int]] = {}
    items: dict[int, list[int]] = {}
    passed: dict[int, int] = {}
    for r in rows:
        b = int(r["bucket"])
        if "skipped" in r:
            passed[b] = passed.get(b, 0) + 1
            continue
        pop = root / r["store"] / "test" / "pop.npz"
        if pop.exists():
            units.setdefault(b, []).append(int(np.unique(np.load(pop)["unit"]).size))
        items.setdefault(b, []).append(int(r["n_items"]))
    out = [
        "| Store size | Stores | Products per store | Test shoppers per store: mean (range) "
        "| Seeds passed over |",
        "|---|---:|---:|---:|---:|",
    ]
    for b in sorted(items):
        u = units.get(b, [])
        shoppers = f"{np.mean(u):.0f} ({min(u)} to {max(u)})" if u else NA
        out.append(
            f"| {size_label(b)} | {len(items[b])} | {np.mean(items[b]):.0f} | {shoppers} | "
            f"{passed.get(b, 0)} |"
        )
    return "\n".join(out)


def replication_table(a: Path, b: Path, metric: str = "ndcg@10") -> str:
    """The same stores and configuration on two machines: mean test metric on each and the
    largest per-store difference, for every model finished on both."""
    if not (a.exists() and b.exists()):
        return ""

    def scores(root: Path) -> dict[tuple[int, str, str], float]:
        out = {}
        for f in root.glob("*/test/*.json"):
            store = f.parent.parent.name
            bucket = int(store.split("-n")[1].split("-")[0])
            out[(bucket, f.stem, store)] = float(json.loads(f.read_text())["summary"][metric])
        return out

    sa, sb = scores(a), scores(b)
    common = sorted(set(sa) & set(sb))
    groups: dict[tuple[int, str], list[tuple[float, float]]] = {}
    for key in common:
        groups.setdefault(key[:2], []).append((sa[key], sb[key]))
    out = [
        f"| Store size | Model | Stores | {a.name} | {b.name} | Δ mean | Largest Δ on one store |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for (bucket, model), pairs in sorted(groups.items()):
        x, y = np.array(pairs).T
        worst = (y - x)[np.argmax(np.abs(y - x))]
        out.append(
            f"| {size_label(bucket)} | {LABELS.get(model, model)} | {len(pairs)} | "
            f"{x.mean():.4f} | {y.mean():.4f} | {y.mean() - x.mean():+.4f} | {worst:+.4f} |"
        )
    return "\n".join(out)


def efficiency_table(root: Path, dataset: str) -> str:
    files = sorted((root / dataset).glob("*/*.json")) if (root / dataset).exists() else []
    if not files:
        return "_Not measured yet._"
    recs = [json.loads(f.read_text()) for f in files]
    stores = sorted({r["store"] for r in recs}, key=lambda s: int(s.split("-n")[1].split("-")[0]))
    rows = [
        "| Store | Model | Items | Fit (s) | Peak RSS (MB) | Model size (MB) | p50 (ms) | p95 (ms) "
        "| Batch throughput (q/s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = {s: i for i, s in enumerate(stores)}
    rows.extend(
        f"| {r['store']} | {LABELS.get(r['model'], r['model'])} | {r['n_items']:,} | "
        f"{r['fit_s']:.1f} | {r['peak_rss_mb']:.0f} | {r['size_mb']:.1f} | "
        f"{r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['qps_batch']:.0f} |"
        for r in sorted(recs, key=lambda r: (order[r["store"]], r["model"]))
    )
    m = recs[0]["machine"]
    ram = m.get("ram_gb", 0)
    rows.append(
        f"\nMachine: {m.get('cpu', '?')}, {ram:.0f} GB RAM, {m.get('platform', '')}; 1 thread."
    )
    return "\n".join(rows)


PLACEMENTS = {  # ground truth -> what the table measures
    "also_view": "Similar items, scored against Amazon 2018 `also_view`",
    "also_buy": "Goes well with, scored against all Amazon 2018 `also_buy` pairs",
    "also_buy_cross": "Goes well with, scored against `also_buy` pairs in different categories",
}
PLACEMENT_LABELS = {
    "mitate": "**Mitate** (shipped engine, store data)",
    "mitate_day_one": "**Mitate, day one** (catalog only)",
    "content_e5-small": "Text similarity, e5-small",
    "content_potion-multi": "Text similarity, potion-multilingual",
    "content_e5-small_other_category": "Text similarity, other categories (e5-small)",
    "content_potion-multi_other_category": "Text similarity, other categories (potion)",
    "pop": "Popularity",
    "same_category_pop": "Same-category popularity",
    "other_category_pop": "Other-category popularity",
    "co_purchase": "Co-purchase (cosine)",
    "itemknn": "ItemKNN (tuned)",
    "ease": "EASE (tuned)",
}


def placement_tables(rows: dict[str, Any]) -> str:
    """Methods x store size (test NDCG@10, mean ± 95% half-interval over stores), then Mitate's
    paired comparisons with the dev-selected and the oracle baseline."""
    sizes = sorted(rows, key=int)
    methods = sorted(
        {m for n in sizes for m in rows[n]["methods"]},
        key=lambda m: (not m.startswith("mitate"), m != "mitate", PLACEMENT_LABELS.get(m, m)),
    )
    head = "| Method | " + " | ".join(size_label(int(n)) for n in sizes) + " |"
    out = [head, "|---|" + "---:|" * len(sizes)]
    best = {n: max(rows[n]["methods"].items(), key=lambda kv: kv[1]["mean"])[0] for n in sizes}
    for m in methods:
        cells = []
        for n in sizes:
            d = rows[n]["methods"].get(m)
            txt = fmt_ci(d) if d else NA
            cells.append(f"**{txt}**" if d and m == best[n] else txt)
        out.append(f"| {PLACEMENT_LABELS.get(m, m)} | " + " | ".join(cells) + " |")
    out.append(
        "| *Test anchors (stores)* | "
        + " | ".join(
            f"*{rows[n]['anchors']:,} ({next(iter(rows[n]['methods'].values()))['stores']})*"
            for n in sizes
        )
        + " |"
    )
    out += [
        "",
        "| Store size | Mitate vs the baseline each store picks on its dev anchors: Δ NDCG@10, "
        "95% CI | p (Holm) | Best baseline on test (oracle) | Mitate vs oracle: Δ, 95% CI "
        "| p (Holm) |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for n in sizes:
        v, o = rows[n]["vs_dev_selected"], rows[n]["vs_oracle"]
        out.append(
            f"| {size_label(int(n))} | {v['diff']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}] "
            f"| {fmt_p(v['p_adj'])} | {PLACEMENT_LABELS.get(o['model'], o['model'])} "
            f"| {o['diff']:+.4f} [{o['ci_low']:+.4f}, {o['ci_high']:+.4f}] | {fmt_p(o['p_adj'])} |"
        )
    return "\n".join(out)


def online_tables(v: dict[str, Any]) -> dict[str, str]:
    ab = [
        "| Regime | Visitors | Estimator | Bias | SD of estimate | 95% CI coverage "
        "| Variance reduction |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    null = [
        "| Regime | Visitors | False-positive rate at zero lift | Tests reported reliable "
        "(n ≥ 355·skew²) | False-positive rate among those |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in v["ab"]:
        for est in ("raw", "cuped"):
            e = r[est]
            ab.append(
                f"| {r['regime']} | {r['n_visitors']:,} | {est} | {e['bias']:+.4f} "
                f"| {e['sd_estimate']:.4f} | {e['ci95_coverage']:.3f} "
                f"| {e['mean_variance_reduction']:.1%} |"
            )
        z = r["null_false_positive_rate"]
        null.append(
            f"| {r['regime']} | {r['n_visitors']:,} | {z['rate']:.3f} "
            f"| {num(z.get('share_reported_reliable', math.nan), '.1%')} "
            f"| {num(z.get('rate_when_reliable', math.nan), '.3f')} |"
        )
    power = [
        "| Regime | Relative MDE | Visitors the calculator asks for | Simulated power at that "
        "size (target 0.80) |",
        "|---|---:|---:|---:|",
    ]
    power += [
        f"| {p['regime']} | {p['rel_mde']:.0%} | {p['visitors_required_formula']:,} "
        f"| {p['simulated_power_at_that_n']:.3f} |"
        for p in v["power"]
    ]
    il = [
        "| Visitors | Power to prefer the better ranker | Wrong direction, significant "
        "| False-positive rate (no difference) | 95% CI coverage (no difference) "
        "| CLT rule satisfied |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    il += [
        f"| {r['n_visitors']:,} | {r['power_detect_better_A']:.3f} "
        f"| {r['wrong_direction_significant']:.3f} | {r['null_false_positive_rate']:.3f} "
        f"| {r['null_ci95_coverage']:.3f} | {r['clt_rule_satisfied']:.1%} |"
        for r in v["interleaving"]
    ]
    return {
        "online_ab": "\n".join(ab),
        "online_null": "\n".join(null),
        "online_power": "\n".join(power),
        "online_interleaving": "\n".join(il),
    }


def render_results(template: Path, generated: Path, out: Path) -> None:
    """``{{name}}`` in the template -> contents of ``generated/name.md`` (or a not-run note)."""
    import re

    def fill(m: re.Match[str]) -> str:
        f = generated / f"{m.group(1)}.md"
        return f.read_text().rstrip() if f.exists() else "_Not run yet._"

    body = re.sub(r"\{\{([\w.-]+)\}\}", fill, template.read_text())
    out.write_text(
        "<!-- Generated by `python -m bench.report` from docs/results.template.md and stored "
        "results. Do not edit by hand. -->\n\n" + body
    )


def crossover_figure(an: dict[str, Any], out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buckets = sorted(an["buckets"], key=int)
    xs = [int(b) for b in buckets]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    palette = [
        *("#8a3b12", "#1f5f8b", "#3b8a5a", "#7a5c99"),
        *("#b07d1a", "#5f6b73", "#9a9a9a", "#c0567a"),
    ]
    for (m, label), color in zip(FIGURE_MODELS, palette, strict=True):
        pts = [(x, an["buckets"][b]["models"].get(m)) for x, b in zip(xs, buckets, strict=True)]
        pts = [(x, d) for x, d in pts if d is not None]
        if not pts:
            continue
        x = [p[0] for p in pts]
        y = [p[1]["mean"] for p in pts]
        lo = [p[1]["lo"] for p in pts]
        hi = [p[1]["hi"] for p in pts]
        lw = 2.6 if m == "mitate_full" else 1.5
        ax.plot(x, y, marker="o", ms=4, lw=lw, color=color, label=label, zorder=3 if lw > 2 else 2)
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0)
    ax.set_xscale("log")
    ax.set_xticks(xs, [size_label(x) for x in xs])
    ax.minorticks_off()
    ax.set_xlabel("Store size: interactions before the test week (log scale)")
    ax.set_ylabel(f"Test {an['metric'].upper()} (mean over 5 stores)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_ylim(bottom=0)  # NDCG cannot be negative; the t-bands of tiny stores can be
    ax.grid(axis="y", color="#e6e2da", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    note = "Bands: 95% t-interval across stores. Mitate-GTS v1 protocol."
    fig.text(0.01, 0.01, note, fontsize=7, color="#6b6862")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    results = Path(cfg.get("results_dir", "experiments/results"))
    gen, figs = Path("docs/generated"), Path("docs/figures")
    gen.mkdir(parents=True, exist_ok=True)
    for ds in cfg["datasets"]:
        for metric in ("ndcg_10", "recall_20"):
            p = results / ds / f"analysis_{metric}.json"
            if not p.exists():
                continue
            an = json.loads(p.read_text())
            (gen / f"{ds}_{metric}_models.md").write_text(model_table(an) + "\n")
            (gen / f"{ds}_{metric}_headline.md").write_text(headline_table(an) + "\n")
            returning = returning_table(an)
            if returning:
                (gen / f"{ds}_{metric}_returning.md").write_text(returning + "\n")
            ablations = ablation_table(an)
            if ablations:
                (gen / f"{ds}_{metric}_ablations.md").write_text(ablations + "\n")
            if metric == "ndcg_10":
                name = DATASET_NAMES.get(ds, ds)
                crossover_figure(
                    an, figs / f"crossover_{ds}", f"Which method wins at each store size? ({name})"
                )
                (gen / f"{ds}_figure.md").write_text(
                    f"![Which method wins at each store size, {name}](figures/crossover_{ds}.svg)\n"
                )
        (gen / f"{ds}_efficiency.md").write_text(
            efficiency_table(Path("experiments/results/efficiency"), ds) + "\n"
        )
        stores = store_table(results / ds)
        if stores:
            (gen / f"{ds}_stores.md").write_text(stores + "\n")
    # buckets that ran on the M1 before moving to Kaggle (ledger D23)
    replication = replication_table(results / "otto-m1", results / "otto")
    if replication:
        (gen / "otto_replication.md").write_text(replication + "\n")
    # the current placement run, and the D19 run kept for the before/after record (ledger D20)
    for tag, folder in (("", "placements"), ("_d19", "placements_d19")):
        placements = results / folder / "analysis.json"
        if placements.exists():
            rep = json.loads(placements.read_text())
            for kind, rows in rep.items():
                if rows:
                    (gen / f"placements{tag}_{kind}.md").write_text(placement_tables(rows) + "\n")
    online = results / "online" / "validation.json"
    if online.exists():
        for name, table in online_tables(json.loads(online.read_text())).items():
            (gen / f"{name}.md").write_text(table + "\n")
    template = Path("docs/results.template.md")
    if template.exists():
        render_results(template, gen, Path("docs/results.md"))
    print(f"wrote {sorted(str(p) for p in gen.iterdir())}")


if __name__ == "__main__":
    main()
