"""Misata demo store: a synthetic Indian kitchen-and-home shop for demos only.

    uv run --group demo python -m serve.demo            # builds data/demo, prints keys
    MITATE_DATA_DIR=data/demo MITATE_DEMO_SITE=data/demo/site uv run python -m serve run
    open http://127.0.0.1:8000/demo/

**Everything here is synthetic and is never used for a benchmark number.**

* Entities come from Misata (github.com/rasinmuhammed/misata, MIT): the product table (category,
  stock) and pseudonymous customers (an id and a delivery-region prefix; no names, emails or
  phones, per docs/privacy.md P1). Titles and prices use Misata's per-column custom generators,
  conditioned on the category.
* Behaviour is simulated here, because a recommender demo needs sessions with structure. Each
  shopper has two favourite categories and a price band; a session starts at a popular item in a
  favourite category and then moves to a similar item (same category), a complement (a fixed,
  documented category map), or another favourite. Viewed items are carted with probability
  0.15; carts are ordered with probability 0.5. Returning shoppers re-buy consumables.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from mitate.complements import CategoryGraph, Edge, EdgeStatus
from mitate.engine import EngineConfig
from serve import schemas as S
from serve.db import DB
from serve.plates import material_of, plate_svg
from serve.service import RetrainPolicy, Service

DAY = 86_400
# leaf category -> (price range in rupees, name stems, materials/variants)
CATEGORIES: dict[str, tuple[tuple[int, int], list[str], list[str]]] = {
    "Cookware > Frying Pans": (
        (600, 3500),
        ["Frying Pan", "Skillet"],
        ["Cast Iron", "Tri-ply Steel", "Non-stick", "Hard Anodised"],
    ),
    "Cookware > Kadai & Woks": (
        (700, 4000),
        ["Kadai", "Wok"],
        ["Cast Iron", "Tri-ply Steel", "Hard Anodised", "Brass"],
    ),
    "Cookware > Tawa": (
        (400, 2500),
        ["Dosa Tawa", "Roti Tawa"],
        ["Cast Iron", "Non-stick", "Iron"],
    ),
    "Cookware > Pressure Cookers": (
        (1200, 5000),
        ["Pressure Cooker", "Inner-lid Cooker"],
        ["Steel", "Hard Anodised", "Aluminium"],
    ),
    "Cookware > Lids": ((250, 900), ["Glass Lid", "Steel Lid"], ["24 cm", "26 cm", "28 cm"]),
    "Tools > Knives": (
        (350, 3000),
        ["Chef's Knife", "Paring Knife", "Santoku"],
        ["Carbon Steel", "Stainless", "Damascus"],
    ),
    "Tools > Chopping Boards": (
        (300, 2000),
        ["Chopping Board", "Butcher Block"],
        ["Teak", "Acacia", "Bamboo"],
    ),
    "Tools > Spatulas & Ladles": (
        (150, 700),
        ["Spatula", "Ladle", "Turner"],
        ["Wooden", "Silicone", "Steel"],
    ),
    "Tea & Coffee > Mugs": ((200, 900), ["Mug", "Cup"], ["Stoneware", "Ceramic", "Enamel"]),
    "Tea & Coffee > Coffee": (
        (300, 1200),
        ["Filter Coffee", "Coffee Beans"],
        ["Chikmagalur", "Coorg", "Araku"],
    ),
    "Tea & Coffee > Coffee Filters": (
        (400, 1500),
        ["South Indian Filter", "Pour-over Dripper"],
        ["Steel", "Brass", "Ceramic"],
    ),
    "Tea & Coffee > Tea Kettles": (
        (800, 3000),
        ["Tea Kettle", "Chai Pan"],
        ["Steel", "Copper", "Enamel"],
    ),
    "Storage > Jars & Containers": (
        (200, 1200),
        ["Masala Box", "Storage Jar"],
        ["Steel", "Glass", "Brass"],
    ),
    "Storage > Lunch Boxes": (
        (400, 1800),
        ["Lunch Box", "Tiffin"],
        ["Steel", "Insulated", "Glass"],
    ),
    "Table > Dinner Plates": (
        (300, 2500),
        ["Dinner Plate", "Thali"],
        ["Stoneware", "Steel", "Brass"],
    ),
    "Table > Bowls": ((200, 1500), ["Bowl", "Katori"], ["Stoneware", "Steel", "Brass"]),
}
# the behavioural truth of the simulation: what shoppers actually buy together
COMPLEMENTS = {
    "Cookware > Frying Pans": ["Cookware > Lids", "Tools > Spatulas & Ladles"],
    "Cookware > Kadai & Woks": ["Cookware > Lids", "Tools > Spatulas & Ladles"],
    "Cookware > Tawa": ["Tools > Spatulas & Ladles"],
    "Cookware > Pressure Cookers": ["Storage > Jars & Containers"],
    "Tools > Knives": ["Tools > Chopping Boards"],
    "Tea & Coffee > Mugs": ["Tea & Coffee > Coffee"],
    "Tea & Coffee > Coffee Filters": ["Tea & Coffee > Coffee", "Tea & Coffee > Mugs"],
    "Tea & Coffee > Tea Kettles": ["Tea & Coffee > Mugs"],
    "Table > Dinner Plates": ["Table > Bowls"],
    "Storage > Lunch Boxes": ["Storage > Jars & Containers"],
}
CONSUMABLE = {"Tea & Coffee > Coffee"}
# plausible pairings left for the merchant to review in the dashboard; not part of the simulation
PROPOSED = [
    ("Tea & Coffee > Tea Kettles", "Tea & Coffee > Coffee Filters"),
    ("Tools > Knives", "Tools > Spatulas & Ladles"),
    ("Table > Dinner Plates", "Storage > Lunch Boxes"),
]
# sizes and capacities that tell otherwise identical products apart
SIZES: dict[str, list[str]] = {
    "Cookware > Frying Pans": ["20 cm", "24 cm", "28 cm"],
    "Cookware > Kadai & Woks": ["1.5 L", "2.5 L", "3.5 L"],
    "Cookware > Tawa": ["26 cm", "28 cm", "30 cm"],
    "Cookware > Pressure Cookers": ["3 L", "5 L", "7.5 L"],
    "Cookware > Lids": ["", "with steam vent"],
    "Tools > Knives": ["9 cm", "15 cm", "20 cm"],
    "Tools > Chopping Boards": ["small", "medium", "large"],
    "Tools > Spatulas & Ladles": ["30 cm", "35 cm"],
    "Tea & Coffee > Mugs": ["250 ml", "350 ml"],
    "Tea & Coffee > Coffee": ["250 g", "500 g"],
    "Tea & Coffee > Coffee Filters": ["2 cup", "4 cup", "6 cup"],
    "Tea & Coffee > Tea Kettles": ["1 L", "1.5 L"],
    "Storage > Jars & Containers": ["500 ml", "1 L", "2 L"],
    "Storage > Lunch Boxes": ["2 tier", "3 tier"],
    "Table > Dinner Plates": ["26 cm", "28 cm", "30 cm"],
    "Table > Bowls": ["150 ml", "300 ml"],
}
EDITIONS = ["Everyday", "Heritage", "Studio", "Classic", "Workshop", "Market", "Festive"]
REGIONS = ["560", "400", "110", "600", "700", "500"]


def catalog_and_customers(n_items: int, n_customers: int, seed: int) -> tuple[Any, Any]:
    import misata  # demo dependency group only

    cats = list(CATEGORIES)
    rng = np.random.default_rng(seed)
    schema = misata.from_dict_schema(
        {
            "products": {
                "product_id": {"type": "integer", "primary_key": True},
                "category": {"type": "string", "enum": cats},
                "title": {"type": "string"},
                "price": {"type": "float"},
                "stock": {"type": "integer", "min": 0, "max": 60},
            },
            "customers": {
                "customer_id": {"type": "integer", "primary_key": True},
                "region": {"type": "string", "enum": REGIONS},
            },
        },
        row_count=max(n_items, n_customers),
        seed=seed,
    )
    for t in schema.tables:
        t.row_count = n_items if t.name == "products" else n_customers

    def titles(df: Any, ctx: Any) -> NDArray[np.str_]:
        out = []
        for c in df["category"]:
            _, stems, variants = CATEGORIES[c]
            size = SIZES[c][rng.integers(len(SIZES[c]))]
            name = f"{variants[rng.integers(len(variants))]} {stems[rng.integers(len(stems))]}"
            out.append(f"{name}, {size}" if size else name)
        return np.array(out)

    def prices(df: Any, ctx: Any) -> NDArray[np.float64]:
        lo_hi = np.array([CATEGORIES[c][0] for c in df["category"]], float)
        u = rng.random(len(df))
        p = np.exp(np.log(lo_hi[:, 0]) + u * (np.log(lo_hi[:, 1]) - np.log(lo_hi[:, 0])))
        return cast(NDArray[np.float64], np.round(p / 10) * 10 - 1)  # ₹1,299-style prices

    tables = misata.generate_from_schema(
        schema, custom_generators={"products": {"title": titles, "price": prices}}
    )
    products, customers = tables["products"], tables["customers"]
    # unique titles: a repeated name gets an edition word ("Heritage Brass Kadai, 2.5 L"); the
    # numbered fallback only runs if a name repeats more often than there are editions
    count: dict[str, int] = {}
    unique = []
    for t in products["title"]:
        n = count[t] = count.get(t, 0) + 1
        unique.append(
            t if n == 1 else f"{EDITIONS[n - 2]} {t}" if n - 2 < len(EDITIONS) else f"{t} ({n})"
        )
    products["title"] = unique
    return products, customers


def stem_of(title: str, category: str) -> str:
    _, stems, _ = CATEGORIES[category]
    return next((s for s in sorted(stems, key=len, reverse=True) if s in title), stems[0])


def simulate(
    products: Any, customers: Any, days: int, seed: int, anon_sessions: int
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + 1)
    cats = list(CATEGORIES)
    item_cat = np.array([cats.index(c) for c in products["category"]])
    price = products["price"].to_numpy(float)
    band = np.digitize(np.log(price), np.quantile(np.log(price), [0.33, 0.66]))
    by_cat = [np.flatnonzero(item_cat == c) for c in range(len(cats))]
    # Zipf popularity within each category
    pop = np.zeros(len(price))
    for idx in by_cat:
        pop[idx] = 1.0 / np.arange(1, idx.size + 1) ** 1.1
        pop[idx] = pop[idx][rng.permutation(idx.size)]
    comp = {cats.index(a): [cats.index(b) for b in bs] for a, bs in COMPLEMENTS.items()}
    now = int(time.time())
    t_start = now - days * DAY
    ids = products["product_id"].astype(str).tolist()
    events: list[dict[str, Any]] = []

    def pick(cat: int, pref_band: int, exclude: set[int]) -> int:
        idx = by_cat[cat]
        w = pop[idx] * np.exp(-1.2 * np.abs(band[idx] - pref_band))
        w[np.isin(idx, list(exclude))] = 0
        return int(idx[rng.choice(idx.size, p=w / w.sum())]) if w.sum() else int(idx[0])

    def session(
        shopper: str | None,
        sid: str,
        t: int,
        favs: NDArray[np.int64],
        pref: int,
        bought: list[int],
    ) -> None:
        seen: set[int] = set()
        carted: list[int] = []
        # returning shoppers often re-buy consumables first
        reorder = [i for i in bought if products["category"].iloc[i] in CONSUMABLE]
        cur = (
            reorder[-1]
            if reorder and rng.random() < 0.4
            else pick(int(rng.choice(favs)), pref, seen)
        )
        for _ in range(1 + int(rng.geometric(0.25))):
            seen.add(cur)
            t += int(rng.integers(20, 240))
            e = {
                "session_id": sid,
                "item_id": ids[cur],
                "type": "view",
                "ts": t,
                "consent": "granted",
            }
            if shopper:
                e["shopper_id"] = shopper
            events.append(e)
            if rng.random() < 0.15 or cur in reorder:
                carted.append(cur)
                events.append(e | {"type": "cart", "ts": t + 5})
            r = rng.random()
            c = int(item_cat[cur])
            if r < 0.5:
                nxt_cat = c
            elif r < 0.8 and c in comp:
                nxt_cat = int(rng.choice(comp[c]))
            else:
                nxt_cat = int(rng.choice(favs))
            cur = pick(nxt_cat, pref, seen)
        if carted and rng.random() < 0.5:
            for j, i in enumerate(dict.fromkeys(carted)):
                o = {"session_id": sid, "item_id": ids[i], "type": "order", "ts": t + 60 + j}
                o |= {"value": float(price[i]), "consent": "granted"}
                if shopper:
                    o["shopper_id"] = shopper
                events.append(o)
                bought.append(i)

    n_cat = len(cats)
    for cid in customers["customer_id"].astype(str):
        favs = rng.choice(n_cat, 2, replace=False)
        pref = int(rng.integers(3))
        bought: list[int] = []
        n_sess = int(rng.geometric(1 / 3))
        for s in sorted(rng.integers(t_start, now - 3600, n_sess).tolist()):
            session(f"demo-{cid}", f"demo-{cid}-{s}", s, favs, pref, bought)
    for a in range(anon_sessions):
        favs = rng.choice(n_cat, 2, replace=False)
        t = int(rng.integers(t_start, now - 3600))
        session(None, f"anon-{a}", t, favs, int(rng.integers(3)), [])
    events.sort(key=lambda e: e["ts"])
    return events


def build(
    out: Path, n_items: int, n_customers: int, anon: int, days: int, seed: int
) -> dict[str, Any]:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    products, customers = catalog_and_customers(n_items, n_customers, seed)
    events = simulate(products, customers, days, seed, anon)
    # the demo merchant never retrains mid-load; one fit at the end
    policy = RetrainPolicy(min_new_events=10**12, max_age_s=10**12)
    svc = Service(DB(out / "mitate.sqlite"), out, EngineConfig(), policy, background=False)
    keys = svc.create_merchant("Misata Demo Kitchen (synthetic)")
    mid = keys["merchant_id"]
    site = out / "site"
    shutil.copytree(Path(__file__).parent / "demo_site", site)
    (site / "plates").mkdir()
    for r in products.itertuples():
        svg = plate_svg(str(r.product_id), r.title, r.category, stem_of(r.title, r.category))
        (site / "plates" / f"{r.product_id}.svg").write_text(svg)
    items = [
        S.Item(
            id=str(r.product_id),
            title=r.title,
            category=r.category,
            price=float(r.price),
            stock=float(r.stock),
            description=f"{r.title}. Synthetic product from the Misata demo store.",
            image_url=f"/demo/plates/{r.product_id}.svg",
        )
        for r in products.itertuples()
    ]
    svc.upsert_items(mid, items, replace=True)
    for s in range(0, len(events), 1000):
        svc.ingest(mid, S.EventBatch(events=[S.Event(**e) for e in events[s : s + 1000]]))
    # the demo merchant has reviewed the complement map: approve the true pairs, reject one
    paths = {p: i for i, p in enumerate(svc._category_paths(mid))}
    g = CategoryGraph()
    for a, bs in COMPLEMENTS.items():
        for b in bs:
            g.edges.append(Edge(paths[a], paths[b], EdgeStatus.APPROVED, origin="demo"))
    g.edges.append(
        Edge(paths["Table > Bowls"], paths["Tools > Knives"], EdgeStatus.REJECTED, "demo")
    )
    # a short review queue, so the dashboard's pairing review has something to decide
    for a, b in PROPOSED:
        g.edges.append(Edge(paths[a], paths[b], EdgeStatus.PROPOSED, "the demo setup"))
    svc._save_graph(mid, g)
    # the demo shows its own x-ray: what the engine did for each rail (no personal data)
    svc.put_rules(mid, S.Rules(allow_xray=True))
    info = svc.retrain(mid)
    catalog = [
        {
            "id": str(r.product_id),
            "title": r.title,
            "category": r.category,
            "price": r.price,
            "stock": int(r.stock),
            "material": material_of(r.title),
            "image": f"/demo/plates/{r.product_id}.svg",
        }
        for r in products.itertuples()
    ]
    (site / "products.json").write_text(json.dumps(catalog))
    shoppers = sorted(
        {e["shopper_id"] for e in events if "shopper_id" in e and e["type"] == "order"}
    )
    demo = {"publicKey": keys["public_key"], "shoppers": shoppers[:20]}
    (site / "config.js").write_text(f"window.DEMO = {json.dumps(demo)};\n")
    (out / "keys.json").write_text(json.dumps(keys, indent=1))
    svc.shutdown()
    return {
        "merchant": mid,
        "items": len(items),
        "events": len(events),
        "stage": info.stage,
        "sources": info.active_sources,
        "fit_seconds": round(info.fit_seconds, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo")
    ap.add_argument("--items", type=int, default=600)
    ap.add_argument("--customers", type=int, default=2500)
    ap.add_argument("--anon-sessions", type=int, default=4000)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    res = build(Path(a.out), a.items, a.customers, a.anon_sessions, a.days, a.seed)
    print(json.dumps(res, indent=1))
    print(f"Keys written to {a.out}/keys.json (synthetic demo merchant).")


if __name__ == "__main__":
    main()
