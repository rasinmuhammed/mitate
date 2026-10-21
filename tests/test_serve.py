import json
import logging
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mitate.engine import EngineConfig
from serve.app import create_app
from serve.db import DB
from serve.importers import mitate_csv, shopify_csv, woocommerce_csv
from serve.service import RetrainPolicy, Service

CATS = ["Kitchen > Pans", "Kitchen > Lids", "Kitchen > Knives", "Home > Candles"]


def _items(n: int = 24) -> list[dict]:
    return [
        {
            "id": f"sku{i}",
            "title": f"{CATS[i % 4].split(' > ')[1][:-1]} {i}",
            "category": CATS[i % 4],
            "price": 10.0 + i,
            "stock": 0 if i == 23 else 5,
        }
        for i in range(n)
    ]


@pytest.fixture
def env(tmp_path: Path):
    cfg = EngineConfig(min_behaviour_events=60, embedder="none", sources=("recpop", "itemknn"))
    policy = RetrainPolicy(min_new_events=50)
    svc = Service(DB(tmp_path / "t.sqlite"), tmp_path, cfg, policy, background=False)
    keys = svc.create_merchant("Test Kitchen")
    client = TestClient(create_app(svc))
    with client:
        r = client.post(
            "/v1/catalog",
            json={"items": _items()},
            headers={"X-Mitate-Key": keys["private_key"]},
        )
        assert r.status_code == 200 and r.json()["upserted"] == 24
        yield client, keys, svc


def _pub(keys: dict) -> dict:
    return {"X-Mitate-Key": keys["public_key"]}


def _priv(keys: dict) -> dict:
    return {"X-Mitate-Key": keys["private_key"]}


def _rec(client: TestClient, keys: dict, **body: object) -> dict:
    return client.post("/v1/recommend", json=body, headers=_pub(keys)).json()


def _send(client: TestClient, keys: dict, events: list[dict]) -> dict:
    return client.post("/v1/events", json={"events": events}, headers=_pub(keys)).json()


def _experiment(client: TestClient, keys: dict, **spec: object) -> dict:
    return client.post("/v1/experiments", json=spec, headers=_priv(keys)).json()


def _sessions(n: int, consent: str = "granted", t0: int | None = None) -> list[dict]:
    """Shoppers who view a pan and then its lid (sku0 -> sku1, sku4 -> sku5, ...)."""
    t0 = t0 or int(time.time()) - 3600
    ev = []
    for s in range(n):
        a = 4 * (s % 5)
        for j, it in enumerate((a, a + 1)):
            ev.append(
                {
                    "shopper_id": f"user-{s}",
                    "session_id": f"sess-{s}",
                    "item_id": f"sku{it}",
                    "type": "order" if j else "view",
                    "ts": t0 + 10 * s + j,
                    "consent": consent,
                    "value": 20.0 if j else None,
                }
            )
    return ev


def test_keys_are_scoped(env) -> None:
    client, keys, _ = env
    assert client.post("/v1/catalog", json={"items": []}, headers=_pub(keys)).status_code == 403
    assert client.get("/v1/status", headers={"X-Mitate-Key": "nope"}).status_code == 401
    assert client.get("/v1/status").status_code == 401


def test_personal_fields_are_rejected(env) -> None:
    client, keys, _ = env
    bad = {"events": [{"session_id": "s", "item_id": "sku1", "type": "view", "email": "a@b.c"}]}
    assert client.post("/v1/events", json=bad, headers=_pub(keys)).status_code == 422
    bad2 = {"events": [{"session_id": "s", "item_id": "sku1", "type": "view", "gender": "f"}]}
    assert client.post("/v1/events", json=bad2, headers=_pub(keys)).status_code == 422


def test_only_consented_events_are_stored_and_ids_are_pseudonymised(env) -> None:
    client, keys, svc = env
    r = client.post("/v1/events", json={"events": _sessions(3, "denied")}, headers=_pub(keys))
    assert r.json() == {"accepted": 0, "dropped_no_consent": 6, "dropped_unknown_item": 0}
    r = client.post("/v1/events", json={"events": _sessions(3)}, headers=_pub(keys))
    assert r.json()["accepted"] == 6
    raw = Path(svc.db.path).read_bytes()
    assert b"user-0" not in raw and b"sess-0" not in raw


def test_catalog_only_then_behavioural_after_threshold(env) -> None:
    client, keys, _ = env
    r = _rec(client, keys, placement="similar", item_ids=["sku0"])
    assert r["stage"] == "catalog_only" and r["items"]
    assert all(i["id"] != "sku0" and i["id"] != "sku23" for i in r["items"])  # anchor, no stock
    client.post("/v1/events", json={"events": _sessions(40)}, headers=_pub(keys))
    st = client.get("/v1/status", headers=_priv(keys)).json()
    assert st["model"]["stage"] == "behavioural" and st["events"] == 80
    r = _rec(
        client, keys, placement="next", shopper_id="user-0", session_id="new", consent="granted"
    )
    assert r["personalised"] and len(r["items"]) == 8
    models = client.get("/v1/models", headers=_priv(keys)).json()
    assert models[0]["active"] and models[0]["active_sources"] == ["recpop", "itemknn"]


def test_minor_and_unconsented_requests_are_not_personalised(env) -> None:
    client, keys, _ = env
    client.post("/v1/events", json={"events": _sessions(40)}, headers=_pub(keys))
    for extra in ({"consent": "granted", "age_status": "minor"}, {"consent": "denied"}):
        body = {"placement": "returning", "shopper_id": "user-0", **extra}
        r = client.post("/v1/recommend", json=body, headers=_pub(keys)).json()
        assert not r["personalised"]


def test_complement_reasons_are_counted_facts(env) -> None:
    client, keys, _ = env
    client.post("/v1/events", json={"events": _sessions(40)}, headers=_pub(keys))
    # each pan->lid pair was viewed-then-ordered by 8 shoppers
    r = _rec(client, keys, placement="next", item_ids=["sku0"])
    top = r["items"][0]
    assert top["id"] == "sku1"
    assert top["reason"] == "8 shoppers viewed this right after Pan 0"
    ex = client.get(f"/v1/explain/{r['impression_id']}", headers=_pub(keys)).json()
    assert ex["items"][0]["reason"]["evidence"]["shoppers"] == 8


def test_rules_apply_without_retraining(env) -> None:
    client, keys, _ = env
    client.post("/v1/events", json={"events": _sessions(40)}, headers=_pub(keys))
    version = client.get("/v1/status", headers=_priv(keys)).json()["model"]["version"]
    rules = client.get("/v1/rules", headers=_priv(keys)).json()
    rules |= {"blocked_items": ["sku1"], "pins": {"next": ["sku22"]}}
    assert client.put("/v1/rules", json=rules, headers=_priv(keys)).status_code == 200
    r = _rec(client, keys, placement="next", item_ids=["sku0"])
    ids = [i["id"] for i in r["items"]]
    assert ids[0] == "sku22" and "sku1" not in ids
    assert r["items"][0]["reason"] == "Featured by the store"
    assert client.get("/v1/status", headers=_priv(keys)).json()["model"]["version"] == version


def test_erasure_removes_data_and_suppresses_future_events(env) -> None:
    client, keys, _ = env
    client.post("/v1/events", json={"events": _sessions(40)}, headers=_pub(keys))
    exp = client.get("/v1/shoppers/user-3/export", headers=_priv(keys)).json()
    assert [e["item_id"] for e in exp["events"]] == ["sku12", "sku13"]
    assert client.delete("/v1/shoppers/user-3", headers=_priv(keys)).json()["events_deleted"] == 2
    assert client.get("/v1/shoppers/user-3/export", headers=_priv(keys)).json()["events"] == []
    again = [e for e in _sessions(4) if e["shopper_id"] == "user-3"]
    r = client.post("/v1/events", json={"events": again}, headers=_pub(keys)).json()
    assert r["accepted"] == 0
    st = client.get("/v1/status", headers=_priv(keys)).json()
    assert st["events"] == 78 and st["model"]["n_events"] == 78  # retrained without them


def test_holdout_experiment_reports_rpv_with_ci(env) -> None:
    client, keys, _ = env
    client.post("/v1/events", json={"events": _sessions(40)}, headers=_pub(keys))
    e = _experiment(client, keys, name="h", kind="holdout", holdout_share=0.5)
    arms = {}
    for s in range(60):
        r = _rec(client, keys, placement="next", shopper_id=f"v{s}", consent="granted")
        arms[f"v{s}"] = r["arm"]
        assert (r["items"] == []) == (r["arm"] == "holdout")
    assert 15 < sum(a == "holdout" for a in arms.values()) < 45
    now = int(time.time()) + 1
    buy = {"item_id": "sku2", "type": "order", "ts": now, "value": 30.0, "consent": "granted"}
    _send(client, keys, [{"shopper_id": v, "session_id": f"x{v}", **buy} for v in list(arms)[::2]])
    rep = client.get(f"/v1/experiments/{e['id']}/report", headers=_priv(keys)).json()
    assert sum(rep["units"].values()) == 60
    rpv = rep["revenue_per_visitor"]
    assert rpv["ci_low"] <= rpv["abs_effect"] <= rpv["ci_high"]
    assert "Clicks on recommendations are never counted as lift" in rep["note"]


def test_interleaving_credits_clicks_to_the_team_that_placed_the_item(env) -> None:
    client, keys, svc = env
    client.post("/v1/events", json={"events": _sessions(40)}, headers=_pub(keys))
    e = _experiment(client, keys, name="il", kind="interleave")
    for s in range(30):
        r = _rec(
            client, keys, placement="next", item_ids=["sku0"], shopper_id=f"w{s}", consent="granted"
        )
        assert r["arm"] == "interleave"
        row = svc.db.one("SELECT teams FROM impressions WHERE id = ?", (r["impression_id"],))
        teams = json.loads(row["teams"])
        target = 0 if s < 20 else 1  # 20 visitors click a Mitate slot, 10 a rival slot
        pos = teams.index(target)
        click = {"session_id": f"w{s}", "item_id": r["items"][pos]["id"], "type": "click"}
        _send(client, keys, [click | {"impression_id": r["impression_id"]}])
    rep = client.get(f"/v1/experiments/{e['id']}/report", headers=_priv(keys)).json()
    il = rep["interleaving"]
    assert (il["n_engaged"], il["wins_a"], il["wins_b"]) == (30, 20, 10)
    assert il["delta"] == pytest.approx(20 / 30 - 0.5)


def test_logs_never_contain_ids(env, caplog) -> None:
    client, keys, _ = env
    caplog.set_level(logging.INFO, logger="mitate.serve")
    client.post("/v1/events", json={"events": _sessions(40)}, headers=_pub(keys))
    _rec(client, keys, placement="returning", shopper_id="user-7", consent="granted")
    text = caplog.text
    assert "/v1/recommend" in text and "user-7" not in text and "sess-7" not in text


# ---------------------------------------------------------------------------- importers
def test_mitate_csv_reports_bad_rows() -> None:
    csv = (
        "id,title,category,price,stock,regions\n"
        "a,Pan,Kitchen > Pans,10,3,56;40\n"
        "b,Lid,,x,,\n"
        ",,,,,\n"
    )
    items, errors = mitate_csv(csv)
    assert [i.id for i in items] == ["a", "b"] and items[0].regions == ["56", "40"]
    assert items[1].price is None and len(errors) == 1


def test_shopify_export_groups_variants() -> None:
    csv = (
        "Handle,Title,Body (HTML),Vendor,Product Category,Type,Published,Variant Price,"
        "Variant Inventory Tracker,Variant Inventory Qty,Status\n"
        "pan,Iron Pan,<p>Cast <b>iron</b></p>,Lodge,Home & Garden > Kitchen > Cookware,,true,"
        "40.00,shopify,3,active\n"
        "pan,,,,,,,35.00,shopify,2,\n"
        "mug,Mug,,,,Mugs,true,8,,,draft\n"
    )
    items, errors = shopify_csv(csv)
    pan, mug = items
    assert not errors and pan.description == "Cast iron" and pan.price == 35.0 and pan.stock == 5
    assert pan.category == "Home & Garden > Kitchen > Cookware" and pan.brand == "Lodge"
    assert mug.category == "Mugs" and mug.stock is None and not mug.active


def test_woocommerce_export_folds_variations() -> None:
    csv = (
        "ID,Type,SKU,Name,Published,In stock?,Stock,Sale price,Regular price,Categories,"
        "Description,Parent\n"
        '10,variable,TS,T-shirt,1,1,,,,"Clothing > Tshirts, Clothing",Soft,\n'
        "11,variation,TS-S,T-shirt S,1,1,4,,20,,,id:10\n"
        "12,variation,TS-M,T-shirt M,1,1,6,18,20,,,id:10\n"
        "13,simple,CAP,Cap,0,0,,,12,Accessories,,\n"
    )
    items, errors = woocommerce_csv(csv)
    ts, cap = items
    assert not errors and ts.price == 18 and ts.stock == 10 and ts.category == "Clothing > Tshirts"
    assert cap.stock == 0 and not cap.active


def test_xray_only_when_the_merchant_allows_it(env) -> None:
    client, keys, _ = env
    client.post("/v1/events", json={"events": _sessions(40)}, headers=_pub(keys))
    asked = {"placement": "next", "item_ids": ["sku0"], "xray": True}
    assert _rec(client, keys, **asked)["xray"] is None  # off by default
    rules = client.get("/v1/rules", headers=_priv(keys)).json() | {"allow_xray": True}
    client.put("/v1/rules", json=rules, headers=_priv(keys))
    r = client.post("/v1/recommend", json=asked, headers=_pub(keys))
    x = r.json()["xray"]
    assert r.headers["Server-Timing"].startswith("app;dur=")
    assert x["stage"] == "behavioural" and x["sources"] == ["recpop", "itemknn"]
    assert [i["id"] for i in x["items"]] == [i["id"] for i in r.json()["items"]]
    assert x["personalisation"].startswith("not personal: consent")
    top = x["items"][0]
    assert (
        top["reason_code"] == "viewed_next" and top["evidence"]["after"] == "sku0"
    )  # merchant ids
    # the best next item (sku1) is blocked: the x-ray says one better item was held back, and why
    client.put("/v1/rules", json=rules | {"blocked_items": ["sku1"]}, headers=_priv(keys))
    assert _rec(client, keys, **asked)["xray"]["hidden"] == {"blocked": 1}


def test_batch_serves_every_placement_on_a_page(env) -> None:
    client, keys, _ = env
    page = [
        {"placement": "similar", "item_ids": ["sku0"]},
        {"placement": "complements", "item_ids": ["sku0"], "k": 4},
        {"placement": "next", "item_ids": ["sku0"]},
    ]
    r = client.post("/v1/recommend/batch", json={"requests": page}, headers=_pub(keys)).json()
    assert [x["placement"] for x in r["responses"]] == ["similar", "complements", "next"]
    assert len(r["responses"][1]["items"]) <= 4
    too_many = {"requests": page * 3}
    assert client.post("/v1/recommend/batch", json=too_many, headers=_pub(keys)).status_code == 422


def test_images_travel_with_recommendations(env) -> None:
    client, keys, _ = env
    items = _items()
    for it in items:
        it["image_url"] = f"/plates/{it['id']}.svg"
    client.post("/v1/catalog", json={"items": items}, headers=_priv(keys))
    r = _rec(client, keys, placement="similar", item_ids=["sku0"])
    assert all(i["image_url"] == f"/plates/{i['id']}.svg" for i in r["items"])
    bad = [{"id": "x", "title": "X", "image_url": "javascript:alert(1)"}]
    assert client.post("/v1/catalog", json={"items": bad}, headers=_priv(keys)).status_code == 422


def test_status_reports_exact_progress_and_daily_totals(env) -> None:
    client, keys, _ = env
    st = client.get("/v1/status", headers=_priv(keys)).json()["progress"]
    assert st["stage"] == "catalog_only" and st["events_to_behavioural"] == 60
    assert [s["name"] for s in st["sources"]] == ["recpop", "itemknn"]
    client.post("/v1/events", json={"events": _sessions(40)}, headers=_pub(keys))
    st = client.get("/v1/status", headers=_priv(keys)).json()["progress"]
    assert st["stage"] == "behavioural" and st["events_to_behavioural"] == 0
    assert all(s["active"] for s in st["sources"])
    days = client.get("/v1/stats/daily?days=7", headers=_priv(keys)).json()
    assert len(days) == 7 and sum(d["views"] for d in days) == 40
    assert sum(d["orders"] for d in days) == 40


def test_importers_keep_safe_image_urls_only() -> None:
    csv = (
        "Handle,Title,Variant Price,Image Src\n"
        "pan,Pan,40,https://cdn.shopify.com/pan.jpg\n"
        "mug,Mug,8,http://insecure.example/mug.jpg\n"
    )
    pan, mug = shopify_csv(csv)[0]
    assert pan.image_url == "https://cdn.shopify.com/pan.jpg" and mug.image_url is None
    woo = 'ID,Type,Name,Regular price,Images\n7,simple,Cap,12,"https://x.in/a.jpg, https://x.in/b.jpg"\n'
    assert woocommerce_csv(woo)[0][0].image_url == "https://x.in/a.jpg"


def test_catalog_lookups_for_pickers(env) -> None:
    client, keys, _ = env
    cats = client.get("/v1/catalog/categories", headers=_priv(keys)).json()
    by_path = {c["path"]: c["products"] for c in cats}
    assert by_path["Kitchen > Pans"] == 6 and by_path["Kitchen"] == 18  # parents count children
    found = client.get("/v1/catalog/items?q=KNIV", headers=_priv(keys)).json()
    assert {f["id"] for f in found} == {f"sku{i}" for i in range(2, 24, 4)}
    assert client.get("/v1/catalog/items?q=50%25", headers=_priv(keys)).json() == []  # literal %
    picked = client.get("/v1/catalog/items?ids=sku1,sku3,nope", headers=_priv(keys)).json()
    assert sorted(p["id"] for p in picked) == ["sku1", "sku3"]
    assert client.get("/v1/catalog/items", headers=_pub(keys)).status_code in (401, 403)


def test_blocking_a_department_blocks_its_subcategories(env) -> None:
    client, keys, _ = env
    rules = client.get("/v1/rules", headers=_priv(keys)).json()
    rules["blocked_categories"] = ["Kitchen"]
    assert client.put("/v1/rules", json=rules, headers=_priv(keys)).status_code == 200
    rec = _rec(client, keys, placement="similar", item_ids=["sku3"], k=20)
    candles = {f"sku{i}" for i in range(3, 24, 4)}
    assert rec["items"] and {it["id"] for it in rec["items"]} <= candles
