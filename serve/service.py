"""Merchant-facing operations behind the API: catalog, events, recommendations, rules, the
complement graph, experiments, data-subject rights, and the model lifecycle.

Model lifecycle: a store's engine is retrained in a background worker when its data crosses a
volume threshold (``RetrainPolicy``), its catalog changes, or a shopper is erased. The model
registry keeps every version with the sources it activated; requests are served by the active
version until the new one is swapped in atomically.
"""

from __future__ import annotations

import json
import logging
import math
import pickle
import random
import threading
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from mitate import constraints as mc
from mitate.complements import CategoryGraph, EdgeStatus, propose_with_llm
from mitate.data import Catalog, Log
from mitate.engine import Engine, EngineConfig, EngineInfo, Recommendation, Shopper
from mitate.online.abtest import estimate, plan
from mitate.online.assign import assign
from mitate.online.interleave import analyse, team_draft
from serve import schemas as S
from serve.db import DB
from serve.privacy import Gate, key_digest, new_api_key, new_secret, pseudonym

log = logging.getLogger("mitate.serve")
EVENT_CODE = {S.EventType.VIEW: 0, S.EventType.CART: 1, S.EventType.ORDER: 2}
HISTORY_LIMIT = 200
PROCESSING_LOG_DAYS = 400  # Rule 8(3) requires at least one year


def _finite(d: dict[str, Any]) -> dict[str, Any]:
    """JSON has no infinities: unbounded or undefined interval ends become null."""
    return {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in d.items()}


class NotFoundError(Exception):
    pass


@dataclass(frozen=True)
class RetrainPolicy:
    min_new_events: int = 200  # absolute growth that triggers a retrain ...
    rel_growth: float = 0.2  # ... or relative growth, whichever is larger
    max_age_s: int = 86_400  # retrain daily when anything changed

    def due(self, n_trained: int, n_now: int, age_s: float, dirty: str | None) -> bool:
        if dirty is not None:  # new items, or an erased shopper who must leave the model
            return True
        grown = n_now - n_trained
        if grown >= max(self.min_new_events, self.rel_growth * n_trained):
            return True
        return grown > 0 and age_s >= self.max_age_s


@dataclass(frozen=True)
class CatalogView:
    """A merchant's catalog as the engine sees it, plus the display fields served with it."""

    catalog: Catalog
    state: mc.ItemState
    ext_ids: list[str]
    titles: list[str]
    prices: list[float | None]
    images: list[str | None]


@dataclass
class Live:
    """A merchant's served state: the engine plus the id maps it was trained with. Display
    fields follow the catalog between retrains (``_refresh_constraints``)."""

    engine: Engine
    version: int
    n_items: int
    ext_ids: list[str]
    titles: list[str]
    prices: list[float | None]
    images: list[str | None]
    trained_events: int
    trained_at: float
    rival: Engine | None = None  # previous version, for interleaving against it


@dataclass
class _Merchant:
    id: str
    secret: bytes
    rules: S.Rules
    live: Live | None = None
    dirty: str | None = None
    training: Future[Any] | None = field(default=None, repr=False)


class Service:
    def __init__(
        self,
        db: DB,
        data_dir: Path,
        engine_cfg: EngineConfig | None = None,
        policy: RetrainPolicy | None = None,
        background: bool = True,
    ) -> None:
        self.db = db
        self.data_dir = data_dir
        self.engine_cfg = engine_cfg or EngineConfig()
        self.policy = policy or RetrainPolicy()
        self._pool = ThreadPoolExecutor(max_workers=1) if background else None
        self._m: dict[str, _Merchant] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ merchants & auth
    def create_merchant(self, name: str) -> dict[str, str]:
        mid = uuid.uuid4().hex[:12]
        pub, priv = new_api_key("pk"), new_api_key("sk")
        self.db.x(
            "INSERT INTO merchants(id, name, secret, public_key, private_key, rules, graph,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?, '[]', ?)",
            (
                mid,
                name,
                new_secret(),
                key_digest(pub),
                key_digest(priv),
                S.Rules().model_dump_json(),
                int(time.time()),
            ),
        )
        return {"merchant_id": mid, "public_key": pub, "private_key": priv}

    def auth(self, key: str) -> tuple[str, str]:
        """(merchant id, scope) for an API key; scope is 'public' (widget) or 'private'. Keys are
        high-entropy random tokens, so looking up their SHA-256 digest by index is safe."""
        d = key_digest(key)
        row = self.db.one("SELECT id FROM merchants WHERE private_key = ?", (d,))
        if row is not None:
            return row["id"], "private"
        row = self.db.one("SELECT id FROM merchants WHERE public_key = ?", (d,))
        if row is not None:
            return row["id"], "public"
        raise PermissionError("invalid API key")

    def _merchant(self, mid: str) -> _Merchant:
        with self._lock:
            m = self._m.get(mid)
            if m is None:
                row = self.db.one("SELECT secret, rules FROM merchants WHERE id = ?", (mid,))
                if row is None:
                    raise NotFoundError("merchant")
                m = _Merchant(mid, bytes(row["secret"]), S.Rules.model_validate_json(row["rules"]))
                self._m[mid] = m
                m.live = self._load_active(mid)
            return m

    # ------------------------------------------------------------------ catalog
    def _category_idx(self, mid: str, path: str, known: dict[str, int]) -> int:
        """Index of a category path, creating it and its ancestors. ``known`` caches path ->
        index across one upload."""
        path = " > ".join(p.strip() for p in path.split(">") if p.strip())
        if not path:
            return -1
        if path in known:
            return known[path]
        parts = path.split(" > ")
        parent = self._category_idx(mid, " > ".join(parts[:-1]), known) if len(parts) > 1 else -1
        idx = len(known)
        self.db.x(
            "INSERT INTO categories VALUES (?, ?, ?, ?, ?)", (mid, idx, path, parts[-1], parent)
        )
        known[path] = idx
        return idx

    def upsert_items(
        self, mid: str, items: Sequence[S.Item], replace: bool, errors: Sequence[str] = ()
    ) -> S.CatalogResult:
        m = self._merchant(mid)
        now = int(time.time())
        existing = {
            r["ext_id"]: int(r["idx"])
            for r in self.db.q("SELECT ext_id, idx FROM items WHERE merchant = ?", (mid,))
        }
        nxt = len(existing)
        known = {
            r["path"]: int(r["idx"])
            for r in self.db.q("SELECT path, idx FROM categories WHERE merchant = ?", (mid,))
        }
        rows = []
        new_items = 0
        for it in items:
            idx = existing.get(it.id)
            if idx is None:
                idx, nxt, new_items = nxt, nxt + 1, new_items + 1
                existing[it.id] = idx
            rows.append(
                (
                    mid,
                    idx,
                    it.id,
                    it.title,
                    it.description,
                    self._category_idx(mid, it.category, known),
                    it.brand,
                    it.price,
                    it.stock,
                    json.dumps(sorted(it.regions)) if it.regions is not None else None,
                    it.launched_at,
                    int(it.active),
                    now,
                    it.image_url,
                )
            )
        self.db.many(
            "INSERT INTO items(merchant, idx, ext_id, title, description, category, brand, price,"
            " stock, regions, launched_at, active, added_at, image)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(merchant, idx) DO"
            " UPDATE SET title=excluded.title, description=excluded.description,"
            " category=excluded.category, brand=excluded.brand, price=excluded.price,"
            " stock=excluded.stock, regions=excluded.regions, launched_at=excluded.launched_at,"
            " active=excluded.active, image=excluded.image",
            rows,
        )
        deactivated = 0
        if replace:
            keep = {it.id for it in items}
            gone = [e for e in existing if e not in keep]
            for e in gone:
                deactivated += self.db.x(
                    "UPDATE items SET active = 0 WHERE merchant = ? AND ext_id = ? AND active = 1",
                    (mid, e),
                )
        self.db.log(mid, "catalog_upsert", len(rows))
        if new_items:
            m.dirty = m.dirty or "catalog"
        self._refresh_constraints(m)
        self.maybe_retrain(mid)  # a new store serves catalog-only recommendations at once
        n = self.db.one("SELECT COUNT(*) AS n FROM items WHERE merchant = ? AND active = 1", (mid,))
        return S.CatalogResult(
            upserted=len(rows),
            deactivated=deactivated,
            n_active=int(n["n"]) if n else 0,
            errors=list(errors)[:100],
        )

    def _catalog(self, mid: str) -> CatalogView:
        rows = self.db.q("SELECT * FROM items WHERE merchant = ? ORDER BY idx", (mid,))
        cats = self.db.q(
            "SELECT idx, name, parent FROM categories WHERE merchant = ? ORDER BY idx", (mid,)
        )
        n = len(rows)
        price = np.array([r["price"] if r["price"] is not None else np.nan for r in rows], float)
        stock = np.array([r["stock"] if r["stock"] is not None else np.nan for r in rows], float)
        stock[np.array([not r["active"] for r in rows], bool)] = 0.0  # inactive = unavailable
        category = np.array([r["category"] for r in rows], np.int32)
        brands: dict[str, int] = {}
        brand = np.array(
            [brands.setdefault(r["brand"], len(brands)) if r["brand"] else -1 for r in rows],
            np.int32,
        )
        launched = [r["launched_at"] for r in rows]
        first_seen = (
            np.array([x if x is not None else np.iinfo(np.int64).max for x in launched], np.int64)
            if any(x is not None for x in launched)
            else None
        )
        catalog = Catalog(
            n_items=n,
            title=[r["title"] for r in rows],
            text=[r["description"] for r in rows],
            category=category,
            category_names=[c["name"] for c in cats],
            category_parent=np.array([c["parent"] for c in cats], np.int32),
            brand=brand,
            brand_names=list(brands),
            price=price,
            first_seen=first_seen,
            raw_ids=[r["ext_id"] for r in rows],
        )
        regions = [frozenset(json.loads(r["regions"])) if r["regions"] else None for r in rows]
        state = mc.ItemState(stock=stock, category=category, price=price, regions=regions)
        return CatalogView(
            catalog,
            state,
            ext_ids=[r["ext_id"] for r in rows],
            titles=[r["title"] for r in rows],
            prices=[r["price"] for r in rows],
            images=[r["image"] for r in rows],
        )

    def _rules(self, mid: str, rules: S.Rules, ext_to_idx: dict[str, int]) -> mc.Rules:
        cats = {
            r["path"]: int(r["idx"])
            for r in self.db.q("SELECT path, idx FROM categories WHERE merchant = ?", (mid,))
        }

        def idx(ids: Sequence[str]) -> list[int]:
            return [ext_to_idx[i] for i in ids if i in ext_to_idx]

        # a blocked category covers everything beneath it ("Cookware" blocks "Cookware > Tawa")
        blocked = {
            i
            for path, i in cats.items()
            for c in rules.blocked_categories
            if path == c or path.startswith(c + " > ")
        }
        return mc.Rules(
            blocked_items=frozenset(idx(rules.blocked_items)),
            blocked_categories=frozenset(blocked),
            pins={p.value: idx(v) for p, v in rules.pins.items()},
            hide_untracked_stock=rules.hide_untracked_stock,
            max_per_category=rules.max_per_category,
            complement_max_price_ratio=rules.complement_max_price_ratio,
            similar_price_ratio=rules.similar_price_ratio,
        )

    def _refresh_constraints(self, m: _Merchant) -> None:
        """Stock, delivery regions and rules apply immediately, without retraining."""
        live = m.live
        if live is None:
            return
        view = self._catalog(m.id)
        state, ext = view.state, view.ext_ids
        n = live.n_items
        live.titles, live.prices, live.images = view.titles[:n], view.prices[:n], view.images[:n]
        state = mc.ItemState(
            stock=state.stock[:n],
            category=state.category[:n],
            price=state.price[:n],
            regions=state.regions[:n],
        )
        rules = self._rules(m.id, m.rules, {e: i for i, e in enumerate(ext[:n])})
        live.engine.constraints = mc.Constraints(state, rules)

    # ------------------------------------------------------------------ events
    def ingest(self, mid: str, batch: S.EventBatch) -> S.IngestResult:
        m = self._merchant(mid)
        ext = {
            r["ext_id"]: int(r["idx"])
            for r in self.db.q("SELECT ext_id, idx FROM items WHERE merchant = ?", (mid,))
        }
        now = int(time.time())
        rows, eng = [], []
        no_consent = unknown = 0
        suppressed = {
            r["shopper"]
            for r in self.db.q("SELECT shopper FROM suppression WHERE merchant = ?", (mid,))
        }
        for e in batch.events:
            item = ext.get(e.item_id)
            if item is None:
                unknown += 1
                continue
            ts = e.ts if e.ts is not None else now
            if e.impression_id:  # engagement with a shown list: aggregate, no shopper id
                eng.append((mid, e.impression_id, item, e.type.value, ts))
            if e.type is S.EventType.CLICK:
                continue
            gate = Gate.of(e.consent, e.age_status, m.rules.minor_safe_for_unknown_age)
            if not gate.personalise:
                no_consent += 1
                continue
            shopper = pseudonym(m.secret, e.shopper_id) if e.shopper_id else None
            if shopper in suppressed:
                no_consent += 1
                continue
            rows.append(
                (
                    mid,
                    shopper,
                    pseudonym(m.secret, "s:" + e.session_id),
                    item,
                    EVENT_CODE[e.type],
                    ts,
                    e.value,
                    e.context.channel,
                    e.context.device_class,
                )
            )
        if rows:
            self.db.many("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)", rows)
        if eng:
            self.db.many("INSERT INTO engagements VALUES (?,?,?,?,?)", eng)
        self.db.log(mid, "events_ingested", len(rows))
        self.maybe_retrain(mid)
        return S.IngestResult(
            accepted=len(rows), dropped_no_consent=no_consent, dropped_unknown_item=unknown
        )

    def _log(self, mid: str, n_items: int) -> Log:
        rows = self.db.q(
            "SELECT shopper, session, item, type, ts FROM events WHERE merchant = ? AND item < ?"
            " ORDER BY ts",
            (mid, n_items),
        )
        users: dict[str, int] = {}
        sessions: dict[str, int] = {}
        u = [users.setdefault(r["shopper"], len(users)) if r["shopper"] else -1 for r in rows]
        s = [sessions.setdefault(r["session"], len(sessions)) for r in rows]
        return Log.build(
            u,
            [r["item"] for r in rows],
            [r["ts"] for r in rows],
            [r["type"] for r in rows],
            s,
            len(users),
            n_items,
        )

    # ------------------------------------------------------------------ model lifecycle
    def n_events(self, mid: str) -> int:
        row = self.db.one("SELECT COUNT(*) AS n FROM events WHERE merchant = ?", (mid,))
        return int(row["n"]) if row else 0

    def maybe_retrain(self, mid: str) -> bool:
        m = self._merchant(mid)
        if m.training is not None and not m.training.done():
            return False
        n_now = self.n_events(mid)
        live = m.live
        if live is None:
            due = True
        else:
            due = self.policy.due(
                live.trained_events, n_now, time.time() - live.trained_at, m.dirty
            )
        if not due:
            return False
        if self._pool is None:
            self.retrain(mid)
        else:
            m.training = self._pool.submit(self.retrain, mid)
        return True

    def retrain(self, mid: str) -> EngineInfo:
        m = self._merchant(mid)
        self.purge(mid)
        view = self._catalog(mid)
        catalog, state, ext = view.catalog, view.state, view.ext_ids
        if catalog.n_items == 0:
            raise NotFoundError("catalog is empty")
        log_ = self._log(mid, catalog.n_items)
        rules = self._rules(mid, m.rules, {e: i for i, e in enumerate(ext)})
        graph = self.graph(mid)
        t0 = time.time()
        engine = Engine(self.engine_cfg)
        dirty_before = m.dirty
        info = engine.fit(log_, catalog, state, rules, graph)
        version = self._register(mid, engine, info)
        live = Live(
            engine,
            version,
            catalog.n_items,
            ext,
            view.titles,
            view.prices,
            view.images,
            len(log_),
            t0,
            rival=m.live.engine if m.live is not None else None,
        )
        with self._lock:
            m.live = live
            if m.dirty == dirty_before:
                m.dirty = None
        self.db.log(mid, "model_trained", len(log_))
        log.info("merchant=%s version=%d stage=%s events=%d", mid, version, info.stage, len(log_))
        return info

    def _register(self, mid: str, engine: Engine, info: EngineInfo) -> int:
        d = self.data_dir / "models" / mid
        d.mkdir(parents=True, exist_ok=True)
        row = self.db.one("SELECT MAX(version) AS v FROM models WHERE merchant = ?", (mid,))
        version = int(row["v"]) + 1 if row and row["v"] is not None else 1
        path = d / f"v{version}.pkl"
        # Engines are produced by this process from the merchant's own data; pickle is never
        # used on anything received over the network.
        with path.open("wb") as f:
            pickle.dump(engine, f, protocol=pickle.HIGHEST_PROTOCOL)
        with self.db.tx() as c:
            c.execute("UPDATE models SET active = 0 WHERE merchant = ?", (mid,))
            c.execute(
                "INSERT INTO models VALUES (?, ?, ?, ?, ?, 1)",
                (mid, version, str(path), json.dumps(asdict(info)), int(time.time())),
            )
        return version

    def _load_active(self, mid: str) -> Live | None:
        row = self.db.one(
            "SELECT version, path, info, created_at FROM models WHERE merchant = ? AND active = 1",
            (mid,),
        )
        if row is None or not Path(row["path"]).exists():
            return None
        with Path(row["path"]).open("rb") as f:
            engine: Engine = pickle.load(f)  # our own artefact, see _register
        n = engine.catalog.n_items
        view = self._catalog(mid)
        info = json.loads(row["info"])
        return Live(
            engine,
            int(row["version"]),
            n,
            view.ext_ids[:n],
            view.titles[:n],
            view.prices[:n],
            view.images[:n],
            int(info["n_events"]),
            float(row["created_at"]),
        )

    def models(self, mid: str) -> list[dict[str, Any]]:
        return [
            {
                "version": r["version"],
                "active": bool(r["active"]),
                "created_at": r["created_at"],
                **json.loads(r["info"]),
            }
            for r in self.db.q(
                "SELECT * FROM models WHERE merchant = ? ORDER BY version DESC", (mid,)
            )
        ]

    def wait(self, mid: str) -> None:
        """Block until a pending retrain finishes (tests, CLI)."""
        m = self._merchant(mid)
        if m.training is not None:
            m.training.result()

    # ------------------------------------------------------------------ recommendations
    def recommend(
        self,
        mid: str,
        placement: S.Placement,
        *,
        item_ids: Sequence[str] = (),
        shopper_id: str | None = None,
        session_id: str | None = None,
        consent: S.Consent = S.Consent.UNKNOWN,
        age_status: S.AgeStatus = S.AgeStatus.UNKNOWN,
        region: str | None = None,
        k: int = 8,
        xray: bool = False,
    ) -> S.RecResponse:
        t_start = time.perf_counter()
        m = self._merchant(mid)
        live = m.live
        imp_id = uuid.uuid4().hex
        if live is None:
            return S.RecResponse(
                impression_id=imp_id,
                placement=placement,
                items=[],
                personalised=False,
                model_version=None,
                stage="untrained",
            )
        gate = Gate.of(consent, age_status, m.rules.minor_safe_for_unknown_age)
        idx = {e: i for i, e in enumerate(live.ext_ids)}
        anchors = [idx[i] for i in item_ids if i in idx]
        unit: str | None = None
        shopper = Shopper(now=int(time.time()))
        if gate.personalise:
            sh = pseudonym(m.secret, shopper_id) if shopper_id else None
            se = pseudonym(m.secret, "s:" + session_id) if session_id else None
            unit = sh or se
            shopper = self._shopper(mid, sh, se, live.n_items)
        place = mc.Placement(placement.value)
        ctx = mc.Context(
            region=region, anchor=anchors[0] if anchors else -1, exclude=tuple(anchors)
        )
        exp, arm = self._arm(mid, placement, unit)
        audit = xray and m.rules.allow_xray
        hidden: dict[str, int] = {}
        teams: list[int] | None = None
        if arm == "holdout":
            assert exp is not None  # _arm returns an arm only together with its experiment
            recs = [] if exp["holdout_serves"] == "nothing" else self._popular(live, place, ctx, k)
        elif arm == "interleave":
            assert exp is not None
            a = live.engine.recommend(place, k, anchors=anchors, shopper=shopper, ctx=ctx)
            b = self._rival(live, exp, place, k, anchors, shopper, ctx)
            rng = random.Random(imp_id)
            il = team_draft([r.item for r in a], [r.item for r in b], k, rng)
            by = {r.item: r for r in [*b, *a]}  # A's reason wins when both have one
            recs, teams = [by[cast(int, i)] for i in il.items], list(il.team)  # engine item ids
        else:
            ranked = live.engine.rank(
                place, k, anchors=anchors, shopper=shopper, ctx=ctx, audit=audit
            )
            recs, hidden = ranked.items, ranked.hidden
        items = [
            S.RecItem(
                id=live.ext_ids[r.item],
                title=live.titles[r.item],
                price=live.prices[r.item],
                reason=r.reason.text if r.reason else None,
                reason_code=r.reason.code if r.reason else None,
                image_url=live.images[r.item],
            )
            for r in recs
        ]
        self.db.x(
            "INSERT INTO impressions VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                imp_id,
                mid,
                placement.value,
                unit if exp else None,
                exp["id"] if exp else None,
                arm,
                json.dumps([r.item for r in recs]),
                json.dumps(teams) if teams else None,
                json.dumps([asdict(r.reason) if r.reason else None for r in recs]),
                int(time.time()),
            ),
        )
        personalised = gate.personalise and bool(shopper.items)
        return S.RecResponse(
            impression_id=imp_id,
            placement=placement,
            items=items,
            personalised=personalised,
            model_version=live.version,
            stage=live.engine.info.stage,
            arm=arm,
            xray=self._xray(live, recs, hidden, gate.reason, shopper, personalised, t_start)
            if audit
            else None,
        )

    def _xray(
        self,
        live: Live,
        recs: Sequence[Recommendation],
        hidden: dict[str, int],
        gate: str,
        shopper: Shopper,
        personalised: bool,
        t_start: float,
    ) -> S.XRay:
        info = live.engine.info
        cats = live.engine.catalog.category_names or []

        def public(evidence: dict[str, Any]) -> dict[str, object]:
            """Internal indices in the evidence become the merchant's ids and category names."""
            out: dict[str, object] = {}
            for key, v in evidence.items():
                if key in ("with", "after", "anchor") and isinstance(v, int):
                    out[key] = live.ext_ids[v] if 0 <= v < len(live.ext_ids) else None
                elif key == "categories":
                    out[key] = [cats[c] if 0 <= c < len(cats) else None for c in v]
                else:
                    out[key] = v
            return out

        if gate == "granted":
            why = "personal: consent granted" if personalised else "consent granted, no history"
        elif gate == "minor_safe":
            why = "not personal: minor-safe mode (no behavioural data)"
        else:
            why = f"not personal: {gate.replace('_', ' ')}"
        return S.XRay(
            stage=info.stage,
            sources=list(info.active_sources),
            ranker=info.ranker,
            events_learned_from=info.n_events,
            trained_at=info.trained_at,
            personalisation=why,
            history_events=len(shopper.items) if personalised else 0,
            hidden=hidden,
            items=[
                S.XRayItem(
                    id=live.ext_ids[r.item],
                    pinned=r.pinned,
                    reason_code=r.reason.code if r.reason else None,
                    evidence=public(r.reason.evidence) if r.reason else {},
                )
                for r in recs
            ],
            server_ms=round((time.perf_counter() - t_start) * 1e3, 2),
        )

    def _shopper(self, mid: str, sh: str | None, se: str | None, n_items: int) -> Shopper:
        if sh:
            # two index range scans (ev_shopper, ev_session); an OR would scan the table
            rows = self.db.q(
                "SELECT item, type, ts, session FROM (SELECT * FROM (SELECT item, type, ts, session"
                " FROM events WHERE merchant = ? AND shopper = ? ORDER BY ts DESC LIMIT ?) UNION"
                " SELECT * FROM (SELECT item, type, ts, session FROM events WHERE merchant = ? AND"
                " session = ? ORDER BY ts DESC LIMIT ?)) ORDER BY ts DESC LIMIT ?",
                (mid, sh, HISTORY_LIMIT, mid, se or "", HISTORY_LIMIT, HISTORY_LIMIT),
            )
        elif se:
            rows = self.db.q(
                "SELECT item, type, ts, session FROM events WHERE merchant = ? AND session = ?"
                " ORDER BY ts DESC LIMIT ?",
                (mid, se, HISTORY_LIMIT),
            )
        else:
            rows = []
        rows = [r for r in reversed(rows) if r["item"] < n_items]
        start = next((j for j, r in enumerate(rows) if r["session"] == se), len(rows))
        return Shopper(
            items=[r["item"] for r in rows],
            ts=[r["ts"] for r in rows],
            events=[r["type"] for r in rows],
            session_start=start,
            known=sh is not None,
            now=int(time.time()),
        )

    def _popular(
        self, live: Live, place: mc.Placement, ctx: mc.Context, k: int
    ) -> list[Recommendation]:
        S_ = np.log1p(live.engine.evidence.shoppers_7d.astype(np.float64))
        slate = live.engine.constraints.slate(S_, k, place, ctx)
        return [
            Recommendation(i, float(S_[i]), p, None)
            for i, p in zip(slate.items, slate.pinned, strict=True)
        ]

    def _rival(
        self,
        live: Live,
        exp: dict[str, Any],
        place: mc.Placement,
        k: int,
        anchors: list[int],
        shopper: Shopper,
        ctx: mc.Context,
    ) -> list[Recommendation]:
        if exp["comparison"] == "previous_model" and live.rival is not None:
            n_old = live.rival.catalog.n_items
            old_shopper = Shopper(
                items=[i for i in shopper.items if i < n_old],
                ts=[t for i, t in zip(shopper.items, shopper.ts, strict=True) if i < n_old],
                events=[e for i, e in zip(shopper.items, shopper.events, strict=True) if i < n_old],
                known=shopper.known,
                now=shopper.now,
            )
            return live.rival.recommend(
                place, k, anchors=[a for a in anchors if a < n_old], shopper=old_shopper, ctx=ctx
            )
        return self._popular(live, place, ctx, k)

    def explain(self, mid: str, impression_id: str) -> dict[str, Any]:
        m = self._merchant(mid)
        row = self.db.one(
            "SELECT * FROM impressions WHERE id = ? AND merchant = ?", (impression_id, mid)
        )
        if row is None:
            raise NotFoundError("impression")
        items = json.loads(row["items"])
        reasons = json.loads(row["reasons"])
        ext = m.live.ext_ids if m.live else []
        return {
            "impression_id": impression_id,
            "placement": row["placement"],
            "items": [
                {"id": ext[i] if i < len(ext) else None, "reason": r}
                for i, r in zip(items, reasons, strict=True)
            ],
        }

    # ------------------------------------------------------------------ rules & graph
    def get_rules(self, mid: str) -> S.Rules:
        return self._merchant(mid).rules

    def put_rules(self, mid: str, rules: S.Rules) -> S.Rules:
        m = self._merchant(mid)
        self.db.x("UPDATE merchants SET rules = ? WHERE id = ?", (rules.model_dump_json(), mid))
        m.rules = rules
        self._refresh_constraints(m)
        return rules

    def graph(self, mid: str) -> CategoryGraph:
        row = self.db.one("SELECT graph FROM merchants WHERE id = ?", (mid,))
        return CategoryGraph.from_json(row["graph"]) if row else CategoryGraph()

    def _category_paths(self, mid: str) -> list[str]:
        return [
            r["path"]
            for r in self.db.q(
                "SELECT path FROM categories WHERE merchant = ? ORDER BY idx", (mid,)
            )
        ]

    def categories(self, mid: str) -> list[dict[str, Any]]:
        """Every category path with its active products, counting those in subcategories."""
        rows = self.db.q(
            "SELECT c.path, COUNT(i.idx) AS n FROM categories c LEFT JOIN items i"
            " ON i.merchant = c.merchant AND i.category = c.idx AND i.active = 1"
            " WHERE c.merchant = ? GROUP BY c.idx ORDER BY c.path",
            (mid,),
        )
        direct = {r["path"]: int(r["n"]) for r in rows}
        return [
            {
                "path": p,
                "products": sum(n for q, n in direct.items() if q == p or q.startswith(p + " > ")),
            }
            for p in direct
        ]

    def find_items(
        self, mid: str, q: str = "", ids: Sequence[str] = (), limit: int = 20
    ) -> list[dict[str, Any]]:
        """Active products by id, or by a case-insensitive substring of the title."""
        cols = (
            "SELECT i.ext_id AS id, i.title, c.path AS category, i.price, i.stock, i.image"
            " FROM items i LEFT JOIN categories c ON c.merchant = i.merchant AND c.idx = i.category"
            " WHERE i.merchant = ? AND i.active = 1"
        )
        if ids:
            marks = ",".join("?" * len(ids))
            rows = self.db.q(f"{cols} AND i.ext_id IN ({marks})", (mid, *ids))
        else:
            pattern = "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            rows = self.db.q(
                f"{cols} AND i.title LIKE ? ESCAPE '\\' ORDER BY i.title LIMIT ?",
                (mid, pattern, limit),
            )
        return [dict(r) for r in rows]

    def graph_view(self, mid: str) -> list[S.GraphEdge]:
        paths = self._category_paths(mid)
        return [
            S.GraphEdge(src=paths[e.src], dst=paths[e.dst], status=e.status.value, origin=e.origin)
            for e in self.graph(mid).edges
        ]

    def review_graph(self, mid: str, review: S.GraphReview) -> list[S.GraphEdge]:
        paths = {p: i for i, p in enumerate(self._category_paths(mid))}
        g = self.graph(mid)
        for e in review.edges:
            if e.src in paths and e.dst in paths:
                g.review(paths[e.src], paths[e.dst], EdgeStatus(e.status))
        self._save_graph(mid, g)
        return self.graph_view(mid)

    def propose_graph(self, mid: str, **llm: Any) -> int:
        cats = self.db.q(
            "SELECT name, parent FROM categories WHERE merchant = ? ORDER BY idx", (mid,)
        )
        names = [c["name"] for c in cats]
        parent = np.array([c["parent"] for c in cats], np.int32)
        g = self.graph(mid)
        added = g.merge_proposals(propose_with_llm(names, parent, **llm))
        self._save_graph(mid, g)
        return added

    def _save_graph(self, mid: str, g: CategoryGraph) -> None:
        self.db.x("UPDATE merchants SET graph = ? WHERE id = ?", (g.to_json(), mid))
        m = self._merchant(mid)
        if m.live is not None:
            # approved edges serve immediately: rebuild the complement scorer only
            eng = m.live.engine
            eng.graph = g
            eng.complements = type(eng.complements)(eng.evidence, g, eng.complements.item_category)

    # ------------------------------------------------------------------ experiments
    def create_experiment(self, mid: str, spec: S.ExperimentCreate) -> S.Experiment:
        if self._running(mid, spec.placement):
            raise ValueError("an experiment is already running on this placement")
        eid = uuid.uuid4().hex[:12]
        now = int(time.time())
        self.db.x(
            "INSERT INTO experiments VALUES (?, ?, ?, 'running', ?, NULL)",
            (eid, mid, spec.model_dump_json(), now),
        )
        return self.experiment(mid, eid)

    def experiment(self, mid: str, eid: str) -> S.Experiment:
        r = self.db.one("SELECT * FROM experiments WHERE id = ? AND merchant = ?", (eid, mid))
        if r is None:
            raise NotFoundError("experiment")
        spec = json.loads(r["spec"])
        return S.Experiment(
            id=eid,
            status=r["status"],
            started_at=r["started_at"],
            stopped_at=r["stopped_at"],
            **spec,
        )

    def experiments(self, mid: str) -> list[S.Experiment]:
        rows = self.db.q(
            "SELECT id FROM experiments WHERE merchant = ? ORDER BY started_at DESC", (mid,)
        )
        return [self.experiment(mid, r["id"]) for r in rows]

    def stop_experiment(self, mid: str, eid: str) -> S.Experiment:
        self.db.x(
            "UPDATE experiments SET status = 'stopped', stopped_at = ? WHERE id = ? AND"
            " merchant = ? AND status = 'running'",
            (int(time.time()), eid, mid),
        )
        return self.experiment(mid, eid)

    def _running(self, mid: str, placement: S.Placement | None) -> dict[str, Any] | None:
        for r in self.db.q(
            "SELECT id, spec FROM experiments WHERE merchant = ? AND status = 'running'", (mid,)
        ):
            spec = json.loads(r["spec"])
            if placement is None or spec["placement"] in (None, placement.value):
                return {"id": r["id"], **spec}
        return None

    def _arm(
        self, mid: str, placement: S.Placement, unit: str | None
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Only consented, identifiable visitors take part; everyone else gets the production
        model and is outside the experiment's population (stated in every report)."""
        exp = self._running(mid, placement)
        if exp is None or unit is None:
            return None, None
        if exp["kind"] == S.ExperimentKind.INTERLEAVE.value:
            arm = "interleave"
        else:
            share = float(exp["holdout_share"])
            arm = assign(exp["id"], unit, [("holdout", share), ("mitate", 1 - share)])
        self.db.x(
            "INSERT OR IGNORE INTO exposures VALUES (?, ?, ?, ?)",
            (exp["id"], unit, arm, int(time.time())),
        )
        return exp, arm

    def report(self, mid: str, eid: str) -> dict[str, Any]:
        e = self.experiment(mid, eid)
        end = e.stopped_at or int(time.time())
        population = (
            "Consented visitors with a shopper or session id; others were served the "
            "production model and are not in this experiment."
        )
        if e.kind is S.ExperimentKind.INTERLEAVE:
            rows = self.db.q(
                "SELECT i.id, i.unit, i.items, i.teams FROM impressions i WHERE i.experiment = ?"
                " AND i.teams IS NOT NULL",
                (eid,),
            )
            eng: dict[str, list[int]] = {}
            for r in self.db.q(
                "SELECT g.impression, g.item FROM engagements g JOIN impressions i ON"
                " i.id = g.impression WHERE i.experiment = ?",
                (eid,),
            ):
                eng.setdefault(r["impression"], []).append(int(r["item"]))
            vis, ca, cb = [], [], []
            for r in rows:
                items, teams = json.loads(r["items"]), json.loads(r["teams"])
                clicked = set(eng.get(r["id"], []))
                a = sum(1 for i, t in zip(items, teams, strict=True) if i in clicked and t == 0)
                b = sum(1 for i, t in zip(items, teams, strict=True) if i in clicked and t == 1)
                vis.append(r["unit"])
                ca.append(a)
                cb.append(b)
            res = analyse(vis, ca, cb) if vis else None
            return {
                "experiment": e.model_dump(),
                "population": population,
                "comparison": f"mitate (A) vs {e.comparison} (B)",
                "interleaving": _finite(asdict(res)) if res else None,
                "note": "Interleaving measures preference between rankers, not revenue lift.",
            }
        units = self.db.q("SELECT unit, arm, first_ts FROM exposures WHERE experiment = ?", (eid,))
        if not units:
            return {"experiment": e.model_dump(), "population": population, "units": 0}
        pre_start = e.started_at - (end - e.started_at)
        y: dict[str, list[float]] = {"holdout": [], "mitate": []}
        x: dict[str, list[float]] = {"holdout": [], "mitate": []}
        conv: dict[str, list[float]] = {"holdout": [], "mitate": []}
        for u in units:
            unit, arm, t0 = u["unit"], u["arm"], int(u["first_ts"])
            post = self.db.one(
                "SELECT COALESCE(SUM(value), 0) AS v, COUNT(*) AS n FROM events WHERE merchant = ?"
                " AND (shopper = ? OR session = ?) AND type = 2 AND ts >= ? AND ts < ?",
                (mid, unit, unit, t0, end),
            )
            pre = self.db.one(
                "SELECT COALESCE(SUM(value), 0) AS v FROM events WHERE merchant = ? AND"
                " (shopper = ? OR session = ?) AND type = 2 AND ts >= ? AND ts < ?",
                (mid, unit, unit, pre_start, e.started_at),
            )
            y[arm].append(float(post["v"]) if post else 0.0)
            conv[arm].append(1.0 if post and post["n"] else 0.0)
            x[arm].append(float(pre["v"]) if pre else 0.0)
        out: dict[str, Any] = {
            "experiment": e.model_dump(),
            "population": population,
            "units": {a: len(v) for a, v in y.items()},
            "window": {"start": e.started_at, "end": end, "pre_period_start": pre_start},
            "note": (
                "Effects are treatment (mitate) minus holdout per visitor, with 95% CIs. "
                "Clicks on recommendations are never counted as lift."
            ),
        }
        if min(len(y["holdout"]), len(y["mitate"])) < 2:
            out["status"] = "not enough visitors in both arms yet"
            return out
        has_pre = any(v > 0 for v in x["holdout"] + x["mitate"])
        rpv = estimate(
            "revenue_per_visitor",
            y["holdout"],
            y["mitate"],
            x["holdout"] if has_pre else None,
            x["mitate"] if has_pre else None,
        )
        cr = estimate("conversion", conv["holdout"], conv["mitate"])
        out["revenue_per_visitor"] = _finite(asdict(rpv))
        out["conversion"] = _finite(asdict(cr))
        return out

    def power(self, mid: str, req: S.PowerRequest) -> dict[str, Any]:
        """Per-visitor revenue spread from the store's own orders over the last 28 days."""
        since = int(time.time()) - 28 * 86_400
        rows = self.db.q(
            "SELECT COALESCE(shopper, session) AS u, SUM(CASE WHEN type = 2 THEN COALESCE(value, 0)"
            " ELSE 0 END) AS v FROM events WHERE merchant = ? AND ts >= ? GROUP BY u",
            (mid, since),
        )
        v = np.array([float(r["v"]) for r in rows])
        if v.size < 30 or v.mean() <= 0:
            return {
                "status": "not enough order data in the last 28 days to estimate variance",
                "visitors_seen": int(v.size),
            }
        p = plan(
            req.daily_visitors,
            float(v.mean()),
            float(v.std(ddof=1)),
            req.rel_mde,
            req.horizon_days,
            treat_share=1 - req.holdout_share,
        )
        return {
            "basis_visitors": int(v.size),
            "mean_rpv": float(v.mean()),
            "std_rpv": float(v.std(ddof=1)),
            **asdict(p),
        }

    # ------------------------------------------------------------------ shopper rights
    def erase(self, mid: str, shopper_id: str) -> dict[str, int]:
        """DPDP s.12 erasure: delete events and experiment exposure, suppress future events,
        and retrain so the shopper leaves the served model (RetrainPolicy.erasure_deadline_s)."""
        m = self._merchant(mid)
        h = pseudonym(m.secret, shopper_id)
        with self.db.tx() as c:
            sessions = [
                r[0]
                for r in c.execute(
                    "SELECT DISTINCT session FROM events WHERE merchant = ? AND shopper = ?",
                    (mid, h),
                )
            ]
            n = c.execute(
                "DELETE FROM events WHERE merchant = ? AND shopper = ?", (mid, h)
            ).rowcount
            for s in sessions:
                n += c.execute(
                    "DELETE FROM events WHERE merchant = ? AND session = ?", (mid, s)
                ).rowcount
            for u in [h, *sessions]:
                c.execute(
                    "DELETE FROM exposures WHERE unit = ? AND experiment IN"
                    " (SELECT id FROM experiments WHERE merchant = ?)",
                    (u, mid),
                )
                c.execute(
                    "UPDATE impressions SET unit = NULL WHERE merchant = ? AND unit = ?", (mid, u)
                )
            c.execute(
                "INSERT OR REPLACE INTO suppression VALUES (?, ?, ?)", (mid, h, int(time.time()))
            )
        self.db.log(mid, "shopper_erased", 1)
        m.dirty = "erasure"
        self.maybe_retrain(mid)
        return {"events_deleted": n}

    def export(self, mid: str, shopper_id: str) -> dict[str, Any]:
        m = self._merchant(mid)
        h = pseudonym(m.secret, shopper_id)
        ext = {
            int(r["idx"]): r["ext_id"]
            for r in self.db.q("SELECT idx, ext_id FROM items WHERE merchant = ?", (mid,))
        }
        names = {v: k for k, v in EVENT_CODE.items()}
        rows = self.db.q(
            "SELECT item, type, ts, value, channel, device FROM events WHERE merchant = ? AND"
            " shopper = ? ORDER BY ts",
            (mid, h),
        )
        self.db.log(mid, "shopper_exported", 1)
        return {
            "shopper": "pseudonymised; raw id not stored",
            "events": [
                {
                    "item_id": ext.get(r["item"]),
                    "type": names[r["type"]].value,
                    "ts": r["ts"],
                    "value": r["value"],
                    "channel": r["channel"],
                    "device_class": r["device"],
                }
                for r in rows
            ],
            "processing": (
                "Events are used to recommend products in this store and to measure "
                "whether recommendations help. They are not shared with other stores."
            ),
        }

    def purge(self, mid: str) -> dict[str, int]:
        """Retention (P7): behavioural events past the merchant's TTL; processing log past 400
        days; impressions past the TTL."""
        m = self._merchant(mid)
        now = int(time.time())
        ttl = now - m.rules.retention_days * 86_400
        ev = self.db.x("DELETE FROM events WHERE merchant = ? AND ts < ?", (mid, ttl))
        imp = self.db.x("DELETE FROM impressions WHERE merchant = ? AND ts < ?", (mid, ttl))
        self.db.x(
            "DELETE FROM processing_log WHERE merchant = ? AND ts < ?",
            (mid, now - PROCESSING_LOG_DAYS * 86_400),
        )
        if ev:
            self.db.log(mid, "retention_purge", ev)
        return {"events": ev, "impressions": imp}

    def status(self, mid: str) -> dict[str, Any]:
        m = self._merchant(mid)
        live = m.live
        n_items = self.db.one(
            "SELECT COUNT(*) AS n FROM items WHERE merchant = ? AND active = 1", (mid,)
        )
        n_now = self.n_events(mid)
        cfg, pol = self.engine_cfg, self.policy
        info = live.engine.info if live else None
        trained = live.trained_events if live else 0
        return {
            "merchant_id": mid,
            "active_items": int(n_items["n"]) if n_items else 0,
            "events": n_now,
            "model": asdict(info) | {"version": live.version} if live and info else None,
            "pending_retrain": m.dirty,
            "training": bool(m.training is not None and not m.training.done()),
            # exact thresholds from the engine config and retrain policy, never estimates
            "progress": {
                "stage": info.stage if info else "untrained",
                "ranker": info.ranker if info else "none",
                "trained_on_events": trained,
                "behavioural_from_events": cfg.min_behaviour_events,
                "events_to_behavioural": max(0, cfg.min_behaviour_events - n_now),
                "sources": [
                    {
                        "name": src,
                        "from_events": cfg.source_min_events.get(src, cfg.min_behaviour_events),
                        "active": bool(info and src in info.active_sources),
                    }
                    for src in cfg.sources
                ],
                "retrain": {
                    "new_events": n_now - trained,
                    "new_events_needed": int(max(pol.min_new_events, pol.rel_growth * trained)),
                    "age_s": round(time.time() - live.trained_at) if live else None,
                    "daily_after_s": pol.max_age_s,
                },
            },
        }

    def daily(self, mid: str, days: int) -> list[dict[str, Any]]:
        """Events, orders and revenue per UTC day over the last ``days`` days (aggregates only)."""
        since = (int(time.time()) // 86_400 - days + 1) * 86_400
        rows = self.db.q(
            "SELECT ts / 86400 AS day, type, COUNT(*) AS n, COALESCE(SUM(value), 0) AS v"
            " FROM events WHERE merchant = ? AND ts >= ? GROUP BY day, type",
            (mid, since),
        )
        first = since // 86_400
        out: dict[int, dict[str, Any]] = {}
        for d in range(first, first + days):
            out[d] = {"day": d * 86_400, "views": 0, "carts": 0, "orders": 0, "revenue": 0.0}
        names = {0: "views", 1: "carts", 2: "orders"}
        for r in rows:
            if r["day"] in out and r["type"] in names:
                out[r["day"]][names[r["type"]]] += int(r["n"])
                if r["type"] == 2:
                    out[r["day"]]["revenue"] += float(r["v"])
        return list(out.values())

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
