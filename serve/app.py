"""HTTP API.

    MITATE_DATA_DIR=data/serve uv run python -m serve run        # API on :8000
    uv run python -m serve create-merchant "My Shop"            # prints keys once

Two keys per merchant: the *public* key (``pk_``) is embedded in the storefront widget and can only
send events and fetch recommendations; the *private* key (``sk_``) manages the catalog, rules,
experiments and shopper rights, and must stay on the merchant's server.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import yaml
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from mitate.engine import EngineConfig
from serve import schemas as S
from serve.db import DB
from serve.importers import IMPORTERS
from serve.service import NotFoundError, RetrainPolicy, Service

log = logging.getLogger("mitate.serve")
ROOT = Path(__file__).resolve().parent.parent


def engine_config(path: str | None) -> EngineConfig:
    if not path:
        return EngineConfig()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if raw.get("booster_dir"):
        raw["booster_dir"] = Path(raw["booster_dir"])
    if "sources" in raw:
        raw["sources"] = tuple(raw["sources"])
    return EngineConfig(**raw)


# ---------------------------------------------------------------------------- auth
def svc(request: Request) -> Service:
    s: Service = request.app.state.svc
    return s


Svc = Annotated[Service, Depends(svc)]


def key(request: Request, x_mitate_key: Annotated[str | None, Header()] = None) -> tuple[str, str]:
    if not x_mitate_key:
        raise HTTPException(401, "missing X-Mitate-Key")
    try:
        return svc(request).auth(x_mitate_key)
    except PermissionError:
        raise HTTPException(401, "invalid key") from None


def public(k: Annotated[tuple[str, str], Depends(key)]) -> str:
    return k[0]


def private(k: Annotated[tuple[str, str], Depends(key)]) -> str:
    if k[1] != "private":
        raise HTTPException(403, "this endpoint needs the private key")
    return k[0]


Pub = Annotated[str, Depends(public)]
Priv = Annotated[str, Depends(private)]


def create_app(service: Service | None = None, cors: list[str] | None = None) -> FastAPI:
    data_dir = Path(os.environ.get("MITATE_DATA_DIR", "data/serve"))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is None:
            data_dir.mkdir(parents=True, exist_ok=True)
            svc = Service(
                DB(data_dir / "mitate.sqlite"),
                data_dir,
                engine_config(os.environ.get("MITATE_ENGINE_CONFIG")),
                RetrainPolicy(),
            )
        else:
            svc = service
        app.state.svc = svc
        yield
        svc.shutdown()

    app = FastAPI(title="Mitate", version="0.1.0", lifespan=lifespan)
    origins = cors if cors is not None else os.environ.get("MITATE_CORS_ORIGINS", "").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o for o in origins if o],
        allow_methods=["GET", "POST"],
        allow_headers=["X-Mitate-Key", "Content-Type"],
        expose_headers=["Server-Timing"],
    )

    @app.middleware("http")
    async def access_log(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # P8: route template, status and timing only; never query strings, bodies or ids
        rid = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        resp = await call_next(request)
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        ms = (time.perf_counter() - t0) * 1e3
        resp.headers["X-Request-Id"] = rid
        resp.headers["Server-Timing"] = f"app;dur={ms:.1f}"
        origin = request.headers.get("origin")
        if origin and origin in origins:  # cross-origin pages may read the timing they caused
            resp.headers["Timing-Allow-Origin"] = origin
        log.info("rid=%s %s %s %d %.1fms", rid, request.method, path, resp.status_code, ms)
        return resp

    @app.exception_handler(NotFoundError)
    async def not_found(request: Request, exc: NotFoundError) -> Response:
        return JSONResponse(status_code=404, content={"detail": f"{exc} not found"})

    # -------------------------------------------------------------- health
    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    # -------------------------------------------------------------- widget scope
    @app.post("/v1/events", response_model=S.IngestResult)
    def events(batch: S.EventBatch, mid: Pub, s: Svc) -> S.IngestResult:
        return s.ingest(mid, batch)

    def one(s: Service, mid: str, req: S.RecRequest) -> S.RecResponse:
        return s.recommend(
            mid,
            req.placement,
            item_ids=req.item_ids,
            shopper_id=req.shopper_id,
            session_id=req.session_id,
            consent=req.consent,
            age_status=req.age_status,
            region=req.region,
            k=req.k,
            xray=req.xray,
        )

    @app.post("/v1/recommend", response_model=S.RecResponse)
    def recommend(req: S.RecRequest, mid: Pub, s: Svc) -> S.RecResponse:
        return one(s, mid, req)

    @app.post("/v1/recommend/batch", response_model=S.RecBatchResponse)
    def recommend_batch(batch: S.RecBatch, mid: Pub, s: Svc) -> S.RecBatchResponse:
        return S.RecBatchResponse(responses=[one(s, mid, r) for r in batch.requests])

    @app.get("/v1/explain/{impression_id}")
    def explain(impression_id: str, mid: Pub, s: Svc) -> dict[str, Any]:
        return s.explain(mid, impression_id)

    # -------------------------------------------------------------- catalog
    @app.post("/v1/catalog", response_model=S.CatalogResult)
    def catalog(up: S.CatalogUpload, mid: Priv, s: Svc) -> S.CatalogResult:
        return s.upsert_items(mid, up.items, up.replace)

    @app.post("/v1/catalog/import/{fmt}", response_model=S.CatalogResult)
    async def catalog_import(
        fmt: str, request: Request, mid: Priv, s: Svc, replace: bool = False
    ) -> S.CatalogResult:
        if fmt not in IMPORTERS:
            raise HTTPException(400, f"format must be one of {sorted(IMPORTERS)}")
        body = await request.body()
        if len(body) > 200 * 2**20:
            raise HTTPException(413, "file too large")
        items, errors = IMPORTERS[fmt](body)
        return s.upsert_items(mid, items, replace, errors)

    @app.get("/v1/catalog/categories")
    def categories(mid: Priv, s: Svc) -> list[dict[str, Any]]:
        return s.categories(mid)

    @app.get("/v1/catalog/items")
    def find_items(
        mid: Priv, s: Svc, q: str = "", ids: str = "", limit: int = 20
    ) -> list[dict[str, Any]]:
        wanted = [i for i in ids.split(",") if i][:200]
        return s.find_items(mid, q[:100], wanted, max(1, min(limit, 50)))

    # -------------------------------------------------------------- models
    @app.get("/v1/status")
    def status(mid: Priv, s: Svc) -> dict[str, Any]:
        return s.status(mid)

    @app.get("/v1/stats/daily")
    def daily(mid: Priv, s: Svc, days: int = 28) -> list[dict[str, Any]]:
        return s.daily(mid, max(1, min(days, 365)))

    @app.get("/v1/models")
    def models(mid: Priv, s: Svc) -> list[dict[str, Any]]:
        return s.models(mid)

    @app.post("/v1/retrain")
    def retrain(mid: Priv, s: Svc) -> dict[str, Any]:
        m = s._merchant(mid)
        m.dirty = m.dirty or "manual"
        return {"scheduled": s.maybe_retrain(mid)}

    # -------------------------------------------------------------- rules & graph
    @app.get("/v1/rules", response_model=S.Rules)
    def get_rules(mid: Priv, s: Svc) -> S.Rules:
        return s.get_rules(mid)

    @app.put("/v1/rules", response_model=S.Rules)
    def put_rules(rules: S.Rules, mid: Priv, s: Svc) -> S.Rules:
        return s.put_rules(mid, rules)

    @app.get("/v1/graph", response_model=list[S.GraphEdge])
    def get_graph(mid: Priv, s: Svc) -> list[S.GraphEdge]:
        return s.graph_view(mid)

    @app.put("/v1/graph", response_model=list[S.GraphEdge])
    def review_graph(review: S.GraphReview, mid: Priv, s: Svc) -> list[S.GraphEdge]:
        return s.review_graph(mid, review)

    @app.post("/v1/graph/propose")
    def propose(mid: Priv, s: Svc) -> dict[str, int]:
        try:
            return {"proposed": s.propose_graph(mid)}
        except Exception as e:  # the LLM is optional; say why it failed
            raise HTTPException(503, f"LLM unavailable: {type(e).__name__}") from None

    # -------------------------------------------------------------- experiments
    @app.post("/v1/experiments", response_model=S.Experiment)
    def create_experiment(spec: S.ExperimentCreate, mid: Priv, s: Svc) -> S.Experiment:
        try:
            return s.create_experiment(mid, spec)
        except ValueError as e:
            raise HTTPException(409, str(e)) from None

    @app.get("/v1/experiments", response_model=list[S.Experiment])
    def list_experiments(mid: Priv, s: Svc) -> list[S.Experiment]:
        return s.experiments(mid)

    @app.post("/v1/experiments/{eid}/stop", response_model=S.Experiment)
    def stop_experiment(eid: str, mid: Priv, s: Svc) -> S.Experiment:
        return s.stop_experiment(mid, eid)

    @app.get("/v1/experiments/{eid}/report")
    def report(eid: str, mid: Priv, s: Svc) -> dict[str, Any]:
        return s.report(mid, eid)

    @app.post("/v1/power")
    def power(req: S.PowerRequest, mid: Priv, s: Svc) -> dict[str, Any]:
        return s.power(mid, req)

    # -------------------------------------------------------------- shopper rights
    @app.delete("/v1/shoppers/{shopper_id}")
    def erase(shopper_id: str, mid: Priv, s: Svc) -> dict[str, int]:
        return s.erase(mid, shopper_id)

    @app.get("/v1/shoppers/{shopper_id}/export")
    def export(shopper_id: str, mid: Priv, s: Svc) -> dict[str, Any]:
        return s.export(mid, shopper_id)

    # -------------------------------------------------------------- static front end
    for name in ("ui", "widget", "dashboard"):
        d = ROOT / name
        if d.exists():
            app.mount(f"/{name}", StaticFiles(directory=d, html=True), name=name)
    demo = os.environ.get("MITATE_DEMO_SITE")  # built by `python -m serve.demo`
    if demo and Path(demo).exists():
        app.mount("/demo", StaticFiles(directory=demo, html=True), name="demo")
    return app
