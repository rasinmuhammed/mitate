"""Request and response schemas. Every inbound model forbids unknown fields, so personal data a
client tries to send (names, emails, addresses, free text) is rejected, not silently stored
(docs/privacy.md, P1). Sensitive attributes have no field to arrive in (P2)."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Id = Annotated[str, StringConstraints(min_length=1, max_length=128, strip_whitespace=True)]
Region = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9-]{1,6}$")]  # PIN prefix / code
# product image: an https URL or a path on the merchant's own site; no data: or javascript: URLs
ImageUrl = Annotated[
    str, StringConstraints(max_length=1000, pattern=r"^(https://|/)[^\s\"'<>\\]*$")
]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Consent(StrEnum):
    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"
    UNKNOWN = "unknown"


class AgeStatus(StrEnum):
    ADULT_VERIFIED = "adult_verified"
    ADULT_DECLARED = "adult_declared"
    MINOR = "minor"
    UNKNOWN = "unknown"


class EventType(StrEnum):
    VIEW = "view"
    CART = "cart"
    ORDER = "order"
    CLICK = "click"  # click on a recommendation (needs impression_id); not a training signal


class Context(Strict):
    channel: Literal["web", "app", "email", "other"] | None = None
    device_class: Literal["mobile", "desktop", "tablet", "other"] | None = None
    region: Region | None = None


class Event(Strict):
    shopper_id: Id | None = None  # merchant's own opaque id; stored only as an HMAC
    session_id: Id
    item_id: Id
    type: EventType
    ts: int | None = Field(default=None, description="unix seconds; server time if omitted")
    value: float | None = Field(default=None, ge=0, description="order line value (orders only)")
    impression_id: Id | None = None
    consent: Consent = Consent.UNKNOWN
    age_status: AgeStatus = AgeStatus.UNKNOWN
    context: Context = Context()


class EventBatch(Strict):
    events: list[Event] = Field(max_length=1000)


class IngestResult(Strict):
    accepted: int
    dropped_no_consent: int
    dropped_unknown_item: int


class Item(Strict):
    id: Id
    title: Annotated[str, StringConstraints(max_length=300)]
    description: Annotated[str, StringConstraints(max_length=5000)] = ""
    category: Annotated[str, StringConstraints(max_length=300)] = ""  # "A > B > C"
    brand: Annotated[str, StringConstraints(max_length=120)] = ""
    price: float | None = Field(default=None, ge=0)
    stock: float | None = None  # None = not tracked
    regions: list[Region] | None = None  # None = ships everywhere
    launched_at: int | None = None
    active: bool = True
    image_url: ImageUrl | None = None


class CatalogUpload(Strict):
    items: list[Item] = Field(max_length=100_000)
    replace: bool = False  # True: items missing from the upload become inactive


class CatalogResult(Strict):
    upserted: int
    deactivated: int
    n_active: int
    errors: list[str] = []


class Placement(StrEnum):
    SIMILAR = "similar"
    COMPLEMENTS = "complements"
    NEXT = "next"
    RETURNING = "returning"


class RecItem(Strict):
    id: str
    title: str
    price: float | None
    reason: str | None
    reason_code: str | None
    image_url: str | None = None


class XRayItem(Strict):
    id: str
    pinned: bool
    reason_code: str | None
    evidence: dict[str, object]  # the counted facts behind the reason, in merchant ids and names


class XRay(Strict):
    """What the engine did for one request; returned only when the merchant allows it."""

    stage: str
    sources: list[str]  # candidate sources active at this store's data volume
    ranker: str  # learned ranker variant, "rrf" (label-free fusion) or "none" (catalog only)
    events_learned_from: int
    trained_at: int
    personalisation: str  # why the list is or is not personal: consent, age, history
    history_events: int  # this shopper's events the engine conditioned on (0 if not personal)
    hidden: dict[str, int]  # better-scored items the merchant's rules held back, by rule
    items: list[XRayItem]
    server_ms: float


class RecResponse(Strict):
    impression_id: str
    placement: Placement
    items: list[RecItem]
    personalised: bool
    model_version: int | None
    stage: str
    arm: str | None = None
    xray: XRay | None = None


class Rules(Strict):
    blocked_items: list[Id] = []
    blocked_categories: list[str] = []
    pins: dict[Placement, list[Id]] = {}
    hide_untracked_stock: bool = False
    max_per_category: int | None = Field(default=None, ge=1)
    complement_max_price_ratio: float | None = Field(default=3.0, gt=0)
    similar_price_ratio: float | None = Field(default=2.0, ge=1)
    minor_safe_for_unknown_age: bool = False  # child-heavy stores should set True (P5)
    retention_days: int = Field(default=180, ge=1, le=3650)  # behavioural data TTL (P7)
    allow_xray: bool = False  # storefront requests may ask what the engine did (no personal data)


class GraphEdge(Strict):
    src: str
    dst: str
    status: Literal["proposed", "approved", "rejected"]
    origin: str = "merchant"


class GraphReview(Strict):
    edges: list[GraphEdge]


class ExperimentKind(StrEnum):
    HOLDOUT = "holdout"  # A/B: share of visitors get no recommendations (or popularity)
    INTERLEAVE = "interleave"  # team-draft: current model vs a comparison ranker


class ExperimentCreate(Strict):
    name: Annotated[str, StringConstraints(min_length=1, max_length=80)]
    kind: ExperimentKind
    placement: Placement | None = None  # None = all placements
    holdout_share: float = Field(default=0.1, gt=0, lt=1)
    holdout_serves: Literal["nothing", "popularity"] = "nothing"
    comparison: Literal["popularity", "previous_model"] = "popularity"  # interleaving rival


class Experiment(Strict):
    id: str
    name: str
    kind: ExperimentKind
    placement: Placement | None
    holdout_share: float
    holdout_serves: str
    comparison: str
    status: Literal["running", "stopped"]
    started_at: int
    stopped_at: int | None


class PowerRequest(Strict):
    daily_visitors: float = Field(gt=0)
    rel_mde: float = Field(default=0.05, gt=0, lt=1)
    horizon_days: int = Field(default=28, ge=1, le=365)
    holdout_share: float = Field(default=0.1, gt=0, lt=1)


class RecRequest(Strict):
    placement: Placement
    item_ids: list[Id] = Field(default=[], max_length=50)  # product on screen, or cart contents
    shopper_id: Id | None = None
    session_id: Id | None = None
    consent: Consent = Consent.UNKNOWN
    age_status: AgeStatus = AgeStatus.UNKNOWN
    region: Region | None = None
    k: int = Field(default=8, ge=1, le=50)
    xray: bool = False  # honoured only when the merchant's rules allow it


class RecBatch(Strict):
    """Every placement on a page in one round trip."""

    requests: list[RecRequest] = Field(min_length=1, max_length=8)


class RecBatchResponse(Strict):
    responses: list[RecResponse]


class MerchantCreate(Strict):
    name: Annotated[str, StringConstraints(min_length=1, max_length=120)]
