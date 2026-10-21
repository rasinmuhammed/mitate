"""Catalog importers: Mitate CSV, Shopify product CSV export, WooCommerce product CSV export.

Files are parsed locally; Mitate never calls a shop's API, so no shop credentials are needed.
Each importer returns validated ``Item``s plus row-level errors (bad rows are reported, never
silently dropped).

* **Mitate CSV**: ``id,title,description,category,brand,price,stock,regions,launched_at,
  image_url``; ``category`` is a path ``A > B > C``; ``regions`` is ``;``-separated PIN prefixes.
* **Shopify** (Products > Export): one row per variant/image, grouped by ``Handle``. Title and
  body come from the first row; price is the lowest variant price; stock is the sum of
  ``Variant Inventory Qty`` when Shopify tracks inventory; category is ``Product Category``
  (Shopify's standard taxonomy) else ``Type``; drafts and archived products are inactive; the
  image is the first ``Image Src``.
* **WooCommerce** (Products > Export): parents and simple products become items; variations
  (``Type=variation``, ``Parent=id:N``) contribute their price and stock to the parent. The first
  entry of ``Categories`` is used (``A > B``); unpublished products are inactive; the image is
  the first URL in ``Images``.

Image URLs must be ``https://`` or a path on the shop's own site. Anything else (``http://``,
``data:``) is dropped from the item rather than failing the row: the product still imports.
"""

from __future__ import annotations

import csv
import html
import io
import re
from collections.abc import Callable, Iterable
from typing import Any

from pydantic import ValidationError

from serve.schemas import Item

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def strip_html(s: str) -> str:
    return _WS.sub(" ", html.unescape(_TAG.sub(" ", s or ""))).strip()


def _num(s: Any) -> float | None:
    if s is None:
        return None
    t = str(s).strip().replace(",", "")
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _image(s: str | None) -> str | None:
    """First URL of a field, kept only if it is https or site-relative (see the module notes)."""
    url = (s or "").split(",")[0].strip()
    if url.startswith("http://"):
        return None
    return url if url.startswith(("https://", "/")) and not _WS.search(url) else None


def _rows(data: bytes | str) -> Iterable[dict[str, str]]:
    text = data.decode("utf-8-sig") if isinstance(data, bytes) else data
    return csv.DictReader(io.StringIO(text))


def _build(rows: Iterable[tuple[int, dict[str, Any]]]) -> tuple[list[Item], list[str]]:
    items, errors = [], []
    for line, r in rows:
        try:
            items.append(Item(**r))
        except ValidationError as e:
            errors.append(f"row {line}: {e.errors()[0]['loc']} {e.errors()[0]['msg']}")
    return items, errors


def mitate_csv(data: bytes | str) -> tuple[list[Item], list[str]]:
    def conv(r: dict[str, str]) -> dict[str, Any]:
        regions = [x.strip() for x in (r.get("regions") or "").split(";") if x.strip()]
        launched = _num(r.get("launched_at"))
        return {
            "id": r.get("id", ""),
            "title": r.get("title", ""),
            "description": strip_html(r.get("description", "")),
            "category": r.get("category", "") or "",
            "brand": r.get("brand", "") or "",
            "price": _num(r.get("price")),
            "stock": _num(r.get("stock")),
            "regions": regions or None,
            "launched_at": int(launched) if launched is not None else None,
            "image_url": _image(r.get("image_url")),
        }

    return _build((i + 2, conv(r)) for i, r in enumerate(_rows(data)))


def shopify_csv(data: bytes | str) -> tuple[list[Item], list[str]]:
    groups: dict[str, dict[str, Any]] = {}
    for i, r in enumerate(_rows(data)):
        h = (r.get("Handle") or "").strip()
        if not h:
            continue
        g = groups.setdefault(h, {"line": i + 2, "prices": [], "stock": [], "tracked": False})
        if r.get("Title") and "title" not in g:
            g["title"] = r["Title"]
            g["description"] = strip_html(r.get("Body (HTML)", ""))
            g["brand"] = r.get("Vendor", "") or ""
            g["category"] = (r.get("Product Category") or r.get("Type") or "").strip()
            status = (r.get("Status") or "").strip().lower()
            published = (r.get("Published") or "").strip().lower()
            g["active"] = status == "active" if status else published in ("true", "1", "")
        if "image_url" not in g and (img := _image(r.get("Image Src"))):
            g["image_url"] = img
        if (p := _num(r.get("Variant Price"))) is not None:
            g["prices"].append(p)
        if (r.get("Variant Inventory Tracker") or "").strip():
            g["tracked"] = True
            g["stock"].append(_num(r.get("Variant Inventory Qty")) or 0.0)

    def conv(h: str, g: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": h,
            "title": g.get("title", ""),
            "description": g.get("description", ""),
            "category": g.get("category", ""),
            "brand": g.get("brand", ""),
            "price": min(g["prices"]) if g["prices"] else None,
            "stock": sum(g["stock"]) if g["tracked"] else None,
            "active": g.get("active", True),
            "image_url": g.get("image_url"),
        }

    return _build((g["line"], conv(h, g)) for h, g in groups.items())


def woocommerce_csv(data: bytes | str) -> tuple[list[Item], list[str]]:
    parents: dict[str, dict[str, Any]] = {}
    variations: list[dict[str, str]] = []
    for i, r in enumerate(_rows(data)):
        kind = (r.get("Type") or "").strip().lower()
        if "variation" in kind.split(",")[0]:
            variations.append(r)
            continue
        pid = (r.get("ID") or r.get("SKU") or "").strip()
        if not pid:
            continue
        price = _num(r.get("Sale price")) or _num(r.get("Regular price"))
        stock = _num(r.get("Stock"))
        if stock is None and (r.get("In stock?") or "").strip() == "0":
            stock = 0.0
        cats = (r.get("Categories") or "").split(",")
        parents[pid] = {
            "line": i + 2,
            "id": pid,
            "title": r.get("Name", "") or "",
            "description": strip_html(r.get("Description") or r.get("Short description") or ""),
            "category": cats[0].strip() if cats else "",
            "brand": (r.get("Brands") or "").split(",")[0].strip(),
            "price": price,
            "stock": stock,
            "active": (r.get("Published") or "1").strip() == "1",
            "image_url": _image(r.get("Images")),
        }
    for v in variations:
        parent = (v.get("Parent") or "").strip()
        pid = parent[3:] if parent.startswith("id:") else parent
        p = parents.get(pid)
        if p is None:
            continue
        vp = _num(v.get("Sale price")) or _num(v.get("Regular price"))
        if vp is not None:
            p["price"] = vp if p["price"] is None else min(p["price"], vp)
        vs = _num(v.get("Stock"))
        if vs is not None:
            p["stock"] = (p["stock"] or 0.0) + vs

    def conv(p: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return p.pop("line"), p

    return _build(conv(dict(p)) for p in parents.values())


IMPORTERS: dict[str, Callable[[bytes | str], tuple[list[Item], list[str]]]] = {
    "mitate": mitate_csv,
    "shopify": shopify_csv,
    "woocommerce": woocommerce_csv,
}
