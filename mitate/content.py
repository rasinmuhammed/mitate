"""Catalog-only signals: item text, text embeddings (cached), and the content kNN recommender.

Embeddings are a pure function of catalog text, so they cannot leak interactions. They are cached
on disk keyed by (model, dim, text hash), which makes re-running the benchmark cheap and keeps the
serving path free of model inference for existing items.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Self, cast

import numpy as np
from numpy.typing import NDArray

from mitate.data import Catalog, Log, Queries
from mitate.models.base import F32, Recommender

EMBEDDERS: dict[str, dict[str, Any]] = {
    # name -> sentence-transformers id, query/doc prompts, default output dim (MRL truncation)
    "qwen3-0.6b": {"id": "Qwen/Qwen3-Embedding-0.6B", "dim": 512, "prefix": ""},
    "e5-small": {"id": "intfloat/multilingual-e5-small", "dim": 384, "prefix": "passage: "},
    "potion-multi": {"id": "minishlab/potion-multilingual-128M", "dim": 256, "prefix": ""},
}
CACHE = Path(os.environ.get("MITATE_CACHE", Path.home() / ".cache" / "mitate"))


def item_texts(cat: Catalog) -> list[str]:
    """``Title; Category; Brand; Price band; Description`` with empty parts dropped."""
    band = cat.price_band()
    names = ["budget", "lower-mid", "mid", "upper-mid", "premium"]
    out = []
    for i in range(cat.n_items):
        parts = []
        if cat.title is not None and cat.title[i]:
            parts.append(str(cat.title[i]))
        if cat.category is not None and cat.category_names is not None and cat.category[i] >= 0:
            parts.append(f"Category: {cat.category_names[int(cat.category[i])]}")
        if cat.brand is not None and cat.brand_names is not None and cat.brand[i] >= 0:
            parts.append(f"Brand: {cat.brand_names[int(cat.brand[i])]}")
        if band[i] >= 0:
            parts.append(f"Price: {names[min(int(band[i]), 4)]}")
        if cat.text is not None and cat.text[i]:
            parts.append(str(cat.text[i])[:1000])
        out.append("; ".join(parts) if parts else f"item {i}")
    return out


def _key(model: str, dim: int, texts: Sequence[str]) -> str:
    h = hashlib.sha256()
    h.update(f"{model}|{dim}|".encode())
    for t in texts:
        h.update(t.encode())
        h.update(b"\x00")
    return h.hexdigest()[:24]


def embed(
    texts: Sequence[str], model: str = "e5-small", batch_size: int = 64
) -> NDArray[np.float32]:
    """L2-normalised embeddings, cached on disk."""
    spec = EMBEDDERS[model]
    path = CACHE / "emb" / f"{model}-{_key(model, spec['dim'], texts)}.npy"
    if path.exists():
        return cast(NDArray[np.float32], np.load(path))
    if model.startswith("potion"):
        from model2vec import StaticModel

        sm = StaticModel.from_pretrained(spec["id"])
        V = np.asarray(sm.encode(list(texts)), np.float32)
    else:
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(spec["id"], device=_device(), truncate_dim=spec["dim"])
        V = st.encode(
            [spec["prefix"] + t for t in texts],
            batch_size=batch_size,
            normalize_embeddings=False,
            show_progress_bar=len(texts) > 5000,
            convert_to_numpy=True,
        ).astype(np.float32)
    V = V[:, : spec["dim"]]
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, V)
    return V


def _device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class ContentKNN(Recommender):
    """Scores items by cosine similarity to a recency-weighted mean of the history's embeddings.

    Needs no interactions at all, so it serves new stores and new items. ``vectors`` may be
    injected (benchmark caches them per dataset); otherwise they are computed from the catalog.
    """

    name: ClassVar[str] = "content_knn"
    catalog_only: ClassVar[bool] = True
    embedder: str = "e5-small"
    decay: float = 0.7  # weight of the j-th most recent item is decay**(j-1)
    max_len: int = 10
    vectors: NDArray[np.float32] | None = field(default=None, repr=False)

    @classmethod
    def space(cls, trial: Any) -> dict[str, Any]:
        return {
            "decay": trial.suggest_float("decay", 0.0, 1.0),
            "max_len": trial.suggest_categorical("max_len", [1, 3, 5, 10, 20]),
        }

    def fit(self, log: Log, catalog: Catalog) -> Self:
        if self.vectors is None:
            self.vectors = embed(item_texts(catalog), self.embedder)
        return self

    def score(self, q: Queries) -> F32:
        V = self.vectors
        assert V is not None
        P = np.zeros((len(q), V.shape[1]), np.float32)
        for r in range(len(q)):
            h = q.history(r)
            if q.anchor[r] >= 0:
                h = np.append(h, q.anchor[r])
            h = h[-self.max_len :][::-1]
            if h.size == 0:
                continue
            w = np.float32(self.decay) ** np.arange(h.size, dtype=np.float32)  # 0**0 == 1
            P[r] = w @ V[h]
        return np.asarray(P @ V.T, np.float32)
