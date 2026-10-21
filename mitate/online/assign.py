"""Deterministic, stateless traffic assignment.

A visitor's arm is a pure function of (experiment salt, pseudonymous visitor id), so the same
shopper always sees the same arm across devices that share the id, nothing needs storing, and
deleting a visitor's data cannot change anyone else's assignment.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence


def unit_hash(salt: str, visitor_id: str) -> float:
    """Uniform in [0, 1) from SHA-256; stable across processes and Python versions."""
    h = hashlib.sha256(f"{salt}\x1f{visitor_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def assign(salt: str, visitor_id: str, arms: Sequence[tuple[str, float]]) -> str:
    """Pick an arm by cumulative share, e.g. ``[("holdout", .1), ("mitate", .9)]``."""
    total = sum(w for _, w in arms)
    if not arms or total <= 0:
        raise ValueError("arms must have positive total share")
    x = unit_hash(salt, visitor_id) * total
    acc = 0.0
    for name, w in arms:
        acc += w
        if x < acc:
            return name
    return arms[-1][0]
