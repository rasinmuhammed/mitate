"""Pseudonymisation and processing gates (docs/privacy.md P3-P5, P8).

* Shopper and session ids are replaced by ``HMAC-SHA256(merchant secret, id)`` before anything is
  stored or logged. The raw id is never written anywhere.
* ``Gate`` decides, per request, whether behavioural data may be written and used. Only
  ``consent=granted`` and an age status that is not (treated as) a minor allow it.
* API keys are stored as SHA-256 digests only.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass

from serve.schemas import AgeStatus, Consent


def pseudonym(secret: bytes, raw: str) -> str:
    return hmac.new(secret, raw.encode(), hashlib.sha256).hexdigest()[:32]


def new_secret() -> bytes:
    return secrets.token_bytes(32)


def new_api_key(prefix: str) -> str:
    return f"{prefix}_{secrets.token_urlsafe(24)}"


def key_digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


@dataclass(frozen=True)
class Gate:
    personalise: bool  # may read and write this shopper's behaviour
    reason: str

    @staticmethod
    def of(consent: Consent, age: AgeStatus, minor_safe_for_unknown_age: bool) -> Gate:
        if age is AgeStatus.MINOR or (age is AgeStatus.UNKNOWN and minor_safe_for_unknown_age):
            return Gate(False, "minor_safe")  # s.9(3): no behavioural monitoring of children
        if consent is not Consent.GRANTED:
            return Gate(False, f"consent_{consent.value}")
        return Gate(True, "granted")
