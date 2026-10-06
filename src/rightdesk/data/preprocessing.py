"""Duplicate keys only; model-facing text is never changed here."""

import hashlib
import unicodedata

NORMALIZATION_VERSION = "nfkc-casefold-whitespace-v1"


def duplicate_key(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
