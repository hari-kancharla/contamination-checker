"""Stable content fingerprints used by the cache and report manifest."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def stable_hash(value: str | bytes, *, length: int = 32) -> str:
    """SHA-256 of ``value``, truncated to ``length`` hex characters."""
    if length < 8 or length > 64:
        raise ValueError("Hash length must be between 8 and 64.")
    if isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = value
    return hashlib.sha256(payload).hexdigest()[:length]


def hash_text(text: str) -> str:
    """Content hash for an exact text string."""
    return stable_hash(text, length=32)


def hash_examples(examples: Iterable[dict[str, str]]) -> str:
    """Order-independent fingerprint of a set of examples.

    IDs and texts are both included.  The result changes if a text is edited,
    an example is added, or an ID is changed, but not if examples are reordered.
    """
    records = sorted(
        (str(example.get("id", "")), str(example.get("text", "")))
        for example in examples
    )
    digest = hashlib.sha256()
    for identifier, text in records:
        digest.update(identifier.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
        digest.update(b"\x01")
    return digest.hexdigest()[:32]


def hash_ordered_examples(examples: Iterable[dict[str, str]]) -> str:
    """Fingerprint exact record order as well as example IDs and texts.

    Canonical-order tests and cyclic prefix controls depend on order.  Keep
    this alongside :func:`hash_examples`, whose legacy set semantics are
    useful for recognizing the same examples in a different order.
    """
    digest = hashlib.sha256()
    for example in examples:
        payload = json.dumps(
            [str(example.get("id", "")), str(example.get("text", ""))],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()[:32]
