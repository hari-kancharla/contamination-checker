"""Sealed, randomized experiment for *additional fine-tuning* exposure.

Only the ``run`` command loads or trains a model.  It accepts an existing
local checkpoint and local JSONL pool, never downloads a model or dataset,
and creates a new run directory.  See docs/RANDOMIZED_EXPOSURE.md.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
from typing import Any, Callable


PROJECT = Path(__file__).resolve().parent.parent
SCHEMA = "contamcheckr.randomized-exposure.v1"
SCORES = ("loss", "mink", "minkpp")
ESTIMAND = (
    "Discrimination of randomized additional fine-tuning exposure within a "
    "fixed local pool; not unknown pretraining membership, false-positive "
    "calibration, or benchmark-performance inflation."
)


@dataclass(frozen=True)
class Settings:
    assignment_seed: int = 17
    training_seed: int = 17
    bootstrap_seed: int = 101
    exposed_fraction: float = 0.5
    group_field: str | None = None
    epochs: int = 1
    learning_rate: float = 0.00005
    training_batch_size: int = 2
    scoring_batch_size: int = 2
    max_tokens: int = 256
    k: float = 20.0
    bootstrap_draws: int = 2000
    device: str = "cpu"

    def validate(self) -> None:
        if not 0 < self.exposed_fraction < 1:
            raise ValueError("exposed_fraction must be strictly between zero and one.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite.")
        if not 0 < self.k <= 100:
            raise ValueError("k must be in (0, 100].")
        for name in ("epochs", "training_batch_size", "scoring_batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least one.")
        if self.max_tokens < 2 or self.bootstrap_draws < 20:
            raise ValueError("max_tokens must be >= 2 and bootstrap_draws >= 20.")
        if self.group_field in {"id", "text"}:
            raise ValueError("group_field must be a separate input metadata field.")
        if self.group_field is not None and not self.group_field.strip():
            raise ValueError("group_field cannot be empty.")
        if self.device not in {"cpu", "mps", "cuda", "auto"}:
            raise ValueError("Unsupported device.")


@dataclass
class Runtime:
    load: Callable[..., Any]
    train: Callable[..., dict[str, Any]]
    score: Callable[..., list[Any]]
    release: Callable[[Any], None]


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("xb") as handle:
        handle.write(_bytes(value))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("xb") as handle:
        for row in rows:
            handle.write(_bytes(row))


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_pool(path: Path, *, group_field: str | None = None) -> list[dict[str, str]]:
    """Validate one pool without inferring labels from text, IDs, or metadata."""
    return _read_pool_payload(path.read_bytes(), group_field=group_field)


def _read_pool_payload(payload: bytes, *, group_field: str | None) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Pool row {line_number} must be an object.")
        identifier, text = row.get("id"), row.get("text")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"Pool row {line_number} requires a nonempty string id.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Pool row {line_number} requires nonempty text.")
        if identifier in seen_ids:
            raise ValueError(f"Duplicate id: {identifier}")
        normalized = " ".join(text.split())
        if normalized in seen_texts:
            raise ValueError(f"Duplicate text (including whitespace variants): {identifier}")
        seen_ids.add(identifier)
        seen_texts.add(normalized)
        group = row.get(group_field) if group_field else identifier
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f"Pool row {line_number} requires a nonempty group string.")
        records.append({"id": identifier, "text": text, "group_id": group})
    if len(records) < 4:
        raise ValueError("Pool requires at least four distinct examples.")
    return sorted(records, key=lambda row: row["id"])


def assign_exposure(records: list[dict[str, str]], settings: Settings) -> list[dict[str, Any]]:
    settings.validate()
    groups = sorted({row["group_id"] for row in records})
    random.Random(settings.assignment_seed).shuffle(groups)
    count = int(len(groups) * settings.exposed_fraction)
    if count < 2 or len(groups) - count < 2:
        raise ValueError("Assignment needs at least two groups per arm; enlarge the pool or change fraction.")
    exposed = set(groups[:count])
    return [
        {
            "id": row["id"], "group_id": row["group_id"],
            "text_sha256": _sha(row["text"].encode("utf-8")),
            "exposed": row["group_id"] in exposed,
        }
        for row in records
    ]
