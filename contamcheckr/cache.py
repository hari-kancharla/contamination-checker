"""Content-addressed SQLite cache for expensive model scores.

The cache stores only hashes as keys and numeric/text results as values.  It is
safe to delete at any time.  A corrupt row is treated as a cache miss rather
than as an error.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

from .hashing import stable_hash


_SCHEMA = """
CREATE TABLE IF NOT EXISTS token_stats (
    key TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sequence_ll (
    key TEXT PRIMARY KEY,
    value REAL NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS generations (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


def cache_key(*parts: Any) -> str:
    """Join cache-key components into a stable 32-character key."""
    return stable_hash(
        "\x00".join(str(part) for part in parts),
        length=32,
    )


def default_cache_path() -> Path:
    """Default on-disk cache location."""
    override = os.environ.get("CONTAMCHECKR_CACHE_DIR")
    if override:
        return Path(override).expanduser() / "scores.sqlite3"
    return Path.home() / ".cache" / "contamcheckr" / "scores.sqlite3"
