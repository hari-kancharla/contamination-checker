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


class ScoreCache:
    """Thread-safe SQLite cache."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser() if path is not None else default_cache_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(
            self.path,
            check_same_thread=False,
            timeout=30.0,
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    def _get(self, table: str, key: str, column: str) -> Any | None:
        with self._lock:
            row = self._connection.execute(
                f"SELECT {column} FROM {table} WHERE key = ?",  # noqa: S608
                (key,),
            ).fetchone()
        if row is None:
            return None
        try:
            return row[0]
        except (TypeError, IndexError):  # pragma: no cover - defensive
            return None

    def _put(self, table: str, key: str, column: str, value: Any) -> None:
        with self._lock:
            self._connection.execute(
                f"INSERT OR REPLACE INTO {table} (key, {column}, created_at) "  # noqa: S608
                "VALUES (?, ?, ?)",
                (key, value, time.time()),
            )
            self._connection.commit()

    def get_token_stats(self, key: str) -> dict[str, Any] | None:
        payload = self._get("token_stats", key, "payload")
        if payload is None:
            return None
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict):
            return None
        return value

    def put_token_stats(self, key: str, payload: dict[str, Any]) -> None:
        self._put(
            "token_stats",
            key,
            "payload",
            json.dumps(payload, separators=(",", ":"), allow_nan=False),
        )

    def get_sequence_ll(self, key: str) -> float | None:
        value = self._get("sequence_ll", key, "value")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def put_sequence_ll(self, key: str, value: float) -> None:
        self._put("sequence_ll", key, "value", float(value))

    def get_generation(self, key: str) -> str | None:
        value = self._get("generations", key, "value")
        return str(value) if value is not None else None

    def put_generation(self, key: str, value: str) -> None:
        self._put("generations", key, "value", str(value))

    def count(self, table: str) -> int:
        if table not in {"token_stats", "sequence_ll", "generations"}:
            raise ValueError("Unknown cache table.")
        with self._lock:
            row = self._connection.execute(
                f"SELECT COUNT(*) FROM {table}"  # noqa: S608
            ).fetchone()
        return int(row[0]) if row else 0

    def stats(self) -> dict[str, int]:
        return {
            table: self.count(table)
            for table in ("token_stats", "sequence_ll", "generations")
        }

    def clear(self) -> None:
        with self._lock:
            for table in ("token_stats", "sequence_ll", "generations"):
                self._connection.execute(f"DELETE FROM {table}")  # noqa: S608
            self._connection.commit()
            self._connection.execute("VACUUM")

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "ScoreCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
