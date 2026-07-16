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


def source_snapshot(root: str | Path | None = None) -> dict[str, Any]:
    """Digest available executable sources and dependency configuration.

    Includes package, experiment, and test Python sources, plus dependency
    manifests at the project root.  Files are identified by relative path
    and content, never timestamps or Git status, so uncommitted edits are
    captured.  This records source identity; it does not prove that model
    weights, external data, or the execution environment were reproduced.
    """
    source_root = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    paths: set[Path] = set()
    for directory in ("contamcheckr", "experiments", "tests"):
        candidate = source_root / directory
        if candidate.is_dir():
            # Saved experiment source snapshots and downloaded checkpoints
            # are artifacts, not the currently executing experiment sources.
            paths.update(candidate.glob("*.py") if directory == "experiments" else candidate.rglob("*.py"))
    for pattern in (
        "pyproject.toml", "requirements*.txt", "requirements*.in", "uv.lock",
        "poetry.lock", "Pipfile", "Pipfile.lock", "setup.py", "setup.cfg",
    ):
        paths.update(source_root.glob(pattern))
    contents = {
        path.relative_to(source_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
        if path.is_file()
    }
    serialized = json.dumps(contents, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "sha256": hashlib.sha256(serialized).hexdigest() if contents else None,
        "files": contents,
        "file_count": len(contents),
        "scope": "available_package_experiment_test_sources_and_dependency_manifests",
        "complete": bool(contents),
    }


def _file_signature(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"name": path.name, "missing": True}
    return {
        "name": path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def local_model_signature(model_id: str) -> dict[str, Any] | None:
    """Signature for a local model directory.

    The signature uses file sizes and modification times rather than hashing
    gigabytes of weights.  A retrained model changes its file timestamps, which
    invalidates the cache.
    """
    path = Path(model_id).expanduser()
    if not path.is_dir():
        return None
    files: list[dict[str, Any]] = []
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ):
        candidate = path / name
        if candidate.exists():
            files.append(_file_signature(candidate))
    for candidate in sorted(path.glob("*.safetensors")):
        files.append(_file_signature(candidate))
    for candidate in sorted(path.glob("*.bin")):
        files.append(_file_signature(candidate))
    return {
        "path": str(path.resolve()),
        "files": files,
        "identity_kind": "file_size_and_mtime_signature",
        "cryptographic_weights": False,
    }
