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


def resolved_commit_hash(loaded: Any) -> str | None:
    """Return the resolved Hub commit exposed by a loaded model/tokenizer."""
    candidates = [
        getattr(getattr(loaded, "config", None), "_commit_hash", None),
        getattr(loaded, "_commit_hash", None),
    ]
    init_kwargs = getattr(loaded, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        candidates.append(init_kwargs.get("_commit_hash"))
    return next((value for value in candidates if isinstance(value, str) and value), None)


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Fingerprint vocabulary and exposed tokenization processing rules.

    Fast-tokenizer serialization contains merges, normalizers, and the
    pre/post processors.  Slow BPE and SentencePiece tokenizers expose
    equivalent rules through different APIs.  Exotic tokenizers that expose
    none of these remain a best-effort identity rather than a proof.
    """
    pieces: list[str] = [
        str(getattr(tokenizer, "name_or_path", "")),
        str(getattr(tokenizer, "vocab_size", "")),
        str(getattr(tokenizer, "model_max_length", "")),
        str(getattr(tokenizer, "pad_token_id", "")),
        str(getattr(tokenizer, "eos_token_id", "")),
        str(getattr(tokenizer, "bos_token_id", "")),
        str(getattr(tokenizer, "unk_token_id", "")),
        str(getattr(tokenizer, "padding_side", "")),
        str(getattr(tokenizer, "truncation_side", "")),
        str(getattr(tokenizer, "chat_template", "")),
        str(resolved_commit_hash(tokenizer) or ""),
    ]
    try:
        vocabulary = tokenizer.get_vocab()
    except Exception:  # pragma: no cover - exotic tokenizers
        vocabulary = {}
    if vocabulary:
        digest = hashlib.sha256()
        for token, identifier in sorted(vocabulary.items()):
            digest.update(str(token).encode("utf-8"))
            digest.update(b"\x00")
            digest.update(str(identifier).encode("utf-8"))
            digest.update(b"\x01")
        pieces.append(digest.hexdigest())
    settings = {}
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    for name in (
        "do_lower_case", "strip_accents", "tokenize_chinese_chars", "add_prefix_space",
        "clean_up_tokenization_spaces", "legacy", "split_special_tokens",
        "spaces_between_special_tokens", "add_bos_token", "add_eos_token",
    ):
        value = getattr(tokenizer, name, None)
        if value is None and isinstance(init_kwargs, dict):
            value = init_kwargs.get(name)
        if value is not None:
            settings[name] = value
    settings["special_tokens_map"] = getattr(tokenizer, "special_tokens_map", {})
    pieces.append(json.dumps(settings, sort_keys=True, default=str, separators=(",", ":")))
    backend = getattr(tokenizer, "backend_tokenizer", None)
    serialize_backend = getattr(backend, "to_str", None)
    if callable(serialize_backend):
        serialized = serialize_backend()
        try:
            backend_settings = json.loads(serialized)
        except (TypeError, json.JSONDecodeError):
            backend_settings = serialized
        if isinstance(backend_settings, dict):
            # These call-time limits are recorded by the scoring parameters;
            # they can be mutated by ordinary encode calls on HF tokenizers.
            backend_settings.pop("padding", None)
            backend_settings.pop("truncation", None)
        pieces.append(json.dumps(backend_settings, sort_keys=True, separators=(",", ":")))
    bpe_ranks = getattr(tokenizer, "bpe_ranks", None)
    if isinstance(bpe_ranks, dict):
        pieces.append(json.dumps(
            sorted((list(pair), int(rank)) for pair, rank in bpe_ranks.items()),
            ensure_ascii=False, separators=(",", ":"),
        ))
    sentencepiece = getattr(tokenizer, "sp_model", None)
    serialize_sentencepiece = getattr(sentencepiece, "serialized_model_proto", None)
    if callable(serialize_sentencepiece):
        pieces.append(stable_hash(serialize_sentencepiece(), length=64))
    return stable_hash("\x00".join(pieces), length=32)


def model_fingerprint(
    *,
    model_id: str,
    revision: str | None,
    dtype: str,
    tokenizer: Any,
    parameter_count: int | None = None,
    transformers_version: str | None = None,
    resolved_revision: str | None = None,
    tokenizer_revision: str | None = None,
) -> str:
    """Best-effort cache identity using resolved revisions when available.

    Local weights use metadata signatures, not cryptographic content hashes.
    Missing resolved Hub revisions are retained as unknown, never inferred
    from a mutable model name or branch.
    """
    payload = {
        "fingerprint_schema": 2,
        "model_id": model_id,
        "revision": revision,
        "resolved_revision": resolved_revision,
        "tokenizer_revision": tokenizer_revision or resolved_commit_hash(tokenizer),
        "dtype": dtype,
        "tokenizer": tokenizer_fingerprint(tokenizer),
        "parameter_count": parameter_count,
        "transformers_version": transformers_version,
        "local_signature": local_model_signature(model_id),
        "cwd": os.getcwd() if Path(model_id).expanduser().is_dir() else None,
    }
    return stable_hash(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        length=32,
    )


def config_fingerprint(values: dict[str, Any]) -> str:
    """Stable fingerprint of a configuration dictionary."""
    return stable_hash(
        json.dumps(values, sort_keys=True, default=str, separators=(",", ":")),
        length=32,
    )
