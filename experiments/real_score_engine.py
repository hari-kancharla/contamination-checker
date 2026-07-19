"""Inspectible, local-only scoring of externally labeled real model inputs.

This module does not infer membership labels, train models, or execute upstream
code.  All variants consume one shared model forward pass.  See
``docs/REAL_SCORING.md`` for the exact conventions and their limits.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import importlib.metadata
import json
import math
from numbers import Integral
from pathlib import Path
import resource
import sys
import time
from typing import Any, Iterable


ENGINE_VERSION = "contamcheckr.real-score.v3"
AUTHOR_REVISION = "6eb88de34a09dea85c4b41deff4048e5a203e1b0"
AUTHOR_SOURCE = f"https://github.com/LuckyDaydreamer/OLMo-Detect/tree/{AUTHOR_REVISION}"
PANDORA_REVISION = "189a1f43fed580f002ebba3c894c8509f9da14b0"
PANDORA_SOURCE = f"https://github.com/safr-ai-lab/pandora_llm/tree/{PANDORA_REVISION}/src/pandora_llm/features"
VARIANTS = (
    "author_native",
    "fp32_second_moment",
    "fp32_centered",
    "fixed_fp32_logp_bf16_reductions",
)
VARIANCE_FLOOR = 1e-9
DENOMINATOR_EPSILON = 1e-9


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _digest_ids(values: Iterable[int]) -> str:
    return hashlib.sha256(_json_bytes(list(values))).hexdigest()


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids(values: Any) -> tuple[int, ...]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple)) or any(
        not isinstance(value, Integral) or isinstance(value, bool) or value < 0
        for value in values
    ):
        raise ValueError("Token IDs must be a one-dimensional sequence of nonnegative integers.")
    return tuple(int(value) for value in values)


@dataclass(frozen=True)
class PreparedInput:
    identifier: str
    input_mode: str
    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    text: str | None
    convention: dict[str, Any]
    original_token_ids: tuple[int, ...] | None = None


@dataclass
class LocalCheckpoint:
    model: Any
    tokenizer: Any
    device: Any
    context_limit: int
    identity: dict[str, Any]


def prepare_olmo_plaintext(
    tokenizer: Any, text: str, identifier: str, *, max_tokens: int = 4096,
) -> PreparedInput:
    """Mirror the pinned author's plain input path, without chat formatting."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Plaintext must be a nonempty string.")
    raw_text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    # Upstream data_utils.extract_text strips before tokenization and Zlib.
    text = text.strip()
    if max_tokens < 2 or max_tokens > 4096:
        raise ValueError("OLMo plaintext max_tokens must be between 2 and 4096.")
    if getattr(tokenizer, "truncation_side", "right") != "right":
        raise ValueError("The author-compatible plaintext path requires right truncation.")
    encoded = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_tokens)
    input_ids = _ids(encoded["input_ids"])
    complete = _ids(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
    if len(input_ids) < 2:
        raise ValueError("Plaintext must yield at least two input tokens.")
    targets = input_ids[1:]
    eos = getattr(tokenizer, "eos_token_id", None)
    removed_eos = bool(targets and eos is not None and targets[-1] == eos)
    if removed_eos:
        targets = targets[:-1]
    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
    if any(value in special_ids for value in targets):
        raise ValueError("Special token found in plaintext targets after final EOS removal.")
    if not targets:
        raise ValueError("Plaintext has no targets after the causal shift and EOS handling.")
    return PreparedInput(str(identifier), "olmo_author_plaintext", input_ids, targets, text, {
        "tokenization": "tokenizer(add_special_tokens=True,truncation=True)",
        "source_text_preprocessing": "str.strip() before tokenization and Zlib, matching upstream extract_text",
        "raw_source_text_sha256": raw_text_sha256,
        "chat_template_applied": False,
        "max_tokens": max_tokens,
        "author_max_tokens": 4096,
        "context_variant": max_tokens != 4096,
        "source_token_count": len(complete),
        "source_token_ids_sha256": _digest_ids(complete),
        "truncated": len(complete) > len(input_ids),
        "final_eos_target_removed": removed_eos,
        "other_special_targets": "rejected",
    })


def prepare_checkmiabench_tokens(
    token_ids: Any, identifier: str, *, text: str | None = None,
    max_tokens: int | None = None,
) -> PreparedInput:
    """Use released token IDs directly; never decode and re-encode them.

    Forward the source except its final token; score every following token,
    including that final external target. This omits an unused final query,
    preserving all original causal targets without exceeding a 2048-input
    context for released 2049-ID sequences. Original IDs are retained in full.
    An explicit cap counts FORWARDED inputs and declares a prefix experiment.
    """
    original = _ids(token_ids)
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("An explicit forwarded-input cap must be at least one token.")
    selected = original if max_tokens is None else original[:max_tokens + 1]
    if len(selected) < 2:
        raise ValueError("A source sequence must contain at least two tokens.")
    if text is not None and not isinstance(text, str):
        raise ValueError("Optional source text must be a string.")
    return PreparedInput(str(identifier), "checkmiabench_original_tokens", selected[:-1],
                         selected[1:], text, {
        "tokenization": "none: original released token IDs",
        "decode_reencode": False,
        "max_tokens": max_tokens,
        "max_tokens_unit": "forwarded input tokens; retains up to max_tokens+1 source IDs",
        "context_variant": max_tokens is not None,
        "source_token_count": len(original),
        "source_token_ids_sha256": _digest_ids(original),
        "truncated": len(selected) < len(original),
        "selected_source_token_count": len(selected),
        "causal_target_layout": "input=selected_source[:-1];targets=selected_source[1:]",
        "unused_final_query_omitted": True,
        "final_target_external_to_forward_input": True,
        "final_target_original_position": len(selected) - 1,
        "original_token_ids_retained": True,
        "final_eos_target_removed": False,
        "other_special_targets": "retained as original training-sequence tokens",
    }, original_token_ids=original)
