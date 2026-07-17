from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..metrics import normalized_levenshtein, paired_bootstrap_ci, rouge_l_f1


GENERATION_CACHE_VERSION = "guided-generation-v2-exact-budget-lossless-boundary"


def _decode_units(tokenizer: Any, token_ids: Any) -> str:
    """Disable optional whitespace cleanup when testing exact reconstruction."""
    try:
        return str(tokenizer.decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        ))
    except TypeError:
        # Small injected tokenizers may not implement the optional HF keyword.
        return str(tokenizer.decode(token_ids, skip_special_tokens=True))


@dataclass(frozen=True)
class TextSplit:
    """Split text with suffix length in tokenizer units, including proxy units."""

    prefix: str
    suffix: str
    suffix_token_count: int


def split_text(tokenizer: Any, text: str, split_ratio: float = 0.5) -> TextSplit:
    """Split at tokenizer units (Unicode code points for a remote proxy)."""
    if not text.strip():
        raise ValueError("Text must contain at least one non-whitespace character.")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) < 2:
        raise ValueError("Text must produce at least two tokenizer units for guided scoring.")
    if _decode_units(tokenizer, token_ids) != text:
        raise ValueError("Text has no lossless tokenizer reconstruction for guided scoring.")
    boundary = min(max(1, int(len(token_ids) * split_ratio)), len(token_ids) - 1)
    # Byte-level BPE tokens may end inside a Unicode character. Pick the
    # nearest token boundary whose two decoded slices reconstruct the input.
    for distance in range(max(boundary, len(token_ids) - boundary)):
        candidates = (boundary,) if distance == 0 else (boundary - distance, boundary + distance)
        for candidate in candidates:
            if not 0 < candidate < len(token_ids):
                continue
            prefix = _decode_units(tokenizer, token_ids[:candidate])
            suffix = _decode_units(tokenizer, token_ids[candidate:])
            if prefix and suffix and prefix + suffix == text:
                return TextSplit(prefix, suffix, len(token_ids) - candidate)
    raise ValueError("Text has no lossless tokenizer boundary for guided scoring.")


def build_guided_prompt(
    prefix: str,
    dataset_name: str,
    split_name: str,
) -> str:
    return (
        f"You are given the first part of an example from the {dataset_name} dataset, "
        f"split {split_name}. Reproduce the exact second part as it appears in the "
        "dataset, with no extra commentary.\n\n"
        f"First part:\n{prefix}\n\nSecond part:\n"
    )


def build_general_prompt(prefix: str) -> str:
    return (
        "You are given the first part of an example from a dataset. Reproduce the exact "
        "second part as it appears in the dataset, with no extra commentary.\n\n"
        f"First part:\n{prefix}\n\nSecond part:\n"
    )


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def score_completion(
    reference: str,
    completion: str,
    *,
    near_threshold: float = 0.2,
) -> dict[str, float | bool]:
    normalized_reference = normalize_text(reference)
    normalized_completion = normalize_text(completion)
    distance = normalized_levenshtein(normalized_reference, normalized_completion)
    return {
        "rouge": rouge_l_f1(normalized_reference, normalized_completion),
        "exact": normalized_reference == normalized_completion,
        "near": distance <= near_threshold,
        "normalized_distance": distance,
    }


def _maybe_format(model: Any, prompt: str, *, chat_template: bool) -> str:
    if not chat_template:
        return prompt
    formatter = getattr(model, "format_prompt", None)
    if not callable(formatter):
        return prompt
    return str(formatter(prompt, chat_template=True))


def _truncate_to_units(tokenizer: Any, text: str, budget: int) -> str:
    """Use the longest lossless decoded prefix within a tokenizer-unit budget."""
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except (TypeError, ValueError, AttributeError):
        raise ValueError("Completion cannot be encoded for lossless truncation.") from None
    if len(ids) <= budget:
        return text
    for length in range(min(budget, len(ids)), 0, -1):
        prefix = _decode_units(tokenizer, ids[:length])
        if text.startswith(prefix):
            return prefix
    # The first character may itself require more tokens than the budget.
    return ""


def _generate_all(
    model: Any,
    prompts: list[str],
    budgets: list[int],
    *,
    batch_size: int,
    cache: Any | None,
    cache_model_fingerprint: str,
    cache_tokenizer_fingerprint: str,
) -> list[str]:
    """Generate one completion per prompt, batching and caching when possible."""
    if not prompts:
        return []
    from ..cache import cache_key

    keys: list[str | None] = []
    outputs: list[str | None] = [None] * len(prompts)
    tokenizer = getattr(model, "tokenizer", None)
    for index, (prompt, budget) in enumerate(zip(prompts, budgets, strict=True)):
        if cache is None or not cache_model_fingerprint:
            keys.append(None)
            continue
        key = cache_key(
            GENERATION_CACHE_VERSION,
            cache_model_fingerprint,
            cache_tokenizer_fingerprint,
            prompt,
            budget,
        )
        keys.append(key)
        try:
            cached = cache.get_generation(key)
        except Exception:
            cached = None
        if cached is not None:
            # Older cache rows may exceed the requested budget. Use the same
            # scoring convention for cached and freshly generated completions.
            outputs[index] = (
                _truncate_to_units(tokenizer, cached, budget)
                if tokenizer is not None
                else cached
            )
    generate_batch = getattr(model, "generate_batch", None)
    # Remote proxies do not know provider tokenization. A batch's maximum
    # budget must not change another example's request, and remote calls are
    # sequential already. Preserve each example's own API request budget.
    if getattr(tokenizer, "is_proxy", False):
        generate_batch = None
    missing = [index for index, value in enumerate(outputs) if value is None]
    if callable(generate_batch):
        # Local generation reserves context for max_new_tokens. Sharing the
        # largest budget across unlike examples would change their prompts,
        # even with greedy decoding, and make partial-cache runs disagree.
        by_budget: dict[int, list[int]] = {}
        for index in missing:
            by_budget.setdefault(budgets[index], []).append(index)
        groups = list(by_budget.values())
    else:
        groups = [missing]
    for group in groups:
        for start in range(0, len(group), max(1, batch_size)):
            chunk = group[start : start + max(1, batch_size)]
            chunk_prompts = [prompts[index] for index in chunk]
            chunk_budgets = [budgets[index] for index in chunk]
            if callable(generate_batch) and len(chunk) > 1:
                chunk_outputs = list(
                    generate_batch(
                        chunk_prompts,
                        chunk_budgets[0],
                        batch_size=len(chunk),
                    )
                )
            else:
                chunk_outputs = [
                    str(model.generate(prompt, budget))
                    for prompt, budget in zip(chunk_prompts, chunk_budgets, strict=True)
                ]
            for index, output, budget in zip(chunk, chunk_outputs, chunk_budgets, strict=True):
                output = str(output)
                if tokenizer is not None:
                    output = _truncate_to_units(tokenizer, output, budget)
                outputs[index] = output
                key = keys[index]
                if key is not None and cache is not None:
                    try:
                        cache.put_generation(key, output)
                    except Exception:
                        continue
    return [str(value) for value in outputs]


def _run_split(
    model: Any,
    examples: list[dict[str, str]],
    *,
    dataset_name: str,
    split_name: str,
    split_ratio: float,
    repro_threshold: float,
    seed: int,
    chat_template: bool,
    generation_batch_size: int,
    cache: Any | None,
    cache_model_fingerprint: str,
    cache_tokenizer_fingerprint: str,
    progress: Callable[[str, int, int], None] | None,
) -> dict[str, object]:
    scorable: list[tuple[dict[str, str], TextSplit]] = []
    skipped: list[dict[str, str]] = []
    for example in examples:
        try:
            text_split = split_text(model.tokenizer, example["text"], split_ratio)
        except ValueError as error:
            skipped.append({"id": example["id"], "reason": str(error)})
            continue
        scorable.append((example, text_split))
    if not scorable:
        raise ValueError("No examples could be scored by the guided method.")

    guided_prompts = [
        _maybe_format(
            model,
            build_guided_prompt(text_split.prefix, dataset_name, split_name),
            chat_template=chat_template,
        )
        for _example, text_split in scorable
    ]
    general_prompts = [
        _maybe_format(
            model,
            build_general_prompt(text_split.prefix),
            chat_template=chat_template,
        )
        for _example, text_split in scorable
    ]
    budgets = [text_split.suffix_token_count + 16 for _, text_split in scorable]
    generate_kwargs = {
        "batch_size": generation_batch_size,
        "cache": cache,
        "cache_model_fingerprint": cache_model_fingerprint,
        "cache_tokenizer_fingerprint": cache_tokenizer_fingerprint,
    }
    guided_outputs = _generate_all(model, guided_prompts, budgets, **generate_kwargs)
    general_outputs = _generate_all(model, general_prompts, budgets, **generate_kwargs)

    # Negative control: ask the model to reproduce each target suffix from a
    # *different* example's prefix (cyclic donor).  Reproduction under the
    # wrong prefix means the output is format or prior driven, not memory
    # of that example.  Skipped when fewer than two examples are scorable.
    mismatched_outputs: list[str | None] | None = None
    if len(scorable) >= 2:
        donor_prefixes = [
            scorable[(index + 1) % len(scorable)][1].prefix
            for index in range(len(scorable))
        ]
        mismatched_prompts = [
            _maybe_format(
                model,
                build_guided_prompt(donor_prefix, dataset_name, split_name),
                chat_template=chat_template,
            )
            for donor_prefix in donor_prefixes
        ]
        mismatched_outputs = _generate_all(
            model, mismatched_prompts, budgets, **generate_kwargs
        )

    per_example: list[dict[str, object]] = []
    for position, (example, text_split) in enumerate(scorable):
        guided_score = score_completion(text_split.suffix, guided_outputs[position])
        general_score = score_completion(text_split.suffix, general_outputs[position])
        mismatched_score = (
            score_completion(text_split.suffix, mismatched_outputs[position])
            if mismatched_outputs is not None
            else None
        )
        per_example.append(
            {
                "id": example["id"],
                "reference": text_split.suffix,
                "guided_output": guided_outputs[position],
                "general_output": general_outputs[position],
                "guided_rouge": guided_score["rouge"],
                "general_rouge": general_score["rouge"],
                "guided_exact": guided_score["exact"],
                "general_exact": general_score["exact"],
                "guided_near": guided_score["near"],
                "general_near": general_score["near"],
                "mismatched_rouge": (
                    mismatched_score["rouge"] if mismatched_score else None
                ),
                "mismatched_exact": (
                    mismatched_score["exact"] if mismatched_score else None
                ),
                "mismatched_near": (
                    mismatched_score["near"] if mismatched_score else None
                ),
            }
        )
        if progress is not None:
            progress("Guided overlap", len(per_example), len(scorable))

    guided_rouges = [float(item["guided_rouge"]) for item in per_example]
    general_rouges = [float(item["general_rouge"]) for item in per_example]
    guided_rouge = float(np.mean(guided_rouges))
    general_rouge = float(np.mean(general_rouges))
    count = len(per_example)

    def rate(key: str) -> float:
        return sum(bool(item[key]) for item in per_example) / count

    mismatched_rouges = [
        float(item["mismatched_rouge"])  # type: ignore[arg-type]
        for item in per_example
        if item["mismatched_rouge"] is not None
    ]
    mismatched_rouge = (
        float(np.mean(mismatched_rouges)) if mismatched_rouges else None
    )
    guided_mismatched_gap = (
        guided_rouge - mismatched_rouge if mismatched_rouge is not None else None
    )
    gap = guided_rouge - general_rouge
    return {
        "guided_rouge": guided_rouge,
        "general_rouge": general_rouge,
        "guided_general_gap": gap,
        "guided_gap_ci": paired_bootstrap_ci(
            guided_rouges, general_rouges, seed=seed
        ),
        "guided_exact_rate": rate("guided_exact"),
        "general_exact_rate": rate("general_exact"),
        "exact_gap": rate("guided_exact") - rate("general_exact"),
        "guided_near_rate": rate("guided_near"),
        "general_near_rate": rate("general_near"),
        "near_gap": rate("guided_near") - rate("general_near"),
        "mismatched_rouge": mismatched_rouge,
        "guided_mismatched_gap": guided_mismatched_gap,
        "mismatched_near_rate": (
            float(np.mean([bool(item["mismatched_near"]) for item in per_example]))
            if mismatched_outputs is not None
            else None
        ),
        "mismatched_exact_rate": (
            float(
                np.mean([bool(item["mismatched_exact"]) for item in per_example])
            )
            if mismatched_outputs is not None
            else None
        ),
        "reproduction_rate": sum(
            score >= repro_threshold for score in guided_rouges
        )
        / count,
        "index_guided": min(max(gap, 0.0), 1.0),
        "per_example": per_example,
        "skipped_examples": skipped,
        "split_ratio": split_ratio,
    }


def run_guided_overlap(
    model: Any,
    examples: list[dict[str, str]],
    *,
    dataset_name: str,
    split_name: str,
    split_ratio: float = 0.5,
    repro_threshold: float = 0.6,
    seed: int = 0,
    chat_template: bool = False,
    generation_batch_size: int = 4,
    cache: Any | None = None,
    cache_model_fingerprint: str = "",
    cache_tokenizer_fingerprint: str = "",
    stability: bool = False,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, object]:
    common = {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "repro_threshold": repro_threshold,
        "seed": seed,
        "chat_template": chat_template,
        "generation_batch_size": generation_batch_size,
        "cache": cache,
        "cache_model_fingerprint": cache_model_fingerprint,
        "cache_tokenizer_fingerprint": cache_tokenizer_fingerprint,
        "progress": progress,
    }
    result = _run_split(model, examples, split_ratio=split_ratio, **common)
    if stability:
        stability_gaps: dict[str, float] = {}
        for ratio in (0.25, 0.75):
            extra = _run_split(
                model, examples, split_ratio=ratio, **common
            )
            stability_gaps[str(ratio)] = float(extra["guided_general_gap"])  # type: ignore[arg-type]
        result["split_stability"] = stability_gaps
    else:
        result["split_stability"] = None
    return result
