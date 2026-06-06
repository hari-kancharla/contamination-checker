from dataclasses import dataclass
from typing import Any

import numpy as np

from ..metrics import normalized_levenshtein, paired_bootstrap_ci, rouge_l_f1


@dataclass(frozen=True)
class TextSplit:
    prefix: str
    suffix: str
    suffix_token_count: int


def split_text(tokenizer: Any, text: str, split_ratio: float = 0.5) -> TextSplit:
    if not text.strip():
        raise ValueError("Text must contain at least one non-whitespace character.")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) < 2:
        raise ValueError("Text must produce at least two tokens for guided scoring.")
    boundary = min(max(1, int(len(token_ids) * split_ratio)), len(token_ids) - 1)
    return TextSplit(
        prefix=tokenizer.decode(token_ids[:boundary], skip_special_tokens=True),
        suffix=tokenizer.decode(token_ids[boundary:], skip_special_tokens=True),
        suffix_token_count=len(token_ids) - boundary,
    )


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


def run_guided_overlap(
    model: object,
    examples: list[dict[str, str]],
    *,
    dataset_name: str,
    split_name: str,
    split_ratio: float = 0.5,
    repro_threshold: float = 0.6,
    seed: int = 0,
) -> dict[str, object]:
    per_example: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    for example in examples:
        try:
            text_split = split_text(model.tokenizer, example["text"], split_ratio)
        except ValueError as error:
            skipped.append({"id": example["id"], "reason": str(error)})
            continue
        guided = model.generate(
            build_guided_prompt(text_split.prefix, dataset_name, split_name),
            text_split.suffix_token_count + 16,
        )
        general = model.generate(
            build_general_prompt(text_split.prefix),
            text_split.suffix_token_count + 16,
        )
        guided_score = score_completion(text_split.suffix, guided)
        general_score = score_completion(text_split.suffix, general)
        per_example.append(
            {
                "id": example["id"],
                "reference": text_split.suffix,
                "guided_output": guided,
                "general_output": general,
                "guided_rouge": guided_score["rouge"],
                "general_rouge": general_score["rouge"],
                "guided_exact": guided_score["exact"],
                "general_exact": general_score["exact"],
                "guided_near": guided_score["near"],
                "general_near": general_score["near"],
            }
        )
    if not per_example:
        raise ValueError("No examples could be scored by the guided method.")
    guided_rouges = [float(item["guided_rouge"]) for item in per_example]
    general_rouges = [float(item["general_rouge"]) for item in per_example]
    guided_rouge = float(np.mean(guided_rouges))
    general_rouge = float(np.mean(general_rouges))
    count = len(per_example)

    def rate(key: str) -> float:
        return sum(bool(item[key]) for item in per_example) / count

    guided_exact = rate("guided_exact")
    general_exact = rate("general_exact")
    guided_near = rate("guided_near")
    general_near = rate("general_near")
    gap = guided_rouge - general_rouge
    return {
        "guided_rouge": guided_rouge,
        "general_rouge": general_rouge,
        "guided_general_gap": gap,
        "guided_gap_ci": paired_bootstrap_ci(
            guided_rouges, general_rouges, seed=seed
        ),
        "guided_exact_rate": guided_exact,
        "general_exact_rate": general_exact,
        "exact_gap": guided_exact - general_exact,
        "guided_near_rate": guided_near,
        "general_near_rate": general_near,
        "near_gap": guided_near - general_near,
        "reproduction_rate": sum(
            score >= repro_threshold for score in guided_rouges
        )
        / count,
        "index_guided": min(max(gap, 0.0), 1.0),
        "per_example": per_example,
        "skipped_examples": skipped,
    }
