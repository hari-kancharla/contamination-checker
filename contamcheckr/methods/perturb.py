"""Perturbation-sensitivity membership signal.

Hypothesis: memorized sequences are *brittle* — likelihood drops more
when the exact surface form is mutated than it does for unseen text.
For each text ``x`` and each surface-preserving mutation ``m`` (sentence
reorder, identifier shift, whitespace squeeze), the statistic is

    delta(x) = mean_logp(x) - mean_logp(m(x)).

Unseen text is hurt about equally by mutation (delta near zero);
memorized exact text collapses (delta positive and large).  This is
orthogonal to overlap counts (n-gram) and to first-order likelihood
(Loss/Min-K%++): it measures *sensitivity*, not level.

Deterministic pure-Python mutations, no external models.  Report-only:
the mean delta across transforms is calibrated like other scores
(AUC vs control) and recorded, never fused into the index.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Sequence

import numpy as np

from .token_stats import TokenStats, token_stats_from_logits


def mutate_sentence_shuffle(text: str) -> str:
    """Reverse sentence order (single-sentence texts unchanged)."""
    if not text.strip():
        return text
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(parts) < 2:
        return text
    return " ".join(reversed(parts))


def mutate_number_shift(text: str) -> str:
    """Shift every digit run by +7 (deterministic; catches ID brittleness)."""
    def _shift(match: re.Match[str]) -> str:
        run = match.group(0)
        # Preserve width when possible (e.g. 100 -> 107, not truncated).
        return str(int(run) + 7)

    return re.sub(r"\d+", _shift, text)


def mutate_whitespace(text: str) -> str:
    """Collapse all whitespace runs to single spaces."""
    return " ".join(text.split())


TRANSFORMS: dict[str, Callable[[str], str]] = {
    "sentence_shuffle": mutate_sentence_shuffle,
    "number_shift": mutate_number_shift,
    "whitespace": mutate_whitespace,
}


def _mean_logprob_from_stats(stats: TokenStats) -> float:
    if not stats.target_logprobs:
        raise ValueError("Token statistics are empty.")
    return float(np.mean(stats.target_logprobs))


def mean_logprob(model: Any, text: str, *, max_tokens: int = 512) -> float:
    """Mean per-token log-probability (LOSS), batch-aware entry point."""
    token_stats = getattr(model, "token_stats", None)
    if callable(token_stats):
        return _mean_logprob_from_stats(token_stats(text, max_tokens=max_tokens))
    logits, targets = model.token_logprob_table(text, max_tokens=max_tokens)
    return _mean_logprob_from_stats(token_stats_from_logits(logits, targets))


def _mean_logprobs_batch(model: Any, texts: Sequence[str], *, max_tokens: int) -> list[float]:
    batch_stats = getattr(model, "token_stats_batch", None)
    if callable(batch_stats):
        return [
            _mean_logprob_from_stats(stats)
            for stats in batch_stats(list(texts), max_tokens=max_tokens)
        ]
    return [mean_logprob(model, text, max_tokens=max_tokens) for text in texts]


def perturb_delta_scores(
    model: Any,
    texts: Sequence[str],
    *,
    transforms: Sequence[str] = ("number_shift", "sentence_shuffle"),
    max_tokens: int = 512,
) -> dict[str, list[float]]:
    """Per-transform deltas: mean_logp(original) - mean_logp(mutated).

    Returns ``{transform_name: [delta per text]}`` in input order.
    """
    if not transforms:
        raise ValueError("At least one transform is required.")
    unknown = [name for name in transforms if name not in TRANSFORMS]
    if unknown:
        raise ValueError(f"Unknown perturb transforms: {', '.join(unknown)}.")
    base = _mean_logprobs_batch(model, list(texts), max_tokens=max_tokens)
    result: dict[str, list[float]] = {}
    for name in transforms:
        mutate = TRANSFORMS[name]
        mutated = [mutate(text) for text in texts]
        mutated_means = _mean_logprobs_batch(model, mutated, max_tokens=max_tokens)
        result[name] = [
            origin - mutated_value
            for origin, mutated_value in zip(base, mutated_means, strict=True)
        ]
    return result


def perturb_delta_mean(
    model: Any,
    texts: Sequence[str],
    *,
    transforms: Sequence[str] = ("number_shift", "sentence_shuffle"),
    max_tokens: int = 512,
) -> list[float]:
    """Mean delta across transforms, one value per text."""
    grouped = perturb_delta_scores(
        model, texts, transforms=transforms, max_tokens=max_tokens
    )
    if not texts:
        return []
    matrix = np.asarray(list(grouped.values()), dtype=float)
    return [float(value) for value in np.mean(matrix, axis=0)]
