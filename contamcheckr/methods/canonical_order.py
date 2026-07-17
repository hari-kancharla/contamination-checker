"""Canonical-order contamination test.

This implements the sharded likelihood comparison test of Oren et al.,
"Proving Test Set Contamination in Black-Box Language Models" (ICLR 2024),
Algorithm 1:

1. Partition the benchmark, in published order, into ``r`` contiguous shards.
2. For each shard, compare the log-likelihood of the canonical order with the
   mean log-likelihood of ``m`` random within-shard permutations.
3. Test whether the shard advantages have a positive mean with a one-sided
   t-test.

The implementation also reports a distribution-free rank test per shard and
combines those p-values as a sensitivity check.  The t-test is the published
test; the rank combination is a diagnostic, not a replacement.

The combined contamination index does not use ``1 - p`` from a permutation
rank.  Oren et al. show that a rank test with ``m`` draws cannot return a
p-value below ``1 / (m + 1)``, so a canonical ordering that wins every draw
by numerical noise still looks decisive.  The index includes the order test
only when the shard count can support the asymptotic t-test, the mean
advantage is larger than accumulation noise, and the one-sided p-value is
below 0.05.  Otherwise the p-value is still reported and the weight is zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Callable, Sequence

import numpy as np

from ..metrics import clamp
from ..stats import fisher_combination, mean_ci, one_sided_t_test, sign_test_p


@dataclass(frozen=True)
class OrderTestResult:
    canonical_advantage: float
    p_value: float
    p_value_rank: float
    index_order: float | None
    aggregate_reason: str | None
    effect_size: float
    effect_size_ci: tuple[float, float] | None
    shard_win_rate: float
    shards: list[dict[str, object]]
    warnings: list[str]
    shard_count: int
    permutation_count: int


# Oren et al. found the sharded t-test unreliable below about 10 shards.
MIN_SHARDS_FOR_AGGREGATE = 10
AGGREGATE_ALPHA = 0.05
# Summed log-likelihoods near 1e2–1e3 have float32 resolution around 1e-4.
# Advantages below 0.001 nats are not a contamination effect.
MIN_ADVANTAGE_NATS = 1e-3


def order_aggregate_index(
    advantages: Sequence[float],
) -> tuple[float, float | None, str | None]:
    """Return the t-test p-value, the index used in the combined score, and why it was withheld.

    The p-value is always the one-sided t-test of Oren et al. Algorithm 1.
    The index is ``1 - p`` only after the reporting gates pass.
    """
    tested = one_sided_t_test(advantages)
    p_value = float(tested["p_value"])
    advantage = float(np.mean(np.asarray(advantages, dtype=float)))
    reasons: list[str] = []
    if len(advantages) < MIN_SHARDS_FOR_AGGREGATE:
        reasons.append(
            f"Canonical order used {len(advantages)} shards. Oren et al. found "
            "the sharded t-test unreliable below about 10 shards, so this "
            "p-value is reported and left out of the combined index."
        )
    if abs(advantage) < MIN_ADVANTAGE_NATS:
        reasons.append(
            f"Mean canonical-order advantage is {advantage:.6g} nats, below "
            f"{MIN_ADVANTAGE_NATS:g}. Differences this small are consistent "
            "with accumulation error and were left out of the combined index."
        )
    if advantage <= 0.0 or p_value >= AGGREGATE_ALPHA:
        reasons.append(
            "The one-sided t-test did not meet the 0.05 reporting threshold, "
            "so canonical order was left out of the combined index."
        )
    if reasons:
        return p_value, None, " ".join(reasons)
    return p_value, float(clamp(1.0 - p_value)), None


def _shard_examples(
    examples: list[dict[str, str]], shard_count: int
) -> list[list[dict[str, str]]]:
    """Split examples into contiguous, non-empty shards of near-equal size."""
    if shard_count < 1:
        raise ValueError("Shard count must be at least one.")
    positions = np.array_split(np.arange(len(examples)), shard_count)
    return [
        [examples[int(index)] for index in shard]
        for shard in positions
        if len(shard) > 0
    ]


def _permutations(
    size: int,
    count: int,
    rng: np.random.Generator,
) -> list[tuple[int, ...]]:
    """Return ``count`` distinct non-identity permutations of ``range(size)``."""
    if size < 2:
        raise ValueError("A shard needs at least two examples.")
    identity = tuple(range(size))
    total = math.factorial(size)
    available = total - 1
    target = min(count, available)
    if size <= 8 and total <= 50_000:
        candidates = [
            permutation
            for permutation in itertools.permutations(range(size))
            if permutation != identity
        ]
        if target >= len(candidates):
            return candidates
        selected = rng.choice(len(candidates), size=target, replace=False)
        return [candidates[int(index)] for index in selected]
    seen: set[tuple[int, ...]] = set()
    result: list[tuple[int, ...]] = []
    while len(result) < target:
        permutation = tuple(int(value) for value in rng.permutation(size))
        if permutation == identity or permutation in seen:
            continue
        seen.add(permutation)
        result.append(permutation)
    return result


def _score_many(
    scorer: Callable[..., object],
    texts: Sequence[str],
) -> list[float]:
    """Call a batch scorer when possible, otherwise a single-text scorer."""
    if not texts:
        return []
    try:
        values = scorer(list(texts))
    except (TypeError, AttributeError, ValueError):
        values = None
    if values is not None:
        try:
            converted = [float(value) for value in values]  # type: ignore[union-attr]
        except (TypeError, ValueError):
            converted = []
        if len(converted) == len(texts):
            return converted
    return [float(scorer(text)) for text in texts]  # type: ignore[arg-type]


def _exchangeability_warnings(examples: list[dict[str, str]]) -> list[str]:
    warnings: list[str] = []
    texts = [example["text"] for example in examples]
    normalized = [" ".join(text.casefold().split()) for text in texts]
    duplicate_texts = len(normalized) - len(set(normalized))
    if duplicate_texts:
        warnings.append(
            f"{duplicate_texts} duplicate example texts break the exchangeability "
            "assumption; deduplicate the benchmark before interpreting the test."
        )
    identifiers = [str(example["id"]) for example in examples]
    if len(identifiers) != len(set(identifiers)):
        warnings.append("Example identifiers are not unique.")
    numeric_ids: list[int] = []
    for identifier in identifiers:
        try:
            numeric_ids.append(int(identifier))
        except ValueError:
            numeric_ids = []
            break
    if len(numeric_ids) > 2:
        differences = np.diff(numeric_ids)
        if np.all(differences == 1):
            warnings.append(
                "Identifiers are a contiguous integer sequence; if the published "
                "order carries index information, the exchangeability null is suspect."
            )
    return warnings


def canonical_order_test(
    examples: list[dict[str, str]],
    scorer: Callable[..., object],
    *,
    shard_count: int = 10,
    permutation_count: int = 25,
    seed: int = 0,
    separator: str = "\n\n---\n\n",
    token_counter: Callable[[str], int] | None = None,
    max_tokens: int | None = None,
) -> OrderTestResult:
    if len(examples) < 4:
        raise ValueError("Canonical-order testing requires at least four examples.")
    if permutation_count < 1:
        raise ValueError("Permutation count must be at least one.")
    effective_shards = min(shard_count, len(examples) // 2)
    if effective_shards < 2:
        raise ValueError(
            "Dataset is too small to form at least two meaningful shards."
        )
    shards = _shard_examples(examples, effective_shards)
    rng = np.random.default_rng(seed)

    canonical_texts: list[str] = []
    permutation_texts: list[list[str]] = []
    shard_metadata: list[dict[str, object]] = []
    for shard_index, shard in enumerate(shards):
        texts = [item["text"] for item in shard]
        canonical_texts.append(separator.join(texts))
        permutations = _permutations(len(texts), permutation_count, rng)
        permutation_texts.append(
            [separator.join(texts[index] for index in order) for order in permutations]
        )
        shard_metadata.append(
            {
                "shard": shard_index,
                "example_ids": [item["id"] for item in shard],
                "example_count": len(shard),
                "permutation_count": len(permutations),
            }
        )

    flat_texts = canonical_texts + [
        text for group in permutation_texts for text in group
    ]
    scores = _score_many(scorer, flat_texts)
    if len(scores) != len(flat_texts):
        raise ValueError("The sequence scorer returned the wrong number of scores.")
    if not all(math.isfinite(value) for value in scores):
        raise ValueError("Canonical-order sequence likelihoods must be finite.")

    cursor = len(canonical_texts)
    canonical_scores: list[float] = list(scores[:cursor])
    shard_advantages: list[float] = []
    shard_rank_pvalues: list[float] = []
    warnings = _exchangeability_warnings(examples)
    for shard_index, metadata in enumerate(shard_metadata):
        canonical = canonical_scores[shard_index]
        count = int(metadata["permutation_count"])
        permutations = scores[cursor : cursor + count]
        cursor += count
        mean_permuted = float(np.mean(permutations))
        advantage = canonical - mean_permuted
        rank_p = (1.0 + sum(value >= canonical for value in permutations)) / (
            count + 1.0
        )
        shard_advantages.append(advantage)
        shard_rank_pvalues.append(rank_p)
        metadata.update(
            {
                "canonical_ll": canonical,
                "permuted_mean_ll": mean_permuted,
                "permuted_std_ll": float(np.std(permutations, ddof=1))
                if count > 1
                else 0.0,
                "permuted_min_ll": float(np.min(permutations)),
                "permuted_max_ll": float(np.max(permutations)),
                "advantage": advantage,
                "rank_p_value": rank_p,
                "canonical_win_fraction": float(
                    np.mean([canonical > value for value in permutations])
                ),
            }
        )
        if token_counter is not None and max_tokens is not None:
            token_count = int(token_counter(canonical_texts[shard_index]))
            metadata["canonical_token_count"] = token_count
            if token_count > max_tokens:
                warnings.append(
                    f"Shard {shard_index} canonical text has {token_count} tokens, "
                    f"above max_tokens={max_tokens}; truncation can bias the test."
                )

    if cursor != len(scores):
        raise ValueError(
            "Canonical-order scores could not be aligned with shards and permutations."
        )

    advantages = np.asarray(shard_advantages, dtype=float)
    p_value, index_order, aggregate_reason = order_aggregate_index(advantages)
    rank_p = fisher_combination(shard_rank_pvalues)
    effect_size = (
        float(np.mean(advantages) / np.std(advantages, ddof=1))
        if len(advantages) > 1 and np.std(advantages, ddof=1) > 0.0
        else 0.0
    )
    interval = mean_ci(advantages)
    win_rate = float(np.mean(advantages > 0.0))
    if len(advantages) < 3:
        warnings.append(
            "Fewer than three shards makes the asymptotic t-test fragile."
        )
    if len(advantages) > 1 and float(np.std(advantages, ddof=1)) > 0.0 and effect_size < 0.2:
        warnings.append(
            "The shard-level effect size is small; the test may be underpowered."
        )
    return OrderTestResult(
        canonical_advantage=float(np.mean(advantages)),
        p_value=p_value,
        p_value_rank=float(rank_p),
        index_order=index_order,
        aggregate_reason=aggregate_reason,
        effect_size=float(effect_size),
        effect_size_ci=interval,
        shard_win_rate=win_rate,
        shards=shard_metadata,
        warnings=warnings,
        shard_count=len(shards),
        permutation_count=permutation_count,
    )
