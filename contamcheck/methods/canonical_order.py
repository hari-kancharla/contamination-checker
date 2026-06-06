from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..metrics import clamp


@dataclass(frozen=True)
class OrderTestResult:
    canonical_advantage: float
    p_value: float
    index_order: float
    shards: list[dict[str, object]]
    permutation_advantages: list[float]


def _shard_examples(
    examples: list[dict[str, str]], shard_count: int
) -> list[list[dict[str, str]]]:
    return [
        [examples[index] for index in positions]
        for positions in np.array_split(np.arange(len(examples)), shard_count)
        if len(positions) > 0
    ]


def canonical_order_test(
    examples: list[dict[str, str]],
    scorer: Callable[[str], float],
    *,
    shard_count: int = 10,
    permutation_count: int = 25,
    seed: int = 0,
    separator: str = "\n\n---\n\n",
) -> OrderTestResult:
    if len(examples) < 4:
        raise ValueError(
            "Canonical-order testing requires at least four examples."
        )
    effective_shards = min(shard_count, len(examples) // 2)
    if effective_shards < 2:
        raise ValueError(
            "Dataset is too small to form at least two meaningful shards."
        )
    shards = _shard_examples(examples, effective_shards)
    rng = np.random.default_rng(seed)
    shard_results: list[dict[str, object]] = []
    permutation_matrix = np.empty((len(shards), permutation_count), dtype=float)
    canonical_scores = np.empty(len(shards), dtype=float)
    for shard_index, shard in enumerate(shards):
        texts = [item["text"] for item in shard]
        canonical = float(scorer(separator.join(texts)))
        if not np.isfinite(canonical):
            raise ValueError(
                f"Canonical sequence likelihood was not finite for shard {shard_index}."
            )
        canonical_scores[shard_index] = canonical
        permutations: list[float] = []
        while len(permutations) < permutation_count:
            order = tuple(int(value) for value in rng.permutation(len(texts)))
            if order == tuple(range(len(texts))):
                continue
            score = float(scorer(separator.join(texts[index] for index in order)))
            if not np.isfinite(score):
                raise ValueError(
                    f"Permuted sequence likelihood was not finite for shard {shard_index}."
                )
            permutations.append(score)
        permutation_matrix[shard_index] = permutations
        shard_results.append(
            {
                "shard": shard_index,
                "example_ids": [item["id"] for item in shard],
                "canonical_ll": canonical,
                "permuted_mean_ll": float(np.mean(permutations)),
                "advantage": canonical - float(np.mean(permutations)),
            }
        )
    observed = float(np.mean(canonical_scores - np.mean(permutation_matrix, axis=1)))
    permutation_means = np.mean(permutation_matrix, axis=0)
    null_center = float(np.mean(permutation_means))
    null_advantages = [
        float(value - null_center) for value in permutation_means
    ]
    p_value = (1 + sum(value >= observed for value in null_advantages)) / (
        permutation_count + 1
    )
    return OrderTestResult(
        canonical_advantage=observed,
        p_value=float(p_value),
        index_order=clamp(1.0 - p_value),
        shards=shard_results,
        permutation_advantages=null_advantages,
    )
