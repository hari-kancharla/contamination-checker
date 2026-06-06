from collections.abc import Callable, Sequence
import math

import numpy as np


def _mean_difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(left - right))


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    if lower > upper:
        raise ValueError("Lower bound cannot exceed upper bound.")
    return float(min(max(value, lower), upper))


def lcs_length(left: Sequence[object], right: Sequence[object]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, start=1):
            if left_item == right_item:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(reference: str, candidate: str) -> float:
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    if not reference_tokens and not candidate_tokens:
        return 1.0
    if not reference_tokens or not candidate_tokens:
        return 0.0
    common = lcs_length(reference_tokens, candidate_tokens)
    precision = common / len(candidate_tokens)
    recall = common / len(reference_tokens)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def normalized_levenshtein(left: str, right: str) -> float:
    denominator = max(len(left), len(right))
    if denominator == 0:
        return 0.0
    return levenshtein(left, right) / denominator


def auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    if len(labels) != len(scores) or not labels:
        raise ValueError("AUC requires equally sized non-empty labels and scores.")
    positives = sum(label == 1 for label in labels)
    negatives = sum(label == 0 for label in labels)
    if positives == 0 or negatives == 0:
        raise ValueError("AUC requires both positive and negative labels.")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("AUC labels must be 0 or 1.")

    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    positive_rank_sum = sum(
        rank for rank, label in zip(ranks, labels, strict=True) if label == 1
    )
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def lowest_percent(values: Sequence[float], percent: float) -> list[float]:
    if not values:
        raise ValueError("Percentile selection requires at least one value.")
    if not 0.0 < percent <= 100.0:
        raise ValueError("Percent must be greater than 0 and at most 100.")
    count = max(1, math.ceil(len(values) * percent / 100.0))
    return sorted(float(value) for value in values)[:count]


def paired_bootstrap_ci(
    left: Sequence[float],
    right: Sequence[float],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    statistic: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> tuple[float, float] | None:
    if len(left) != len(right):
        raise ValueError("Paired bootstrap inputs must have equal length.")
    if len(left) < 2:
        return None
    if samples < 1:
        raise ValueError("Bootstrap sample count must be at least 1.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must be between 0 and 1.")
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if not np.isfinite(left_values).all() or not np.isfinite(right_values).all():
        raise ValueError("Bootstrap inputs must be finite.")
    if statistic is None:
        statistic = _mean_difference
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = rng.integers(0, len(left_values), len(left_values))
        estimates[index] = statistic(left_values[selected], right_values[selected])
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(low), float(high)
