"""Independently reconstruct a COMPLETED fixed CheckMIA numerical analysis.

No project metric/analysis module is imported and no language model is called.
Do not run against interim outcomes. Both scoring and analysis must be complete.
Point AUCs use literal pair comparisons; bootstrap AUCs use SciPy midranks on
regenerated paired indices, rather than the analyzer's weighted score buckets.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys
import time
import warnings
from typing import Any

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata


TOLERANCE = 1e-12
DRAWS = 3000
SEED = 20261107
MODEL_REPO = "EleutherAI/pythia-1.4b-deduped"
MODEL_REVISION = "29376c0a610279f1d79507fc887c6170082f8839"
DATASET_REPO = "jeffreygwang/pythia_dedupe_mia_0-97000_97000-98500"
DATASET_REVISION = "236fd818e7c877518d588eacc44658bdbf6b3a1b"
VARIANTS = ("author_native", "fp32_second_moment", "fp32_centered", "fixed_fp32_logp_bf16_reductions")
METRICS = ("loss", "mink", "minkpp", "zlib")
BLIND = "blind_char_tfidf.logistic"
CONTRASTS = (
    ("primary", "fp32_centered.minkpp", "fp32_centered.loss"),
    ("secondary", "fp32_centered.minkpp", "fp32_centered.mink"),
    ("secondary", "fp32_centered.minkpp", "pandora_uniform_fp32.minkxx"),
    ("secondary", "fp32_centered.minkpp", "author_native.minkpp"),
    ("secondary", "fp32_centered.minkpp", "fp32_second_moment.minkpp"),
    ("secondary", "pandora_uniform_fp32.minkxx", "pandora_uniform_native.minkxx"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def encoded(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def sha_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def close_number(actual: Any, expected: Any, label: str) -> None:
    if expected is None:
        require(actual is None, f"{label}: unavailable result was replaced by a number.")
        return
    require(actual is not None and isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual) and math.isfinite(expected), f"{label}: invalid/missing number.")
    require(abs(float(actual) - float(expected)) <= TOLERANCE,
            f"{label}: differs by {abs(float(actual) - float(expected))!r}.")


def close_array(actual: Any, expected: Any, label: str) -> float:
    left, right = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all(),
            f"{label}: shape/nonfinite mismatch.")
    maximum = float(np.max(np.abs(left - right))) if left.size else 0.0
    require(maximum <= TOLERANCE, f"{label}: max absolute difference {maximum!r}.")
    return maximum


def pair_wins_twice(positive: Any, negative: Any) -> int:
    """Integer twice-win count, including ties, without a project AUC helper."""
    positive, negative = np.asarray(positive, dtype=float), np.asarray(negative, dtype=float)
    require(np.isfinite(positive).all() and np.isfinite(negative).all(), "Nonfinite pair-count input.")
    total = 0
    for start in range(0, len(positive), 128):
        left = positive[start:start + 128, None]
        total += 2 * int(np.count_nonzero(left > negative[None, :]))
        total += int(np.count_nonzero(left == negative[None, :]))
    return total


def direct_auc(positive: Any, negative: Any) -> float:
    require(len(positive) > 0 and len(negative) > 0, "Point AUC needs both complete arms.")
    return pair_wins_twice(positive, negative) / (2 * len(positive) * len(negative))


def rank_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Mann-Whitney identity using average ranks, independently of bucket counts."""
    m, n = len(positive), len(negative)
    ranks = rankdata(np.concatenate((positive, negative)), method="average")
    return float((ranks[:m].sum() - m * (m + 1) / 2) / (m * n))


def linear_interval(values: Any) -> list[float]:
    """R7 2.5/97.5 percentiles from exact fractional order positions."""
    ordered = np.sort(np.asarray(values, dtype=float))
    require(len(ordered) > 0 and np.isfinite(ordered).all(), "Invalid interval draws.")
    result = []
    for numerator in (1, 39):
        left, remainder = divmod((len(ordered) - 1) * numerator, 40)
        right = min(left + 1, len(ordered) - 1)
        result.append(float(ordered[left] + (ordered[right] - ordered[left]) * remainder / 40))
    return result
