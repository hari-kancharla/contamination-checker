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


def bootstrap_ranks(vectors: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    random = np.random.default_rng(SEED)
    result = {name: np.empty(DRAWS) for name in vectors}
    require(all(len(a) == len(b) == 1000 for a, b in vectors.values()), "Bootstrap arms must remain 1,000/1,000.")
    for draw in range(DRAWS):
        # Exactly the same paired indices, but no bincount/grouped-score AUC.
        positive_indices = random.integers(0, 1000, size=1000)
        negative_indices = random.integers(0, 1000, size=1000)
        for name, (positive, negative) in vectors.items():
            result[name][draw] = rank_auc(positive[positive_indices], negative[negative_indices])
        if (draw + 1) % 500 == 0:
            print(json.dumps({"stage": "independent_rank_bootstrap", "draws_completed": draw + 1}), flush=True)
    return result


def blind_replay(source: list[dict[str, Any]], parameters: dict[str, Any], stored: dict[str, float],
                 *, refit: bool) -> dict[str, Any]:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    train = [row for row in source if row["study_split"] == "dev"]
    test = sorted((row for row in source if row["study_split"] == "test"), key=lambda row: row["id"])
    require(parameters["training_ids"] == [row["id"] for row in train], "Blind training IDs/order are not sealed dev only.")
    require(len(train) == 2000 and len(test) == 2000 and
            Counter(row["official_label"] for row in train) == {0: 1000, 1: 1000}, "Wrong blind training/test cohort.")
    require(not set(parameters["training_ids"]) & {row["id"] for row in test}, "Blind train/test overlap.")
    require(parameters["classes"] == [0, 1], "Blind score orientation/classes changed.")
    require(parameters["features"] == "char3-5 TFIDF20k sublinear_tf L2 lowercase; no metadata"
            and parameters["classification"] == "C1 liblinear balanced max_iter1000 seed20261107; dev only, no tuning",
            "Blind declared hyperparameters changed.")
    vocabulary = parameters["vocabulary"]
    require(0 < len(vocabulary) <= 20000 and sorted(vocabulary.values()) == list(range(len(vocabulary))),
            "Invalid blind feature mapping.")
    saved_idf = np.asarray(parameters["idf"], dtype=float)
    coef = np.asarray(parameters["coef"], dtype=float)
    intercept = np.asarray(parameters["intercept"], dtype=float)
    require(saved_idf.shape == (len(vocabulary),) and coef.shape == (1, len(vocabulary))
            and intercept.shape == (1,) and all(np.isfinite(x).all() for x in (saved_idf, coef, intercept)),
            "Invalid blind parameters.")
    settings = dict(analyzer="char", ngram_range=(3, 5), lowercase=True, min_df=1,
                    max_features=20000, sublinear_tf=True, norm="l2")
    train_texts, test_texts = [row["text"] for row in train], [row["text"] for row in test]
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        fixed = TfidfVectorizer(vocabulary=vocabulary, **settings)
        fixed.fit(train_texts)
        idf_difference = close_array(fixed.idf_, saved_idf, "Blind IDF reconstructed from dev text")
        fixed.idf_ = saved_idf
        prediction = expit(np.asarray(fixed.transform(test_texts) @ coef[0]).ravel() + intercept[0])
        expected = np.asarray([stored[row["id"]] for row in test])
        prediction_difference = close_array(prediction, expected, "Blind prediction replay from saved JSON parameters")
        result = {"training_records": 2000, "test_records": 2000, "training_ids_dev_only": True,
                  "saved_parameter_prediction_max_absolute_difference": prediction_difference,
                  "dev_reconstructed_idf_max_absolute_difference": idf_difference,
                  "independent_refit_requested": refit}
        if refit:
            fresh = TfidfVectorizer(**settings)
            matrix = fresh.fit_transform(train_texts)
            require(fresh.vocabulary_ == vocabulary, "Dev-only refit feature selection differs.")
            fresh_idf_difference = close_array(fresh.idf_, saved_idf, "Independent blind refit IDF")
            classifier = LogisticRegression(C=1., solver="liblinear", class_weight="balanced", max_iter=1000,
                                            random_state=SEED)
            classifier.fit(matrix, [row["official_label"] for row in train])
            predicted = classifier.predict_proba(fresh.transform(test_texts))[:, 1]
            result.update(independent_refit_prediction_max_absolute_difference=close_array(
                predicted, expected, "Independent dev-only blind refit predictions"),
                independent_refit_idf_max_absolute_difference=fresh_idf_difference,
                independent_refit_coefficient_max_absolute_difference=float(np.max(np.abs(classifier.coef_ - coef))))
    return result
