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


def verify(sealed: Path, scoring: Path, analysis: Path, output: Path, *, refit_blind: bool) -> dict[str, Any]:
    # Gate before opening any scalar outcome artifact.
    execution = read_json(scoring / "execution.json")
    summary = read_json(analysis / "summary.json")
    require(execution["status"] == "complete" and execution["scored_records"] == 3000
            and execution["planned_scored_records"] == 3000 and summary["status"] == "complete",
            "Numerical verification requires completed scoring AND frozen analysis; no interim outcomes allowed.")
    require(execution["model_repo"] == summary["model_repo"] == MODEL_REPO
            and execution["model_revision"] == summary["model_revision"] == MODEL_REVISION,
            "Wrong checkpoint identity.")
    require(execution["amendment_id"] == summary["amendment_id"] == "checkmia-weighted-vs-uniform-v1-amendment-1"
            and execution["protocol_id"] == summary["protocol_id"] == "checkmia-weighted-vs-uniform-v1", "Wrong amendment/protocol.")
    initial_hashes = {"scoring_execution": sha_file(scoring / "execution.json"),
                      "analysis_summary": sha_file(analysis / "summary.json"), "data_seal": sha_file(sealed / "seal.json")}
    require(sha_file(scoring / "execution.json") == summary["source_execution_sha256"], "Analysis/execution identity differs.")
    require(sha_file(sealed / "seal.json") == execution["seal_sha256"] == summary["sealed_data_sha256"], "Seal differs.")
    seal = read_json(sealed / "seal.json")
    require(seal["dataset_revision"] == DATASET_REVISION and seal["dataset"] == DATASET_REPO,
            "Wrong released dataset/revision.")
    require(set(seal["files_sha256"]) == {"checkmia_data_audit.py", "protocol.json", "download_receipts.json",
            "record_index.jsonl", "assignment.jsonl", "quarantine.json", "inputs.jsonl.gz"}, "Unexpected sealed file set.")
    for name, expected in seal["files_sha256"].items():
        require(Path(name).name == name and sha_file(sealed / name) == expected, f"Sealed file changed: {name}.")
    require(set(summary["artifact_sha256"]) == {"checkmia_real_analysis.py", "blind_parameters.json", "paired_bootstrap_aucs.npz",
            "methods.json", "contrasts.json", "tail_records.jsonl", "test_score_vectors.jsonl"}, "Unexpected analysis artifact set.")
    for name, expected in summary["artifact_sha256"].items():
        require(Path(name).name == name and sha_file(analysis / name) == expected, f"Analysis artifact changed: {name}.")
    require(sha_file(analysis / "checkmia_real_analysis.py") == execution["source_sha256"]["checkmia_real_analysis.py"],
            "Analysis did not use the source frozen before model scoring.")
    require(set(execution["source_sha256"]) == {"checkmia_real_scoring.py", "real_score_engine.py", "checkmia_real_analysis.py"},
            "Unexpected frozen source set.")
    for name, expected in execution["source_sha256"].items():
        require(Path(name).name == name and sha_file(scoring / name) == expected, f"Frozen scoring source changed: {name}.")
    require(sha_file(scoring / "scores.jsonl") == execution["scores_sha256"], "Scalar artifact changed.")
    require(sha_file(scoring / "protocol.json") == execution["protocol_sha256"]
            and sha_file(scoring / "amendment.json") == execution["amendment_sha256"] == summary["amendment_sha256"],
            "Protocol/amendment changed.")
    protocol = read_json(scoring / "protocol.json")
    require(protocol["scoring"]["higher_member_orientation"] is True and protocol["scoring"]["k_percent"] == 10,
            "Wrong score direction or declared k.")
    require(protocol["source"]["dataset"] == DATASET_REPO and protocol["source"]["dataset_revision"] == DATASET_REVISION
            and protocol["source"]["target"] == MODEL_REPO and protocol["source"]["target_revision"] == MODEL_REVISION,
            "Protocol source/checkpoint differs.")
    receipts = read_json(sealed / "download_receipts.json")
    matching = [r for r in receipts if r["repo_type"] == "model" and r["repo_id"] == MODEL_REPO
                and r["revision"] == MODEL_REVISION]
    require(len(matching) == 1 and execution["model_files_verified"] == matching[0]["files"],
            "Executed model receipt differs from the sealed pinned receipt.")
    source = read_lines(sealed / "inputs.jsonl.gz")
    require(len(source) == 4000 and len({r["id"] for r in source}) == 4000, "Wrong/duplicate sealed cohort.")
    require(len({r["component_id"] for r in source}) == 4000,
            "Duplicate component identity within or across sealed development/test arms.")
    require(Counter((r["study_split"], r["arm"]) for r in source) ==
            {("dev", "member"): 1000, ("dev", "nonmember"): 1000,
             ("test", "member"): 1000, ("test", "nonmember"): 1000}, "Wrong sealed per-arm split counts.")
    for row in source:
        require(type(row["official_label"]) is int and row["official_label"] == int(row["arm"] == "member")
                and row["id"] == f"{row['arm']}:{row['source_index']}", "Source ID/official-label orientation differs.")
    expected = [row for row in source if row["study_split"] == "test" or row["official_label"] == 0]
    records = read_lines(scoring / "scores.jsonl")
    require([r["id"] for r in records] == [r["id"] for r in expected] and len(records) == 3000,
            "Scoring must retain exactly the full scheduled 3,000-record cohort and order.")
    by_id = {r["id"]: r for r in records}
    optional = []
    for row, original in zip(records, expected):
        require(row["source"] == {k: v for k, v in original.items() if k not in {"tokens", "text"}},
                f"Scored/source identity differs: {row['id']}.")
        require(row["runner_eligibility"]["primary_complete"] is True and row["target_count"] == 2048,
                "Selected/incomplete primary cohort.")
        for variant in ("fp32_second_moment", "fp32_centered"):
            part = row["variants"][variant]
            require(part["status"] == "complete" and part["supported_target_count"] == 2048
                    and part["aggregate_valid"] is True, "Required FP32 support differs.")
        require(row["pandora_comparators"]["fp32"]["status"] == "complete"
                and row["pandora_comparators"]["fp32"]["supported_target_count"] == 2048,
                "Required uniform FP32 support differs.")
        failed = sorted([f"variants/{k}" for k, v in row["variants"].items() if v["status"] != "complete"
                         and k not in ("fp32_second_moment", "fp32_centered")] +
                        [f"pandora_comparators/{k}" for k, v in row["pandora_comparators"].items()
                         if v["status"] != "complete" and k != "fp32"])
        require(sorted(row["runner_eligibility"]["optional_failed_sections"]) == failed, "Optional failures differ.")
        if failed:
            optional.append({"id": row["id"], "split": row["source"]["study_split"],
                             "official_label": row["source"]["official_label"], "failed_sections": failed})
    normalized = lambda rows: [{**r, "failed_sections": sorted(r["failed_sections"])} for r in rows]
    require(optional == normalized(execution["optional_failure_records"])
            and len(optional) == execution["optional_failure_record_count"], "Execution optional-failure accounting differs.")
    require(summary["scored_records"] == 3000 and summary["scored_targets"] == 3000 * 2048
            and summary["primary_failure_records"] == 0 and summary["all_attempted_records_accounted"] is True,
            "Analysis cohort totals differ.")
    require(summary["split_counts"] == dict(Counter(f"{r['source']['study_split']}/{r['source']['arm']}" for r in records)),
            "Analysis split denominators differ.")
    require(summary["optional_failure_records"] == len(optional)
            and normalized(summary["optional_failure_record_ids"]) == optional, "Summary optional-failure records differ.")
    support = {}
    for row in records:
        for name, part in row["variants"].items():
            entry = support.setdefault(name, {"records": 0, "clamped_targets": 0})
            entry["records"] += 1
            entry["clamped_targets"] += part["clamped_target_count"]
    require(summary["numerical_support"] == support, "Numerical-support presence/clamp totals differ.")
    test = sorted((r for r in source if r["study_split"] == "test"), key=lambda r: r["id"])
    positive_ids = [r["id"] for r in test if r["official_label"] == 1]
    negative_ids = [r["id"] for r in test if r["official_label"] == 0]
    dev_ids = sorted(r["id"] for r in source if r["study_split"] == "dev" and r["official_label"] == 0)
    require(len(positive_ids) == len(negative_ids) == len(dev_ids) == 1000, "Wrong metric/calibration arms.")
    values = {}
    for variant in VARIANTS:
        for metric in METRICS:
            values[f"{variant}.{metric}"] = {identifier: r["variants"][variant]["scores"][metric]
                                               for identifier, r in by_id.items()}
    for variant in ("native", "fp32"):
        name = f"pandora_uniform_{variant}.minkxx"
        values[name] = {identifier: r["pandora_comparators"][variant]["scores"]["pandora_minkxx"]
                        for identifier, r in by_id.items()}
        for row in records:
            scores = row["pandora_comparators"][variant]["scores"]
            high, low = scores["pandora_minkxx"], scores["pandora_minkxx_feature"]
            require((high is None and low is None) or (high is not None and low == -high), "Uniform score sign differs.")
    require(len(values) == 18 and all(v is None or (not isinstance(v, bool) and math.isfinite(v))
                                     for method in values.values() for v in method.values()), "Invalid scalar methods.")
    required = [f"{variant}.{metric}" for variant in ("fp32_centered", "fp32_second_moment") for metric in METRICS]
    required.append("pandora_uniform_fp32.minkxx")
    require(all(value is not None for name in required for value in values[name].values()),
            "A required method is missing a dev/test scalar; no primary cohort selection is allowed.")
    stored_vectors = read_lines(analysis / "test_score_vectors.jsonl")
    require([r["id"] for r in stored_vectors] == [r["id"] for r in test], "Stored test vectors/order differ.")
    for row, original in zip(stored_vectors, test):
        require(row["official_label"] == original["official_label"] and set(row["scores"]) == set(values) | {BLIND},
                "Test-vector score family/labels differ.")
        for name in values:
            close_number(row["scores"][name], values[name][row["id"]], f"Test vector {name}/{row['id']}")
    values[BLIND] = {r["id"]: r["scores"][BLIND] for r in stored_vectors}
    blind = blind_replay(source, read_json(analysis / "blind_parameters.json"), values[BLIND], refit=refit_blind)
    methods = read_json(analysis / "methods.json")
    require(set(methods) == set(values), "Missing/extra reported method.")
    tails = read_lines(analysis / "tail_records.jsonl")
    tail_lookup = {(r["method"], r["id"]): r for r in tails}
    require(len(tails) == len(tail_lookup) == 18 * 2000 and
            set(tail_lookup) == {(name, i) for name in values if name != BLIND for i in positive_ids + negative_ids},
            "Missing/duplicate/extra tail records.")
    own_methods = {}
    complete_vectors = {}
    for name, scores in values.items():
        pos = [scores[i] for i in positive_ids if scores[i] is not None]
        neg = [scores[i] for i in negative_ids if scores[i] is not None]
        missing_pos, missing_neg = 1000 - len(pos), 1000 - len(neg)
        complete = missing_pos == missing_neg == 0
        win2 = pair_wins_twice(pos, neg)
        bounds = [win2 / 2_000_000, (win2 + 2 * (1_000_000 - len(pos) * len(neg))) / 2_000_000]
        point = direct_auc(pos, neg) if complete else None
        entry = methods[name]
        require(entry["status"] == ("complete" if complete else "incomplete_full_cohort_no_complete_case_auc")
                and entry["test_members"] == entry["test_nonmembers"] == 1000
                and entry["missing_member_scores"] == missing_pos and entry["missing_nonmember_scores"] == missing_neg,
                f"Full-cohort status/counts differ: {name}.")
        close_number(entry["auc"], point, f"AUC {name}")
        close_array(entry["auc_completion_bounds"], bounds, f"AUC completion bounds {name}")
        own_methods[name] = {"auc": point, "auc_completion_bounds": bounds,
                             "missing_test_members": missing_pos, "missing_test_nonmembers": missing_neg}
        if complete:
            complete_vectors[name] = (np.asarray(pos), np.asarray(neg))
        else:
            require(entry["paired_within_arm_bootstrap_95pct_interval"] is None, "Incomplete method received bootstrap CI.")
        if name == BLIND:
            require("dev_calibrated_tail" not in entry, "Blind baseline received undeclared model-score calibration.")
            continue
        missing_dev = sum(scores[i] is None for i in dev_ids)
        available_dev = 1000 - missing_dev
        tail_complete = complete and missing_dev == 0
        tail = entry["dev_calibrated_tail"]
        require(tail["status"] == ("complete" if tail_complete else "unavailable_original_calibration_or_test_scores_missing")
                and tail["calibration_nonmembers"] == tail["attempted_calibration_nonmembers"] == 1000
                and tail["available_calibration_scores"] == available_dev and tail["missing_calibration_nonmembers"] == missing_dev
                and tail["test_nonmember_denominator"] == tail["test_member_denominator"] == 1000,
                f"Tail availability/denominators differ: {name}.")
        close_number(tail["alpha"], .01, f"Tail alpha {name}")
        close_number(tail["minimum_p_value"], 1 / 1001 if tail_complete else None, f"Minimum p {name}")
        p_values = {}
        flags = {}
        if tail_complete:
            calibration = np.asarray([scores[i] for i in dev_ids])
            for identifier in positive_ids + negative_ids:
                greater_or_equal = int(np.count_nonzero(calibration >= scores[identifier]))
                p_values[identifier] = (1 + greater_or_equal) / 1001
                flags[identifier] = 100 * (1 + greater_or_equal) <= 1001
        for identifiers, field in ((positive_ids, "test_member_flags"), (negative_ids, "test_nonmember_flags")):
            require(tail[field] == (sum(flags[i] for i in identifiers) if tail_complete else None), f"Tail flag total {name}/{field}.")
        for identifier in positive_ids + negative_ids:
            stored = tail_lookup[name, identifier]
            require(stored["official_label"] == by_id[identifier]["source"]["official_label"], "Tail label differs.")
            close_number(stored["score"], scores[identifier], f"Tail score {name}/{identifier}")
            close_number(stored["calibration_p"], p_values.get(identifier), f"Tail p {name}/{identifier}")
            expected_flag = flags.get(identifier)
            require((stored["flag_at_0_01"] is None if expected_flag is None else
                     type(stored["flag_at_0_01"]) is bool and stored["flag_at_0_01"] == expected_flag),
                    f"Tail flag/missingness differs: {name}/{identifier}.")
        own_methods[name].update(missing_dev_nonmembers=missing_dev, tail_available=tail_complete,
                                 tail_test_member_flags=tail["test_member_flags"], tail_test_nonmember_flags=tail["test_nonmember_flags"])
    require(summary["bootstrap"]["draws"] == DRAWS and summary["bootstrap"]["seed"] == SEED
            and summary["bootstrap"]["paired_across_methods"] is True, "Bootstrap design differs.")
    computed = bootstrap_ranks(complete_vectors)
    saved = np.load(analysis / "paired_bootstrap_aucs.npz", allow_pickle=False)
    require(set(saved.files) == set(computed), "Bootstrap includes missing methods or omits complete methods.")
    max_bootstrap = {}
    for name, draws in computed.items():
        max_bootstrap[name] = close_array(saved[name], draws, f"All bootstrap draws {name}")
        close_array(methods[name]["paired_within_arm_bootstrap_95pct_interval"], linear_interval(draws), f"AUC CI {name}")
    saved.close()
    np.savez_compressed(output / "independent_rank_bootstrap.npz", **computed)
    contrasts = read_json(analysis / "contrasts.json")
    require(len(contrasts) == len(CONTRASTS), "Wrong declared contrast count.")
    for stored, (role, left, right) in zip(contrasts, CONTRASTS):
        require((stored["role"], stored["left"], stored["right"]) == (role, left, right), "Contrast selection/order differs.")
        complete = left in computed and right in computed
        require(stored["status"] == ("complete" if complete else "unavailable_incomplete_method_coverage"), "Contrast missingness differs.")
        close_number(stored["auc_difference"], own_methods[left]["auc"] - own_methods[right]["auc"] if complete else None,
                     f"Contrast point {left}/{right}")
        bounds = [own_methods[left]["auc_completion_bounds"][0] - own_methods[right]["auc_completion_bounds"][1],
                  own_methods[left]["auc_completion_bounds"][1] - own_methods[right]["auc_completion_bounds"][0]]
        close_array(stored["auc_difference_completion_bounds"], bounds, f"Contrast completion bounds {left}/{right}")
        if complete:
            close_array(stored["paired_within_arm_bootstrap_95pct_interval"], linear_interval(computed[left] - computed[right]),
                        f"Paired contrast CI {left}/{right}")
        else:
            require(stored["paired_within_arm_bootstrap_95pct_interval"] is None, "Incomplete contrast received CI.")
    final_hashes = {"scoring_execution": sha_file(scoring / "execution.json"),
                    "analysis_summary": sha_file(analysis / "summary.json"), "data_seal": sha_file(sealed / "seal.json")}
    require(initial_hashes == final_hashes, "Execution, analysis summary, or data seal changed during verification.")
    for directory, hashes in ((sealed, seal["files_sha256"]), (analysis, summary["artifact_sha256"]),
                              (scoring, execution["source_sha256"])):
        for name, expected_hash in hashes.items():
            require(sha_file(directory / name) == expected_hash, f"Frozen artifact changed during verification: {name}.")
    require(sha_file(scoring / "scores.jsonl") == execution["scores_sha256"]
            and sha_file(scoring / "protocol.json") == execution["protocol_sha256"]
            and sha_file(scoring / "amendment.json") == execution["amendment_sha256"],
            "Scalars, protocol, or amendment changed during verification.")
    optional_counts = Counter(f"{row['split']}/label{row['official_label']}/{section}"
                              for row in optional for section in row["failed_sections"])
    return {"status": "verified_numeric_artifacts", "tolerance": TOLERANCE,
            "cohort": {"scored": 3000, "dev_nonmembers": 1000, "test_members": 1000, "test_nonmembers": 1000},
            "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
            "methods_checked": own_methods, "blind_parameter_replay": blind,
            "optional_failure_record_count": len(optional), "tail_records_checked": len(tails), "numerical_support": support,
            "optional_failed_sections_by_split_label": dict(optional_counts),
            "bootstrap": {"draws": DRAWS, "seed": SEED, "implementation": "SciPy average ranks on shared resampling indices",
                          "max_absolute_draw_difference_by_method": max_bootstrap},
            "declared_contrasts_checked": len(CONTRASTS),
            "input_hashes": final_hashes,
            "limits": ["Numerical reconstruction, not validation by assistant agreement or a new empirical sample",
                       "Publisher before/after-bank labels remain conditional; original training consumption not independently replayed",
                       "No population FPR/FDR, unknown-source independence, novelty, or causal-effect guarantee",
                       "Raw per-token scalar reconstruction belongs to the separate support verifier"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-data", type=Path, required=True)
    parser.add_argument("--scoring", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New verification directory.")
    parser.add_argument("--refit-blind", action="store_true", help="Also reproduce the bounded dev-only text classifier fit.")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output must be new; existing evidence is preserved.")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), args.output / "verify_checkmia_analysis.py")
    code_hash = sha_file(Path(__file__))
    started = time.perf_counter()
    report: dict[str, Any] = {"status": "starting", "code_sha256": code_hash}
    destination = args.output / "verification.json"
    destination.write_bytes(encoded(report))
    try:
        report = verify(args.sealed_data, args.scoring, args.analysis, args.output, refit_blind=args.refit_blind)
        require(sha_file(Path(__file__)) == code_hash, "Verifier source changed during execution.")
        return 0
    except BaseException as error:
        report = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
        raise
    finally:
        report.update(code_sha256=code_hash, wall_seconds=time.perf_counter() - started,
                      versions={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")})
        report["artifact_sha256"] = {p.name: sha_file(p) for p in args.output.iterdir()
                                     if p.is_file() and p.name != "verification.json"}
        destination.write_bytes(encoded(report))


if __name__ == "__main__":
    raise SystemExit(main())
