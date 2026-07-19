"""Analyze the completed sealed CheckMIABench study without retuning it."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def sha_file(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def auc(positive, negative):
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.sort(np.asarray(negative, dtype=np.float64))
    less = np.searchsorted(negative, positive, side="left")
    at_most = np.searchsorted(negative, positive, side="right")
    return float(np.sum((less + at_most) / 2) / (len(positive) * len(negative)))


def bootstrap_aucs(method_vectors, draws=3000, seed=20261107):
    """Paired within-arm bootstrap using tie-aware weighted direct pair counts.

    A fixed score map is grouped into equal-score buckets. Repeated sampled
    rows contribute their multiplicities, preserving cross-method pairing.
    """
    first = next(iter(method_vectors.values()))
    m, n = len(first[0]), len(first[1])
    buckets = {}
    for name, (positive, negative) in method_vectors.items():
        assert len(positive) == m and len(negative) == n
        unique, inverse = np.unique(np.concatenate([positive, negative]), return_inverse=True)
        buckets[name] = (inverse[:m], inverse[m:], len(unique))
    random = np.random.default_rng(seed)
    result = {name: np.empty(draws) for name in buckets}
    for draw in range(draws):
        mc = np.bincount(random.integers(0, m, size=m), minlength=m)
        nc = np.bincount(random.integers(0, n, size=n), minlength=n)
        for name, (mg, ng, count) in buckets.items():
            gm = np.bincount(mg, weights=mc, minlength=count)
            gn = np.bincount(ng, weights=nc, minlength=count)
            result[name][draw] = np.dot(gm, np.cumsum(gn) - .5 * gn) / (m * n)
    return result


def interval(values):
    return [float(v) for v in np.quantile(values, [.025, .975])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-data", type=Path, required=True)
    parser.add_argument("--scoring", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new directory.")
    execution = json.loads((args.scoring / "execution.json").read_text())
    if (execution["status"] != "complete" or execution["scored_records"] != 3000
            or execution.get("amendment_id") != "checkmia-weighted-vs-uniform-v1-amendment-1"):
        raise ValueError("Only the completed, fixed 3000-record execution can be analyzed.")
    if sha_file(Path(__file__)) != execution["source_sha256"]["checkmia_real_analysis.py"]:
        raise ValueError("Executing analysis differs from the source frozen before scoring.")
    if sha_file(args.scoring / "scores.jsonl") != execution["scores_sha256"]:
        raise ValueError("Completed scalar score artifact changed.")
    if sha_file(args.scoring / "amendment.json") != execution["amendment_sha256"]:
        raise ValueError("Execution amendment changed.")
    if sha_file(args.scoring / "protocol.json") != execution["protocol_sha256"]:
        raise ValueError("Execution protocol changed.")
    seal = json.loads((args.sealed_data / "seal.json").read_text())
    for name, expected in seal["files_sha256"].items():
        if sha_file(args.sealed_data / name) != expected:
            raise ValueError(f"Sealed input changed: {name}.")
    if sha_file(args.sealed_data / "seal.json") != execution["seal_sha256"]:
        raise ValueError("Scorer used a different data seal.")
    records = [json.loads(line) for line in (args.scoring / "scores.jsonl").read_text().splitlines()]
    by_id = {row["id"]: row for row in records}
    if len(records) != 3000 or len(by_id) != 3000 or any(not row["runner_eligibility"]["primary_complete"] or row["target_count"] != 2048 for row in records):
        raise ValueError("Repeated IDs, incomplete support, or a wrong target count.")
    for row in records:
        for name in ("fp32_second_moment", "fp32_centered"):
            variant = row["variants"][name]
            if (variant["status"] != "complete" or variant["supported_target_count"] != 2048 or not variant["aggregate_valid"]):
                raise ValueError("A required FP32 variant lacks full primary support.")
        if row["pandora_comparators"]["fp32"]["status"] != "complete":
            raise ValueError("The required FP32 uniform comparator lacks support.")
    with gzip.open(args.sealed_data / "inputs.jsonl.gz", "rt") as handle:
        source = [json.loads(line) for line in handle]
    train = [row for row in source if row["study_split"] == "dev"]
    test = sorted((row for row in source if row["study_split"] == "test"), key=lambda row: row["id"])
    for row in source:
        if row["study_split"] == "test" or row["official_label"] == 0:
            scored = by_id[row["id"]]
            if scored["source"] != {key: value for key, value in row.items() if key not in {"tokens", "text"}}:
                raise ValueError("Scored/source identity mismatch.")
    pos_ids = [row["id"] for row in test if row["official_label"] == 1]
    neg_ids = [row["id"] for row in test if row["official_label"] == 0]
    dev_ids = sorted(row["id"] for row in source if row["study_split"] == "dev" and row["official_label"] == 0)
    assert len(pos_ids) == len(neg_ids) == len(dev_ids) == 1000
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), args.output / "checkmia_real_analysis.py")
    started = time.perf_counter()
    # All method names are declared before outcomes, including negative results.
    extractors = {}
    for variant in ("author_native", "fp32_second_moment", "fp32_centered", "fixed_fp32_logp_bf16_reductions"):
        for metric in ("loss", "mink", "minkpp", "zlib"):
            extractors[f"{variant}.{metric}"] = ("variants", variant, metric)
    for variant in ("native", "fp32"):
        extractors[f"pandora_uniform_{variant}.minkxx"] = ("pandora_comparators", variant, "pandora_minkxx")
    values = {}
    for name, (section, variant, metric) in extractors.items():
        values[name] = {identifier: (None if row[section][variant]["scores"][metric] is None
                                     else float(row[section][variant]["scores"][metric])) for identifier, row in by_id.items()}
        if any(value is not None and not np.isfinite(value) for value in values[name].values()):
            raise ValueError("Nonfinite scalars must be explicitly missing with support failures retained.")
    required_methods = [f"{variant}.{metric}" for variant in ("fp32_second_moment", "fp32_centered")
                        for metric in ("loss", "mink", "minkpp", "zlib")] + ["pandora_uniform_fp32.minkxx"]
    if any(value is None or not np.isfinite(value) for name in required_methods for value in values[name].values()):
        raise ValueError("A required primary/comparator scalar is missing; no complete-case primary analysis is permitted.")
    # Fit the fixed model-free comparator only on the sealed development text.
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=True, min_df=1,
                                 max_features=20000, sublinear_tf=True, norm="l2")
    classifier = LogisticRegression(C=1., solver="liblinear", class_weight="balanced", max_iter=1000, random_state=20261107)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(vectorizer.fit_transform([row["text"] for row in train]), [row["official_label"] for row in train])
        blind_test = classifier.predict_proba(vectorizer.transform([row["text"] for row in test]))[:, 1]
    values["blind_char_tfidf.logistic"] = {row["id"]: float(score) for row, score in zip(test, blind_test)}
    blind_parameters = {
        "vocabulary": {term: int(index) for term, index in vectorizer.vocabulary_.items()}, "idf": vectorizer.idf_.tolist(),
        "coef": classifier.coef_.tolist(), "intercept": classifier.intercept_.tolist(),
        "classes": classifier.classes_.tolist(), "training_ids": [row["id"] for row in train],
        "features": "char3-5 TFIDF20k sublinear_tf L2 lowercase; no metadata",
        "classification": "C1 liblinear balanced max_iter1000 seed20261107; dev only, no tuning",
    }
    (args.output / "blind_parameters.json").write_bytes(encoded(blind_parameters))
    method_vectors = {name: ([scores[i] for i in pos_ids], [scores[i] for i in neg_ids]) for name, scores in values.items()
                      if all(scores[i] is not None for i in pos_ids + neg_ids)}
    boot = bootstrap_aucs(method_vectors)
    np.savez_compressed(args.output / "paired_bootstrap_aucs.npz", **boot)
    methods = {}
    tail_rows = []
    for name, scores in values.items():
        positive = [scores[i] for i in pos_ids if scores[i] is not None]
        negative = [scores[i] for i in neg_ids if scores[i] is not None]
        missing_positive, missing_negative = len(pos_ids) - len(positive), len(neg_ids) - len(negative)
        complete = missing_positive == missing_negative == 0
        known_pairs = len(positive) * len(negative)
        total_pairs = len(pos_ids) * len(neg_ids)
        known_wins = auc(positive, negative) * known_pairs if known_pairs else 0.
        bounds = [known_wins / total_pairs, (known_wins + total_pairs - known_pairs) / total_pairs]
        entry = {"status": "complete" if complete else "incomplete_full_cohort_no_complete_case_auc",
                 "auc": auc(positive, negative) if complete else None,
                 "auc_completion_bounds": bounds,
                 "paired_within_arm_bootstrap_95pct_interval": interval(boot[name]) if complete else None,
                 "test_members": len(pos_ids), "test_nonmembers": len(neg_ids),
                 "missing_member_scores": missing_positive, "missing_nonmember_scores": missing_negative}
        if not name.startswith("blind_"):
            missing_dev = sum(scores[identifier] is None for identifier in dev_ids)
            tail_complete = complete and missing_dev == 0
            calibration = np.sort([scores[identifier] for identifier in dev_ids if scores[identifier] is not None])
            ps = ({identifier: (1 + len(calibration) - np.searchsorted(calibration, scores[identifier], side="left")) / (len(calibration) + 1)
                   for identifier in pos_ids + neg_ids} if tail_complete else {})
            entry["dev_calibrated_tail"] = {
                "status": "complete" if tail_complete else "unavailable_original_calibration_or_test_scores_missing",
                "alpha": .01, "calibration_nonmembers": len(dev_ids), "available_calibration_scores": len(calibration),
                "minimum_p_value": 1 / (len(dev_ids) + 1) if tail_complete else None,
                "attempted_calibration_nonmembers": len(dev_ids), "missing_calibration_nonmembers": missing_dev,
                "test_nonmember_flags": int(sum(ps[i] <= .01 for i in neg_ids)) if tail_complete else None,
                "test_member_flags": int(sum(ps[i] <= .01 for i in pos_ids)) if tail_complete else None,
                "test_nonmember_denominator": len(neg_ids), "test_member_denominator": len(pos_ids),
                "scope": "Per-record empirical fixed-cohort false-positive and sensitivity counts; no multiple-testing/FDR or population guarantee.",
            }
            for identifier in pos_ids + neg_ids:
                tail_rows.append({"method": name, "id": identifier, "official_label": by_id[identifier]["source"]["official_label"],
                                  "score": values[name][identifier], "calibration_p": float(ps[identifier]) if tail_complete else None,
                                  "flag_at_0_01": bool(ps[identifier] <= .01) if tail_complete else None})
        methods[name] = entry
    contrasts = []
    declared = [
        ("primary", "fp32_centered.minkpp", "fp32_centered.loss"),
        ("secondary", "fp32_centered.minkpp", "fp32_centered.mink"),
        ("secondary", "fp32_centered.minkpp", "pandora_uniform_fp32.minkxx"),
        ("secondary", "fp32_centered.minkpp", "author_native.minkpp"),
        ("secondary", "fp32_centered.minkpp", "fp32_second_moment.minkpp"),
        ("secondary", "pandora_uniform_fp32.minkxx", "pandora_uniform_native.minkxx"),
    ]
    for role, left, right in declared:
        complete = methods[left]["auc"] is not None and methods[right]["auc"] is not None
        contrasts.append({"role": role, "left": left, "right": right,
                          "status": "complete" if complete else "unavailable_incomplete_method_coverage",
                          "auc_difference": methods[left]["auc"] - methods[right]["auc"] if complete else None,
                          "paired_within_arm_bootstrap_95pct_interval": interval(boot[left] - boot[right]) if complete else None,
                          "auc_difference_completion_bounds": [methods[left]["auc_completion_bounds"][0] - methods[right]["auc_completion_bounds"][1],
                                                               methods[left]["auc_completion_bounds"][1] - methods[right]["auc_completion_bounds"][0]],
                          "interpretation": "Exploratory conditional fixed-bank contrast, not a causal effect or confirmatory novelty claim."})
    (args.output / "methods.json").write_bytes(encoded(methods))
    (args.output / "contrasts.json").write_bytes(encoded(contrasts))
    with (args.output / "tail_records.jsonl").open("wb") as handle:
        for row in tail_rows:
            handle.write(encoded(row))
    with (args.output / "test_score_vectors.jsonl").open("wb") as handle:
        for row in test:
            handle.write(encoded({"id": row["id"], "official_label": row["official_label"],
                                  "scores": {name: scores[row["id"]] for name, scores in values.items()}}))
    numerical_support = {name: {"records": sum(name in row["variants"] for row in records),
                                "clamped_targets": sum(row["variants"].get(name, {}).get("clamped_target_count", 0) for row in records)}
                         for name in {n for row in records for n in row["variants"]}}
    summary = {
        "status": "complete", "protocol_id": "checkmia-weighted-vs-uniform-v1",
        "amendment_id": execution["amendment_id"], "amendment_sha256": execution["amendment_sha256"],
        "source_execution_sha256": sha_file(args.scoring / "execution.json"),
        "sealed_data_sha256": sha_file(args.sealed_data / "seal.json"),
        "model_revision": execution["model_revision"], "model_repo": execution["model_repo"],
        "scored_records": len(records), "scored_targets": sum(row["target_count"] for row in records),
        "split_counts": dict(Counter(f"{row['source']['study_split']}/{row['source']['arm']}" for row in records)),
        "numerical_support": numerical_support,
        "model_forward_seconds": sum(row["timing"]["forward_seconds"] for row in records),
        "score_engine_total_seconds": sum(row["timing"]["total_seconds"] for row in records),
        "execution_wall_seconds": execution["wall_seconds"], "analysis_wall_seconds": time.perf_counter() - started,
        "primary_failure_records": 0, "optional_failure_records": execution["optional_failure_record_count"],
        "optional_failure_record_ids": execution["optional_failure_records"], "all_attempted_records_accounted": True,
        "bootstrap": {"draws": 3000, "seed": 20261107, "paired_across_methods": True,
                      "scope": "Within-label item bootstrap conditional on this finite released bank; unknown source/semantic dependencies and family generalization are not covered."},
        "claim_limits": ["Publisher coarse before/after labels, not independently replayed training consumption",
                         "One intermediate checkpoint and one family; no population low-FPR validation",
                         "No benchmark-task inflation measurements or causal randomized exposure",
                         "Uniform comparators match inspected normalization/reduction, not a historic run manifest",
                         "New real-data baseline/definition comparison; novelty or top-venue readiness not established"],
    }
    files = [p for p in args.output.iterdir() if p.is_file()]
    summary["artifact_sha256"] = {p.name: sha_file(p) for p in files}
    (args.output / "summary.json").write_bytes(encoded(summary))
    print(json.dumps({"status": "complete", "primary_contrast": contrasts[0], "methods": methods,
                      "scored_records": len(records), "analysis_seconds": summary["analysis_wall_seconds"]}))


if __name__ == "__main__":
    main()
