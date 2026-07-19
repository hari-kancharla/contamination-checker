#!/usr/bin/env python3
"""Sealed, real-data-only evaluation of the frozen observable comparison policy.

Two phases: `seal` builds source components/cases and fits only development
text; `evaluate` verifies that seal before computing held-out case outcomes.
No target-model inference, upstream program execution, or synthetic text.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import time
from urllib.parse import urlsplit, urlunsplit
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits


PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from contamcheckr.calibration import (  # noqa: E402
    control_null_check, domain_fingerprint, domain_shift_check,
)
from contamcheckr.stats import mann_whitney_u_test  # noqa: E402
from contamcheckr.validity import (  # noqa: E402
    POLICY_VERSION, combine_diagnostic_pvalues, likelihood_comparison_validity,
)


NAMESPACE = "real-data-selectivity-v1/20261107"
DOMAINS = ("dclm", "openwebmath", "pes2o", "starcoder")
MODELS = tuple(f"OLMo2-{s}-Instruct" for s in ("1B", "7B", "13B", "32B"))
IMPLEMENTATIONS = ("contamcheckr/validity.py", "contamcheckr/calibration.py",
                   "contamcheckr/stats.py", "contamcheckr/metrics.py",
                   "contamcheckr/methods/canonical_order.py")
RESOLUTIONS = {
    "hash_serialization": "SHA256 of UTF8 JSON compact array of namespace and ordered string parts; ensure_ascii=False",
    "source_normalization": "Global url: identities across web domains: strip surrounding whitespace and fragment, lower scheme/hostname only, preserve path/query/port/userinfo; paper IDs strip str(value); repository names strip+casefold",
    "unknown_source_components": "Exclude a primary component if any row lacks its domain-specific source identity, including unknown code repository provenance",
    "cross_domain_components": "Exclude from all primary banks and blind training/evaluation; count and preserve",
    "dev_representatives": "Within dev split, one min(raw_text_sha256,path,row) per component/label/condition",
    "test_representatives": "Within test split after whole-component dev quarantine, same fixed representative rule",
    "dev_partitions": "Hash-sort16 components by namespace,dev_partition,domain,block,attempt,cid; first8/last8; skip identical and complementary group partitions; record attempts until32 unique",
    "folds": "Within each origin, hash-sort by namespace,origin_fold,cid then cyclically assign to four folds; no case outcome or membership label enters hash",
    "blind_baseline": "One pooled matched-dev-member versus dev-nonmember fit across4domains, balanced class weights; report pooled and per-domain heldout AUCs, no tuning",
    "other_sizes": "Same sealed cases and dev text fits; 1B primary,7B/13B/32B correlated secondary views",
    "text_inputs": "Unmodified full raw text for origin/blind/fingerprint features; canonical score lookup uses SHA256(text.strip()) and source leaf; no metadata features",
    "common_quality": "Exactly8+8 distinct components, verified raw-text/vector joins, all LOSS and MinK++ scores and positive token counts; never choose substitutes using score values",
    "unknown_code_sensitivity": "Separate descriptive bank admitting unknown-source code components, with identical hash/block/representative rules; no new independence claim",
    "unavailable_member_views": "Count every null-linked member template as attempted; preserve unavailable8-member construction separately from complete actual comparisons",
    "classifier_failure": "Empty vocabulary predicts0.5 with explicit status; convergence warnings or other fitting failures make gate unavailable",
    "matched_coverage": "Within each model×view×domain stratum and each model×view all-domain stratum separately; rank complete cases without diagnostic outcomes by origin statistic or abs(log median length ratio), tie case hash",
    "ablations": "Descriptive leave-one-blocker-out numeric_shift,length_shift,heterogeneous_control on same cases; all remaining reasons preserved",
}


def encoded(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def hash_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hkey(*parts):
    return digest(encoded([NAMESPACE, *[str(p) for p in parts]]))


def write_json(path, obj):
    path.write_bytes(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode() + b"\n")


def write_jsonl(path, rows):
    with path.open("wb") as stream:
        for row in rows:
            stream.write(encoded(row) + b"\n")


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def auc(positive, negative):
    if not positive or not negative:
        return None
    return sum((p > n) + .5 * (p == n) for p in positive for n in negative) / (len(positive) * len(negative))


def normalize_url(value):
    parts = urlsplit(value.strip())
    if not parts.hostname:
        return None
    # Preserve userinfo/port and path/query; only the actual hostname changes case.
    netloc = parts.netloc
    host = parts.hostname
    start = netloc.rfind("@") + 1
    if netloc[start:start + 1] == "[":
        stop = netloc.find("]", start) + 1
        netloc = netloc[:start] + netloc[start:stop].lower() + netloc[stop:]
    else:
        netloc = netloc[:start] + host.lower() + netloc[start + len(host):]
    return urlunsplit((parts.scheme.lower(), netloc, parts.path, parts.query, ""))


def source_identity(domain, record):
    field = {"dclm": "url", "openwebmath": "url", "pes2o": "id", "starcoder": "max_stars_repo_name"}[domain]
    value = record.get(field)
    if value is None or not str(value).strip():
        return None
    value = str(value).strip()
    if domain in ("dclm", "openwebmath"):
        value = normalize_url(value)
        return f"url:{value}" if value else None
    elif domain == "starcoder":
        value = value.casefold()
        if value in {"none", "null", "unknown", "n/a"}:
            value = None
    return f"{domain}:{field}:{value}" if value else None


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        a, b = self.find(i), self.find(j)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


@dataclass
class Data:
    records: list
    by_id: dict
    components: list
    representatives: list
    sources: dict
    source_identity_inventory: dict


def collect_data(snapshot, audit_dir):
    sources = {}
    # Validate the independently audited canonical artifacts and their inputs.
    for name, expected in read_json(audit_dir / "artifact_hashes.json").items():
        path = audit_dir / name
        if hash_file(path) != expected:
            raise ValueError(f"Canonical audit artifact changed: {name}")
    for _, entry in read_json(audit_dir / "sources.json").items():
        path = Path(entry["path"])
        if hash_file(path) != entry["sha256"]:
            raise ValueError(f"Canonical source changed: {path}")
        sources[str(path.resolve())] = entry["sha256"]
    for name in ("normalized_rows.jsonl", "benchmark_rows.jsonl", "summary.json", "artifact_hashes.json", "sources.json"):
        sources[str((audit_dir / name).resolve())] = hash_file(audit_dir / name)
    canonical = read_jsonl(audit_dir / "normalized_rows.jsonl")
    score_map = {}
    for row in canonical:
        key = (row["model"], row["domain"], row["subset"], row["condition"], row["split"], row["text_hash"], row["method"])
        if key in score_map:
            raise ValueError("Duplicate canonical method key")
        score_map[key] = row
    records = []
    for path in sorted((snapshot / "benchmark/pretraining").rglob("*.jsonl")):
        relative = str(path.relative_to(snapshot))
        parts = Path(relative).parts
        domain = parts[2]
        label = int(parts[3] == "member")
        condition = parts[4] if label else "shared"
        split = parts[-2]
        subset = next((s for s in ("java", "python") if f"_{s}_" in path.name), "") if domain == "starcoder" else ""
        sources[str(path.resolve())] = hash_file(path)
        for index, raw in enumerate(read_jsonl(path)):
            text = raw["text"]
            raw_hash = digest(text.encode())
            text_hash = digest(text.strip().encode())
            normalized_hash = digest(" ".join(text.casefold().split()).encode())
            identifier = f"{relative}:{index}:{raw_hash}"
            scores = {}
            for model in MODELS:
                scores[model] = {}
                for method in ("LOSS", "MinK", "MinK++"):
                    key = (model, domain, subset, condition, split, text_hash, method)
                    saved = score_map.get(key)
                    if saved is None:
                        continue
                    if saved["benchmark_path"] != relative or saved["label"] != label:
                        raise ValueError(f"Canonical source-leaf mismatch: {identifier}")
                    # Saved source record_index is intentionally NOT a join key.
                    if saved["join_status"] != "unique_exact_stripped_text_hash":
                        raise ValueError(f"Unresolved score hash join: {identifier}")
                    scores[model][method] = {"score": saved["score"], "token_count": saved["token_count"],
                                            "path": saved["path"], "result_row_index": saved["result_row_index"]}
            records.append({"row_id": identifier, "path": relative, "row_index": index,
                            "raw_text_sha256": raw_hash, "text_hash": text_hash,
                            "normalized_text_sha256": normalized_hash, "text": text,
                            "domain": domain, "subset": subset, "label": label, "condition": condition,
                            "split": split, "source_identity": source_identity(domain, raw), "scores": scores})
    union = UnionFind(len(records))
    seen = {}
    labels_by_text = defaultdict(set)
    for i, row in enumerate(records):
        labels_by_text[row["normalized_text_sha256"]].add(row["label"])
        keys = [("text", row["normalized_text_sha256"])]
        if row["source_identity"]:
            keys.append(("source", row["source_identity"]))
        for key in keys:
            if key in seen:
                union.union(i, seen[key])
            else:
                seen[key] = i
    grouped = defaultdict(list)
    for i, row in enumerate(records):
        grouped[union.find(i)].append(row)
    components, representatives = [], []
    for rows in grouped.values():
        cid = min(r["row_id"] for r in rows)
        reasons = []
        if len({r["domain"] for r in rows}) > 1:
            reasons.append("cross_domain_component")
        if any(len(labels_by_text[r["normalized_text_sha256"]]) > 1 for r in rows):
            reasons.append("contradictory_official_labels_for_normalized_text")
        if any(r["source_identity"] is None for r in rows):
            reasons.append("unknown_source_provenance")
        dev_touch = any(r["split"] == "dev" for r in rows)
        component = {"component_id": cid, "row_ids": sorted(r["row_id"] for r in rows),
                     "domains": sorted({r["domain"] for r in rows}),
                     "source_identities": sorted({r["source_identity"] for r in rows if r["source_identity"]}),
                     "touches_dev": dev_touch, "base_exclusions": reasons,
                     "test_exclusions": reasons + (["component_touches_dev"] if dev_touch else []),
                     "unknown_source_rows": sum(r["source_identity"] is None for r in rows)}
        components.append(component)
        cells = defaultdict(list)
        for row in rows:
            row["component_id"] = cid
            cells[row["split"], row["label"], row["condition"]].append(row)
        for (split, label, condition), options in cells.items():
            representative = min(options, key=lambda r: (r["raw_text_sha256"], r["path"], r["row_index"]))
            exclusions = component["test_exclusions"] if split == "test" else component["base_exclusions"]
            representatives.append({"row_id": representative["row_id"], "component_id": cid,
                                    "domain": representative["domain"], "subset": representative["subset"],
                                    "split": split, "label": label, "condition": condition,
                                    "primary_eligible": not exclusions, "exclusions": exclusions,
                                    "unknown_source": "unknown_source_provenance" in reasons})
    url_domains = defaultdict(set)
    for row in records:
        if row["source_identity"] and row["source_identity"].startswith("url:"):
            url_domains[row["source_identity"]].add(row["domain"])
    identity_inventory = {"unique_global_canonical_URLs": len(url_domains),
                          "cross_domain_URL_collisions": {url: sorted(domains) for url, domains in url_domains.items() if len(domains) > 1},
                          "URL_namespace": "global_across_DCLM_and_OpenWebMath",
                          "preseal_change_from_domain_prefixed_URLs": "No component/case impact on pinned data:903 canonical URLs and zero cross-domain URL collisions"}
    return Data(records, {r["row_id"]: r for r in records}, sorted(components, key=lambda c: c["component_id"]),
                sorted(representatives, key=lambda r: (r["component_id"], r["split"], r["label"], r["condition"])), sources, identity_inventory)


def ordered(representatives, domain):
    return sorted(representatives, key=lambda r: (hkey(domain, r["component_id"]), r["component_id"]))


def eligible_reps(data, *, split, label, condition, domain=None, unknown_code=False):
    result = []
    for row in data.representatives:
        if row["split"] != split or row["label"] != label or row["condition"] != condition:
            continue
        if domain is not None and row["domain"] != domain:
            continue
        remaining = [r for r in row["exclusions"] if not (unknown_code and row["domain"] == "starcoder" and r == "unknown_source_provenance")]
        if not remaining:
            result.append(row)
    return result


def assignments(data, *, unknown_code=False):
    cases, remainders = [], []
    bank = "unknown_code_inclusive_sensitivity" if unknown_code else "primary"
    null_by_domain = {}
    for domain in DOMAINS:
        reps = ordered(eligible_reps(data, split="test", label=0, condition="shared", domain=domain,
                                     unknown_code=unknown_code), domain)
        complete = len(reps) // 16
        null_by_domain[domain] = []
        for index in range(complete):
            block = reps[index * 16:(index + 1) * 16]
            case = {"case_id": f"{bank}/{domain}/null/{index}", "bank": bank, "domain": domain,
                    "condition": "nonmember_null", "index": index,
                    "suspect": [r["row_id"] for r in block[:8]], "reference": [r["row_id"] for r in block[8:]],
                    "construction_status": "available", "published_suspect_label": 0,
                    "published_reference_label": 0, "independence": "disjoint_primary_nonmember_components" if not unknown_code else "descriptive_provenance_sensitivity_dependent_on_primary"}
            case["case_hash"] = hkey("case", case["case_id"])
            cases.append(case)
            null_by_domain[domain].append(case)
        remainders.append({"bank": bank, "domain": domain, "eligible_nonmember_groups": len(reps),
                           "complete_null_cases": complete, "unused_remainder": [r["row_id"] for r in reps[complete * 16:]]})
    for domain in DOMAINS:
        for condition in ("matched", "shifted"):
            pool = ordered(eligible_reps(data, split="test", label=1, condition=condition, domain=domain,
                                         unknown_code=unknown_code), domain)
            used = set()
            for null in null_by_domain[domain]:
                reference_components = {data.by_id[r]["component_id"] for r in null["reference"]}
                options = [r for r in pool if r["component_id"] not in used | reference_components]
                selected = []
                if domain == "starcoder":
                    required = Counter(data.by_id[r]["subset"] for r in null["reference"])
                    for row in options:
                        if required[row["subset"]] > 0:
                            selected.append(row)
                            required[row["subset"]] -= 1
                        if len(selected) == 8:
                            break
                else:
                    selected = options[:8]
                available = len(selected) == 8
                if available:
                    used.update(r["component_id"] for r in selected)
                case_id = f"{bank}/{domain}/{condition}_member/{null['index']}"
                cases.append({**null, "case_id": case_id, "case_hash": hkey("case", case_id),
                              "condition": condition + "_member", "suspect": [r["row_id"] for r in selected] if available else [],
                              "construction_status": "available" if available else "unavailable_member_groups_or_code_quota",
                              "published_suspect_label": 1, "paired_null_case": null["case_id"],
                              "independence": "paired_reference_reuse_and_correlated_member_conditions"})
    for index, domain in enumerate(DOMAINS):
        next_domain = DOMAINS[(index + 1) % len(DOMAINS)]
        references = null_by_domain[next_domain]
        for null in null_by_domain[domain]:
            available = bool(references)
            reference_case = references[null["index"] % len(references)] if available else None
            case_id = f"{bank}/{domain}/mismatch/{null['index']}"
            cases.append({**null, "case_id": case_id, "case_hash": hkey("case", case_id),
                          "condition": "cross_domain_nonmember_stress", "reference_domain": next_domain,
                          "reference": reference_case["reference"] if available else [],
                          "construction_status": "available" if available else "unavailable_next_domain_block",
                          "paired_null_case": null["case_id"],
                          "reference_null_case": reference_case["case_id"] if available else None,
                          "independence": "dependent_reused_nonmember_mismatch_stress_not_additional_null_trials"})
    return cases, remainders


def vectorizer(vocabulary=None):
    return TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=True, min_df=1,
                           max_features=20000, sublinear_tf=True, norm="l2", vocabulary=vocabulary)


def classifier(*, balanced=False):
    return LogisticRegression(C=1, solver="liblinear", max_iter=1000, random_state=20261107,
                              class_weight="balanced" if balanced else None)


class Origin:
    def __init__(self, data):
        self.data = data
        self.fit_counts = Counter()

    def score(self, suspect, reference):
        fold_map = {}
        for origin, ids in ((1, suspect), (0, reference)):
            ordered_ids = sorted(ids, key=lambda rid: (hkey("origin_fold", self.data.by_id[rid]["component_id"]), rid))
            for index, rid in enumerate(ordered_ids):
                fold_map[rid] = (origin, index % 4)
        if len(fold_map) != 16:
            return {"status": "unavailable_nonunique_or_incomplete_groups", "statistic": None, "folds": []}
        results = []
        for fold in range(4):
            training = [rid for rid, (_, f) in fold_map.items() if f != fold]
            test = [rid for rid, (_, f) in fold_map.items() if f == fold]
            train_y = [fold_map[rid][0] for rid in training]
            test_y = [fold_map[rid][0] for rid in test]
            self.fit_counts["attempted"] += 1
            try:
                v = vectorizer()
                matrix = v.fit_transform([self.data.by_id[rid]["text"] for rid in training])
                model = classifier()
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(matrix, train_y)
                predictions = model.decision_function(v.transform([self.data.by_id[rid]["text"] for rid in test])).tolist()
                status = "complete"
                self.fit_counts["successful"] += 1
            except ValueError as exc:
                if "empty vocabulary" not in str(exc).lower():
                    self.fit_counts["failed"] += 1
                    return {"status": "unavailable_fit_error", "statistic": None, "error": str(exc), "folds": results}
                predictions = [0.5] * len(test)
                status = "empty_vocabulary_constant_half"
                self.fit_counts["empty_vocabulary"] += 1
            except Exception as exc:
                self.fit_counts["failed"] += 1
                return {"status": "unavailable_fit_error", "statistic": None,
                        "error": f"{type(exc).__name__}: {exc}", "folds": results}
            fold_auc = auc([p for p, y in zip(predictions, test_y) if y],
                           [p for p, y in zip(predictions, test_y) if not y])
            results.append({"fold": fold, "status": status, "auc": fold_auc,
                            "training_row_ids": training, "heldout_row_ids": test,
                            "heldout_origin_labels": test_y, "heldout_decision_scores": predictions})
        return {"status": "complete", "statistic": sum(r["auc"] for r in results) / 4, "folds": results}


def fit_development(data):
    origin = Origin(data)
    partitions, blocks, remainders = [], [], []
    for domain in DOMAINS:
        reps = ordered(eligible_reps(data, split="dev", label=0, condition="shared", domain=domain), domain)
        n_blocks = len(reps) // 16
        remainders.append({"domain": domain, "eligible_groups": len(reps), "complete_blocks": n_blocks,
                           "unused_representatives": [r["row_id"] for r in reps[n_blocks * 16:]]})
        for block_index in range(n_blocks):
            block = reps[block_index * 16:(block_index + 1) * 16]
            identifiers = [r["row_id"] for r in block]
            used, attempts, successes = set(), 0, 0
            while successes < 32:
                if attempts >= 10000:
                    raise ValueError("Could not construct32 unique balanced dev partitions")
                permutation = sorted(identifiers, key=lambda rid: (hkey("dev_partition", domain, block_index, attempts,
                                                                        data.by_id[rid]["component_id"]), rid))
                signature = min(tuple(sorted(permutation[:8])), tuple(sorted(permutation[8:])))
                attempt = attempts
                attempts += 1
                if signature in used:
                    continue
                used.add(signature)
                record = {"domain": domain, "block": block_index, "partition_index": successes,
                          "attempt_index": attempt, "suspect": permutation[:8], "reference": permutation[8:],
                          **origin.score(permutation[:8], permutation[8:])}
                partitions.append(record)
                successes += 1
            blocks.append({"domain": domain, "block": block_index, "row_ids": identifiers,
                           "unique_partitions": successes, "attempts": attempts, "duplicate_or_complement_skips": attempts - successes})
    observed = [p["statistic"] for p in partitions if p["status"] == "complete"]
    operational = len(blocks) >= 2 and len(observed) == len(partitions)
    threshold = float(np.quantile(observed, .95, method="higher")) if operational else None
    calibration = {"threshold": threshold, "status": "development_fitted_heuristic" if operational else "uncalibrated",
                   "distinct_blocks": len(blocks), "partition_views": len(partitions),
                   "complete_partition_views": len(observed), "guarantee": "none; views share underlying blocks",
                   "quantile": .95, "quantile_method": "higher", "blocks": blocks,
                   "remainders": remainders, "fit_counts": dict(origin.fit_counts)}

    training = eligible_reps(data, split="dev", label=0, condition="shared") + eligible_reps(data, split="dev", label=1, condition="matched")
    training.sort(key=lambda r: r["row_id"])
    v, model = vectorizer(), classifier(balanced=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(v.fit_transform([data.by_id[r["row_id"]]["text"] for r in training]), [r["label"] for r in training])
    blind = {"status": "dev_fit_complete", "training": training, "classes": model.classes_.tolist(),
             "vocabulary": {k: int(i) for k, i in v.vocabulary_.items()}, "idf": v.idf_.tolist(),
             "coefficients": model.coef_.tolist(), "intercept": model.intercept_.tolist(),
             "class_weight": "balanced", "feature_count": len(v.vocabulary_),
             "training_counts": dict(Counter(f"{r['domain']}:{r['label']}" for r in training)),
             "test_results_computed": False}
    reconstructed = blind_decision(blind, [data.by_id[r["row_id"]]["text"] for r in training])
    expected = model.decision_function(v.transform([data.by_id[r["row_id"]]["text"] for r in training]))
    if not np.allclose(reconstructed, expected, rtol=0, atol=1e-12):
        raise ValueError("Blind model JSON roundtrip failed on development texts")
    blind["development_parameter_roundtrip_max_abs_error"] = float(np.max(np.abs(reconstructed - expected)))
    return calibration, partitions, blind


def blind_decision(model, texts):
    v = vectorizer(vocabulary=model["vocabulary"])
    v.idf_ = np.asarray(model["idf"], dtype=float)
    weights = np.asarray(model["coefficients"][0], dtype=float)
    return np.asarray(v.transform(texts) @ weights + float(model["intercept"][0])).reshape(-1)
