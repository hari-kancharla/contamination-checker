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


def source_hashes(data, protocol):
    result = dict(data.sources)
    paths = [PROJECT / name for name in IMPLEMENTATIONS] + [Path(__file__), protocol,
             PROJECT / "docs/REAL_DATA_STUDY_PROTOCOL.md"]
    for path in paths:
        result[str(path.resolve())] = hash_file(path)
    return result


def verify_sources(sources):
    changed = [path for path, expected in sources.items() if not Path(path).is_file() or hash_file(Path(path)) != expected]
    if changed:
        raise ValueError(f"Frozen source bytes changed: {changed}")


def seal(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    protocol = read_json(args.protocol)
    if protocol["namespace"] != NAMESPACE or protocol["policy"]["version"] != POLICY_VERSION:
        raise ValueError("Unexpected protocol namespace or policy version")
    data = collect_data(args.snapshot, args.audit_dir)
    sources = source_hashes(data, args.protocol)
    cases, remainder = assignments(data)
    sensitivity, sensitivity_remainder = assignments(data, unknown_code=True)
    # The sensitivity is restricted to code; other domains' duplicate views
    # provide no additional information and are not emitted as extra trials.
    sensitivity = [c for c in sensitivity if c["domain"] == "starcoder"]
    calibration, partitions, blind = fit_development(data)
    verify_sources(sources)
    (args.output / "frozen_source").mkdir()
    for relative in IMPLEMENTATIONS:
        shutil.copyfile(PROJECT / relative, args.output / "frozen_source" / Path(relative).name)
    shutil.copyfile(Path(__file__), args.output / "frozen_source/real_selectivity_study.py")
    shutil.copyfile(args.protocol, args.output / "frozen_source/protocol.json")
    shutil.copyfile(PROJECT / "docs/REAL_DATA_STUDY_PROTOCOL.md", args.output / "frozen_source/PROTOCOL.md")
    write_json(args.output / "components.json", data.components)
    write_json(args.output / "representatives.json", data.representatives)
    write_jsonl(args.output / "case_assignments.jsonl", cases + sensitivity)
    write_json(args.output / "remainders.json", remainder + [r for r in sensitivity_remainder if r["domain"] == "starcoder"])
    write_json(args.output / "origin_calibration.json", calibration)
    write_jsonl(args.output / "origin_dev_partitions.jsonl", partitions)
    with gzip.open(args.output / "blind_dev_model.json.gz", "wb") as stream:
        stream.write(encoded(blind) + b"\n")
    raw_snapshot = [{k: v for k, v in r.items() if k not in ("text", "scores")} for r in data.records]
    write_jsonl(args.output / "record_provenance.jsonl", raw_snapshot)
    artifacts = {str(p.relative_to(args.output)): hash_file(p) for p in sorted(args.output.rglob("*")) if p.is_file()}
    manifest = {"protocol_id": protocol["protocol_id"], "status": "sealed_before_heldout_case_outcomes",
                "created_utc": datetime.now(timezone.utc).isoformat(), "namespace": NAMESPACE,
                "snapshot": str(args.snapshot.resolve()), "audit_dir": str(args.audit_dir.resolve()),
                "protocol_path": str(args.protocol.resolve()), "implementation_resolutions": RESOLUTIONS,
                "sources": sources, "artifacts": artifacts,
                "versions": {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "scikit-learn", "threadpoolctl")},
                "python": platform.python_version(), "models": list(MODELS), "primary_model": MODELS[0],
                "target_model_calls": 0, "upstream_code_executed": False, "synthetic_text_records": 0,
                "score_signs": {"LOSS": "mean_log_likelihood_higher_is_member", "MinK++": "unchanged_higher_is_member", "MinK": "exploratory_unchanged_higher_is_member"},
                "component_count": len(data.components), "raw_records": len(data.records),
                "source_identity_inventory": data.source_identity_inventory,
                "base_exclusion_counts": dict(Counter(reason for c in data.components for reason in c["base_exclusions"])),
                "test_components_touching_dev": sum(c["touches_dev"] and any(data.by_id[r]["split"] == "test" for r in c["row_ids"]) for c in data.components),
                "case_templates": dict(Counter(f"{c['bank']}:{c['condition']}:{c['construction_status']}" for c in cases + sensitivity)),
                "origin_threshold": calibration["threshold"], "origin_distinct_dev_blocks": calibration["distinct_blocks"],
                "preprocessing_and_dev_fit_seconds": time.perf_counter() - started,
                "policy_min_calibration_examples": 4, "missing_effective_scoring_identity": True,
                "outcomes_computed": False}
    write_json(args.output / "sealed_manifest.json", manifest)
    write_json(args.output / "seal_receipt.json", {"sealed_manifest_sha256": hash_file(args.output / "sealed_manifest.json"),
                                                  "outcomes_computed": False})
    print(json.dumps({k: manifest[k] for k in ("status", "component_count", "raw_records", "base_exclusion_counts",
                                              "test_components_touching_dev", "case_templates", "origin_threshold",
                                              "origin_distinct_dev_blocks", "preprocessing_and_dev_fit_seconds")}, indent=2), flush=True)


def common_quality(case, data, model):
    problems = []
    if case["construction_status"] != "available":
        problems.append(case["construction_status"])
    if len(case["suspect"]) != 8 or len(case["reference"]) != 8:
        problems.append("not_eight_records_per_side")
    ids = case["suspect"] + case["reference"]
    if len({data.by_id[r]["component_id"] for r in ids}) != len(ids):
        problems.append("reused_component_within_case")
    for rid in ids:
        row = data.by_id[rid]
        if row["split"] != "test":
            problems.append("non_test_record")
        for method in ("LOSS", "MinK++"):
            score = row["scores"][model].get(method)
            if score is None or score["score"] is None or not math.isfinite(score["score"]):
                problems.append(f"missing_finite_{method}:{rid}")
            elif type(score["token_count"]) is not int or score["token_count"] <= 0:
                problems.append(f"invalid_token_count_{method}:{rid}")
    return {"status": "complete" if not problems else "unavailable", "reasons": sorted(set(problems))}


def endpoint(positive, negative, *, diagnostic_slot="minkpp"):
    test = mann_whitney_u_test(positive, negative, alternative="greater")
    adjusted = combine_diagnostic_pvalues({diagnostic_slot: float(test["p_value"]), "guided": None, "order": None})
    direct = auc(positive, negative)
    if abs(test["auc"] - direct) > 1e-12:
        raise ValueError("Frozen statistical AUC disagrees with direct pair counting")
    return {"raw_auc": direct, "unadjusted_p": float(test["p_value"]),
            "adjusted_p": adjusted["p_value"], "mann_whitney_method": test["method"],
            "mann_whitney_label_assignments": test["label_assignments"],
            "family_size": adjusted["family_size"], "missing_components_p": 1.0,
            "ungated_diagnostic_rejection": adjusted["p_value"] < .05,
            "interpretation": "Conditional score-distribution diagnostic, not calibrated contamination evidence"}


def analyze_case(case, data, model, origin_result, threshold, fingerprint):
    result = {**case, "model": model, "common_quality": common_quality(case, data, model),
              "missing_checks": ["effective_scoring_token_identity_unavailable"],
              "origin_mismatch": origin_result,
              "diagnostics": {}, "gates": {}, "checks": {}, "vectors": []}
    gate_names = ("no_gate", "length_only", "origin_threshold", "full_policy", "ablate_numeric_shift",
                  "ablate_length_shift", "ablate_heterogeneous_control")
    if result["common_quality"]["status"] != "complete":
        result["gates"] = {g: {"action": "unavailable", "reported_adjusted_p": None,
                                "diagnostic_rejection": False, "reason": "common_quality_unavailable"} for g in gate_names}
        return result
    suspect = [data.by_id[r] for r in case["suspect"]]
    reference = [data.by_id[r] for r in case["reference"]]
    s_lengths = [r["scores"][model]["MinK++"]["token_count"] for r in suspect]
    r_lengths = [r["scores"][model]["MinK++"]["token_count"] for r in reference]
    shift = domain_shift_check(s_lengths, r_lengths)
    null = control_null_check([r["scores"][model]["MinK++"]["score"] for r in reference], min_examples=4)
    validity = likelihood_comparison_validity(
        [r["text"] for r in suspect], [r["text"] for r in reference],
        benchmark_scored=8, control_scored=8, min_examples=4,
        domain_shift=shift, fingerprint=fingerprint, control_null=null,
        benchmark_scoring_ids=None, control_scoring_ids=None,
    )
    if fingerprint is None or fingerprint.get("digit_mann_whitney_p_value") is None:
        result["missing_checks"].append("per_text_numeric_fingerprint_insufficient_finite_numeric_texts")
    if null is None:
        result["missing_checks"].append("control_self_check_unavailable")
    result["checks"] = {"policy": validity, "length_shift": shift, "numeric_fingerprint": fingerprint,
                         "control_self_check": null, "scored_input_identity": "unavailable"}
    for method in ("MinK++", "LOSS"):
        result["diagnostics"][method] = endpoint([r["scores"][model][method]["score"] for r in suspect],
                                                  [r["scores"][model][method]["score"] for r in reference])
    # Entire exploratory method list: MinK AUC only; never selected/combined.
    exploratory = [r["scores"][model].get("MinK") for r in suspect + reference]
    if all(v is not None and v["score"] is not None and math.isfinite(v["score"]) for v in exploratory):
        result["diagnostics"]["MinK_exploratory"] = {"raw_auc": auc(
            [v["score"] for v in exploratory[:8]], [v["score"] for v in exploratory[8:]]),
            "status": "descriptive_only_no_test_or_selection"}
    else:
        result["diagnostics"]["MinK_exploratory"] = {"raw_auc": None, "status": "unavailable_exploratory_score"}
    ratio = shift["benchmark_median_tokens"] / max(shift["control_median_tokens"], 1.0)
    result["absolute_log_median_length_ratio"] = abs(math.log(ratio))
    reason_codes = {r["code"] for r in validity["reasons"]}
    decisions = {"no_gate": True, "length_only": "length_shift" not in reason_codes,
                 "full_policy": validity["eligible"],
                 "origin_threshold": None if threshold is None or origin_result["statistic"] is None else
                 origin_result["statistic"] <= threshold}
    for code in ("numeric_shift", "length_shift", "heterogeneous_control"):
        decisions["ablate_" + code] = not (reason_codes - {code})
    p = result["diagnostics"]["MinK++"]["adjusted_p"]
    for gate, accepts in decisions.items():
        result["gates"][gate] = {"action": "unavailable" if accepts is None else "accepted" if accepts else "abstained",
                                "reported_adjusted_p": p if accepts else None,
                                "diagnostic_rejection": bool(accepts and p < .05),
                                "reason": "origin_uncalibrated_or_fit_failed" if accepts is None else None}
    result["stored_scored_target_footprint"] = {"suspect": sum(s_lengths), "reference": sum(r_lengths),
                                               "total_within_case": sum(s_lengths) + sum(r_lengths),
                                               "target_model_calls_for_this_study": 0}
    result["individual_screen"] = {"q": .1, "control_count": 8, "minimum_conformal_p": 1 / 9,
                                    "flags": 0, "reason": "structurally_zero_power_at_this_control_count_not_empirical_safety"}
    for side, rows in (("suspect", suspect), ("reference", reference)):
        for row in rows:
            result["vectors"].append({"side": side, "row_id": row["row_id"], "component_id": row["component_id"],
                                      "domain": row["domain"], "subset": row["subset"], "official_label": row["label"],
                                      "raw_text_sha256": row["raw_text_sha256"], "score_text_hash": row["text_hash"],
                                      "scores": row["scores"][model]})
    return result


def blind_evaluate(data, model):
    results = []
    all_rows = []
    for condition in ("matched", "shifted"):
        rows = eligible_reps(data, split="test", label=0, condition="shared") + eligible_reps(data, split="test", label=1, condition=condition)
        values = blind_decision(model, [data.by_id[r["row_id"]]["text"] for r in rows]).tolist()
        observations = [{"condition": condition, "row_id": r["row_id"], "component_id": r["component_id"],
                         "domain": r["domain"], "label": r["label"], "decision_score": value}
                        for r, value in zip(rows, values)]
        all_rows.extend(observations)
        for domain in ("ALL", *DOMAINS):
            selected = [r for r in observations if domain == "ALL" or r["domain"] == domain]
            positive = [r["decision_score"] for r in selected if r["label"]]
            negative = [r["decision_score"] for r in selected if not r["label"]]
            results.append({"condition": condition, "domain": domain, "member_records": len(positive),
                            "nonmember_records": len(negative), "auc": auc(positive, negative),
                            "interpretation": "Dev-supervised text-only baseline; same shared nonmembers across member conditions; no target-model inference"})
    return results, all_rows


def proportion(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize(cases):
    grouped = defaultdict(list)
    for case in cases:
        key = (case["model"], case["bank"], case["condition"])
        grouped[key + (case["domain"],)].append(case)
        grouped[key + ("ALL",)].append(case)
    summaries, paired = [], []
    for (model, bank, condition, domain), rows in sorted(grouped.items()):
        complete = [r for r in rows if r["common_quality"]["status"] == "complete"]
        n = len(complete)
        ungated = sum(r["diagnostics"]["MinK++"]["ungated_diagnostic_rejection"] for r in complete)
        full_ids = {r["case_id"] for r in complete if r["gates"]["full_policy"]["action"] == "accepted"}
        k = len(full_ids)
        header = {"model": model, "bank": bank, "condition": condition, "domain": domain,
                  "attempted_templates": len(rows), "formed_cases": sum(r["construction_status"] == "available" for r in rows),
                  "common_quality_complete": n, "common_quality_unavailable": len(rows) - n,
                  "sampling_inference": "finite_bank_descriptive_no_binomial_or_independence_assumption"}
        for gate in rows[0]["gates"]:
            accepted = [r for r in rows if r["gates"][gate]["action"] == "accepted"]
            flagged = [r for r in accepted if r["gates"][gate]["diagnostic_rejection"]]
            summaries.append({**header, "gate": gate, "accepted": len(accepted), "diagnostic_rejections": len(flagged),
                              "abstained": sum(r["gates"][gate]["action"] == "abstained" for r in rows),
                              "gate_unavailable": sum(r["gates"][gate]["action"] == "unavailable" for r in rows),
                              "coverage_per_attempted": proportion(len(accepted), len(rows)),
                              "coverage_per_complete": proportion(len(accepted), n),
                              "rejections_per_attempted": proportion(len(flagged), len(rows)),
                              "rejections_per_accepted": proportion(len(flagged), len(accepted)),
                              "accepted_case_ids": [r["case_id"] for r in accepted],
                              "rejected_case_ids": [r["case_id"] for r in flagged]})
        loss_flags = [r for r in complete if r["diagnostics"]["LOSS"]["ungated_diagnostic_rejection"]]
        summaries.append({**header, "gate": "fixed_LOSS_ungated_separate_comparator", "accepted": n,
                          "diagnostic_rejections": len(loss_flags), "abstained": 0, "gate_unavailable": len(rows) - n,
                          "coverage_per_attempted": proportion(n, len(rows)), "coverage_per_complete": 1.0 if n else None,
                          "rejections_per_attempted": proportion(len(loss_flags), len(rows)),
                          "rejections_per_accepted": proportion(len(loss_flags), n),
                          "rejected_case_ids": [r["case_id"] for r in loss_flags]})
        for name, value in (("origin", lambda r: r["origin_mismatch"]["statistic"]),
                            ("length", lambda r: r.get("absolute_log_median_length_ratio"))):
            supported = all(value(r) is not None for r in complete)
            chosen = sorted(complete, key=lambda r: (value(r), r["case_hash"]))[:k] if supported else []
            chosen_ids = {r["case_id"] for r in chosen}
            flags = [r for r in chosen if r["diagnostics"]["MinK++"]["ungated_diagnostic_rejection"]]
            summaries.append({**header, "gate": f"transductive_{name}_matched_coverage",
                              "status": "descriptive" if supported else "unavailable_ranking_statistic",
                              "accepted": len(chosen) if supported else None,
                              "diagnostic_rejections": len(flags) if supported else None,
                              "coverage_per_complete": proportion(k, n) if supported else None,
                              "rejections_per_attempted": proportion(len(flags), len(rows)) if supported else None,
                              "rejections_per_accepted": proportion(len(flags), k) if supported else None,
                              "accepted_case_ids": sorted(chosen_ids), "rejected_case_ids": [r["case_id"] for r in flags]})
            full_flags = {r["case_id"] for r in complete if r["gates"]["full_policy"]["diagnostic_rejection"]}
            paired.append({**header, "comparison": "full_policy_minus_transductive_" + name,
                           "accepted_count_difference": k - len(chosen) if supported else None,
                           "diagnostic_rejection_difference": len(full_flags) - len(flags) if supported else None,
                           "full_only_accepted": sorted(full_ids - chosen_ids), "comparator_only_accepted": sorted(chosen_ids - full_ids)})
        pmf = None
        if n and k:
            pmf = [{"retained_rejections": x, "probability": math.comb(ungated, x) * math.comb(n - ungated, k - x) / math.comb(n, k)}
                   for x in range(max(0, k - (n - ungated)), min(k, ungated) + 1)]
        summaries.append({**header, "gate": "analytical_random_matched_coverage", "accepted": k if n and k else None,
                          "expected_diagnostic_rejections": k * ungated / n if n and k else None,
                          "ungated_diagnostic_rejections": ungated,
                          "hypergeometric_distribution": pmf,
                          "status": "analytical_uniform_selection" if n and k else "unavailable_zero_complete_or_accepted_count",
                          "interpretation": "Expectation over selecting k of N fixed complete cases; not new evidence or an independence claim"})
        for gate in ("no_gate", "length_only", "origin_threshold"):
            full_flags = {r["case_id"] for r in complete if r["gates"]["full_policy"]["diagnostic_rejection"]}
            other_flags = {r["case_id"] for r in complete if r["gates"][gate]["diagnostic_rejection"]}
            other_accepted = {r["case_id"] for r in complete if r["gates"][gate]["action"] == "accepted"}
            paired.append({**header, "comparison": "full_policy_minus_" + gate,
                           "accepted_count_difference": k - len(other_accepted),
                           "diagnostic_rejection_difference": len(full_flags) - len(other_flags),
                           "full_only_rejections": sorted(full_flags - other_flags),
                           "comparator_only_rejections": sorted(other_flags - full_flags),
                           "full_only_accepted": sorted(full_ids - other_accepted),
                           "comparator_only_accepted": sorted(other_accepted - full_ids)})
    return summaries, paired


def evaluate(args):
    started = time.perf_counter()
    manifest_path = args.output / "sealed_manifest.json"
    manifest = read_json(manifest_path)
    seal_hash = read_json(args.output / "seal_receipt.json")["sealed_manifest_sha256"]
    if hash_file(manifest_path) != seal_hash:
        raise ValueError("Sealed manifest changed")
    if (args.output / "evaluation.json").exists() or (args.output / "case_results.jsonl").exists():
        raise FileExistsError("An evaluation exists; heldout results cannot be overwritten")
    verify_sources(manifest["sources"])
    for relative, expected in manifest["artifacts"].items():
        if hash_file(args.output / relative) != expected:
            raise ValueError(f"Sealed artifact changed: {relative}")
    data = collect_data(Path(manifest["snapshot"]), Path(manifest["audit_dir"]))
    cases = read_jsonl(args.output / "case_assignments.jsonl")
    calibration = read_json(args.output / "origin_calibration.json")
    with gzip.open(args.output / "blind_dev_model.json.gz", "rt") as stream:
        blind_model = json.load(stream)
    origin = Origin(data)
    results = []
    per_case = {}
    execution = {"status": "running", "sealed_manifest_sha256": seal_hash, "started_utc": datetime.now(timezone.utc).isoformat(),
                 "target_model_calls": 0, "upstream_code_executed": False, "outcomes_previously_computed": False}
    write_json(args.output / "evaluation.json", execution)
    try:
        # Origin gate and text fingerprint are model-independent; fit once per
        # sealed case, then use the exact same diagnostics for all four sizes.
        for index, case in enumerate(cases):
            ids = case["suspect"] + case["reference"]
            if case["construction_status"] == "available":
                origin_result = origin.score(case["suspect"], case["reference"])
                fingerprint = domain_fingerprint([data.by_id[r]["text"] for r in case["suspect"]],
                                                 [data.by_id[r]["text"] for r in case["reference"]])
            else:
                origin_result = {"status": "unavailable_case_construction", "statistic": None, "folds": []}
                fingerprint = None
            per_case[case["case_id"]] = {"origin": origin_result, "fingerprint": fingerprint}
            for model in MODELS:
                result = analyze_case(case, data, model, origin_result, calibration["threshold"], fingerprint)
                results.append(result)
            print(f"Evaluated fixed case {index + 1}/{len(cases)}: {case['case_id']}", flush=True)
        summaries, paired = summarize(results)
        blind_results, blind_rows = blind_evaluate(data, blind_model)
        verify_sources(manifest["sources"])
        if hash_file(manifest_path) != seal_hash:
            raise ValueError("Seal changed during evaluation")
        write_jsonl(args.output / "case_results.jsonl", results)
        write_json(args.output / "summary_counts.json", summaries)
        write_json(args.output / "paired_differences.json", paired)
        write_json(args.output / "blind_baseline_results.json", blind_results)
        write_jsonl(args.output / "blind_baseline_predictions.jsonl", blind_rows)
        write_json(args.output / "origin_case_fits.json", per_case)
        unique = defaultdict(set)
        footprint = {}
        for result in results:
            if result["bank"] == "primary" and result["common_quality"]["status"] == "complete":
                for vector in result["vectors"]:
                    unique[result["model"]].add(vector["row_id"])
        for model, identifiers in unique.items():
            footprint[model] = {"unique_real_records_in_primary_cases": len(identifiers),
                                "stored_scored_targets_unique_records": sum(data.by_id[r]["scores"][model]["MinK++"]["token_count"] for r in identifiers),
                                "target_model_calls_for_study": 0}
        execution.update(status="complete", outcomes_computed=True, wall_seconds=time.perf_counter() - started,
                         case_templates=len(cases), model_case_views=len(results),
                         origin_test_fit_counts=dict(origin.fit_counts), blind_dev_classifier_fits=1,
                         source_hashes_unchanged_after_evaluation=True, stored_target_footprint=footprint,
                         caveats=["Finite heterogeneous bank; no population-FPR confidence claim",
                                  "Other sizes, conditions, mismatch stress and provenance sensitivities are dependent views",
                                  "Origin threshold is dev-fitted heuristic; partition views are not independent blocks",
                                  "Effective scored-token identity absent; this validates only observable saved-vector policy variant",
                                  "Model-free blind baseline is supervised on dev membership labels",
                                  "Retrospective public benchmark after source inspection, not blinded preregistration"],
                         individual_screen={"q": .1, "controls": 8, "minimum_p": 1 / 9, "flags_structurally_zero": True})
    except BaseException as exc:
        execution.update(status="failed", wall_seconds=time.perf_counter() - started,
                         error_type=type(exc).__name__, error=str(exc), model_case_views_completed=len(results))
        if results:
            write_jsonl(args.output / "partial_case_results.jsonl", results)
        raise
    finally:
        write_json(args.output / "evaluation.json", execution)
    hashes = {str(p.relative_to(args.output)): hash_file(p) for p in sorted(args.output.rglob("*")) if p.is_file()}
    write_json(args.output / "final_artifact_hashes.json", hashes)
    print(json.dumps({k: execution[k] for k in ("status", "wall_seconds", "case_templates", "model_case_views",
                                               "origin_test_fit_counts", "stored_target_footprint")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("seal", "evaluate"))
    parser.add_argument("--snapshot", type=Path, default=PROJECT / ".cache/research/olmo_detect/OLMo-Detect-6eb88de34a09dea85c4b41deff4048e5a203e1b0")
    parser.add_argument("--audit-dir", type=Path, default=PROJECT / "experiments/results/olmo-release-audit/reconstruction-2026-11-07-v2")
    parser.add_argument("--protocol", type=Path, default=PROJECT / "experiments/protocols/real-data-selectivity-v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.phase == "seal":
            seal(args)
        else:
            evaluate(args)


if __name__ == "__main__":
    main()
