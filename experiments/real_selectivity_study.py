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
