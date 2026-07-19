#!/usr/bin/env python3
"""Independently verify a COMPLETED amended CheckMIABench run's support/lineage.

Standard library only. Does not import the scorer, analysis, project, model
libraries, or upstream code. Computes no AUCs, thresholds, or study outcomes.
Optional failures remain failures; they do not disqualify complete FP32 rows.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path


AMENDMENT = "checkmia-weighted-vs-uniform-v1-amendment-1"
REQUIRED = ("fp32_second_moment", "fp32_centered")
TARGETS = 2048


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_hash(values):
    return hashlib.sha256(json.dumps(values, separators=(",", ":"), ensure_ascii=False,
                                       sort_keys=True, allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def verify_required(record, *, with_vectors=False):
    for name in REQUIRED:
        variant = record["variants"][name]
        require(variant["status"] == "complete" and variant["aggregate_valid"], f"{record['id']} incomplete {name}")
        require(variant["supported_target_count"] == TARGETS, f"{record['id']} {name} does not support2048targets")
        require(all(finite(variant["scores"].get(k)) for k in ("loss", "mink", "minkpp", "sum_logp", "zlib")),
                f"{record['id']} missing required {name} aggregate")
        require(variant["selected_token_count"] == 204, f"{record['id']} wrong frozen k selection")
        if with_vectors:
            per_token = variant["per_token"]
            require(len(per_token["support"]) == TARGETS and all(v is True for v in per_token["support"]),
                    f"{record['id']} {name} raw support mask incomplete")
            for field in ("logp", "mu", "variance", "sigma", "z"):
                require(len(per_token[field]) == TARGETS and all(finite(v) for v in per_token[field]),
                        f"{record['id']} {name}/{field} raw vector incomplete")
    uniform = record["pandora_comparators"]["fp32"]
    require(uniform["status"] == "complete" and uniform["supported_target_count"] == TARGETS,
            f"{record['id']} FP32 Pandora does not support2048targets")
    require(uniform["selected_token_count"] == 204, f"{record['id']} wrong Pandora k selection")
    require(all(finite(uniform["scores"].get(k)) for k in ("pandora_minkxx", "pandora_minkxx_feature")),
            f"{record['id']} missing FP32 Pandora aggregate")
    if with_vectors:
        per_token = uniform["per_token"]
        require(len(per_token["support"]) == TARGETS and all(v is True for v in per_token["support"]),
                f"{record['id']} FP32 Pandora raw support incomplete")
        for field in ("logp", "mean", "std", "z"):
            require(len(per_token[field]) == TARGETS and all(finite(v) for v in per_token[field]),
                    f"{record['id']} FP32 Pandora/{field} raw vector incomplete")
    if "cpu_float64_centered" in record["variants"]:
        gold = record["variants"]["cpu_float64_centered"]
        require(gold["status"] == "complete" and gold["aggregate_valid"] and gold["supported_target_count"] == TARGETS,
                f"{record['id']} requested CPU64 diagnostic incomplete")
        require(all(finite(gold["scores"].get(k)) for k in ("loss", "mink", "minkpp", "sum_logp", "zlib")),
                f"{record['id']} requested CPU64 aggregate missing")
        if with_vectors:
            per_token = gold["per_token"]
            require(len(per_token["support"]) == TARGETS and all(v is True for v in per_token["support"]),
                    f"{record['id']} requested CPU64 raw support incomplete")
            for field in ("logp", "mu", "variance", "sigma", "z"):
                require(len(per_token[field]) == TARGETS and all(finite(v) for v in per_token[field]),
                        f"{record['id']} CPU64/{field} raw vector incomplete")


def verify(args):
    execution_path = args.scoring / "execution.json"
    execution_hash = file_hash(execution_path)
    execution = json.loads(execution_path.read_text())
    require(execution["status"] == "complete" and execution["scored_records"] == 3000,
            "Run is not completed3000rows; do not interpret a partial cohort")
    require(execution["amendment_id"] == AMENDMENT, "Wrong amendment")
    require(file_hash(args.scoring / "scores.jsonl") == execution["scores_sha256"], "Scalar bytes changed")
    for filename, key in (("amendment.json", "amendment_sha256"), ("protocol.json", "protocol_sha256")):
        require(file_hash(args.scoring / filename) == execution[key], f"Frozen {filename} changed")
    for name, expected in execution["source_sha256"].items():
        require(file_hash(args.scoring / name) == expected, f"Frozen scoring source changed: {name}")
    require(file_hash(args.sealed_data / "seal.json") == execution["seal_sha256"], "Different data seal")
    seal = json.loads((args.sealed_data / "seal.json").read_text())
    for name, expected in seal["files_sha256"].items():
        require(file_hash(args.sealed_data / name) == expected, f"Sealed input changed: {name}")
    with gzip.open(args.sealed_data / "inputs.jsonl.gz", "rt") as stream:
        original = [json.loads(line) for line in stream]
    expected = [r for r in original if r["study_split"] == "test" or r["official_label"] == 0]
    records = [json.loads(line) for line in (args.scoring / "scores.jsonl").read_text().splitlines()]
    require(len(records) == len(expected) == len({r["id"] for r in records}) == 3000, "Wrong unique full-cohort count")
    require([r["id"] for r in records] == [r["id"] for r in expected], "Scored order/identity differs from sealed assignment")
    require(execution["model_repo"] == seal["planned_target_model"] and execution["model_revision"] == seal["planned_target_model_revision"],
            "Checkpoint identity differs from seal")
    gold_ids = []
    optional = []
    for index, (record, source) in enumerate(zip(records, expected)):
        require(record["source"] == {k: v for k, v in source.items() if k not in ("text", "tokens")},
                f"{record['id']} source metadata differs from sealed record")
        require(record["target_count"] == TARGETS and record["runner_eligibility"]["primary_complete"],
                f"{record['id']} primary eligibility incomplete")
        require(record["input_token_ids_sha256"] == ids_hash(source["tokens"][:-1]), f"{record['id']} wrong input IDs hash")
        require(record["target_token_ids_sha256"] == ids_hash(source["tokens"][1:]), f"{record['id']} wrong target IDs hash")
        require(record["source_text_sha256"] == hashlib.sha256(source["text"].encode()).hexdigest(), f"{record['id']} wrong source text hash")
        require(("cpu_float64_centered" in record["variants"]) == (index < 2), "CPU64 requests differ from frozen first2dev rule")
        if index < 2:
            require(source["study_split"] == "dev" and source["official_label"] == 0, "CPU64 diagnostics are not development nonmembers")
            gold_ids.append(record["id"])
        verify_required(record)
        failures = [f"variants/{n}" for n, v in record["variants"].items() if v["status"] != "complete" and n not in REQUIRED]
        failures += [f"pandora_comparators/{n}" for n, v in record["pandora_comparators"].items() if v["status"] != "complete" and n != "fp32"]
        require(sorted(failures) == sorted(record["runner_eligibility"]["optional_failed_sections"]), f"{record['id']} optional-failure accounting differs")
        if failures:
            optional.append({"id": record["id"], "split": source["study_split"], "official_label": source["official_label"], "failed_sections": failures})
    require(optional == execution["optional_failure_records"] and len(optional) == execution["optional_failure_record_count"],
            "Optional failure manifest differs from raw sections")
    vector_count = 0
    if args.verify_vectors:
        by_id = {r["id"]: r for r in records}
        source_by_id = {r["id"]: r for r in expected}
        seen = set()
        for shard in execution["vector_shards"]:
            path = Path(shard["path"])
            require(path.stat().st_size == shard["bytes"] and file_hash(path) == shard["sha256"], f"Vector shard changed: {path}")
            with gzip.open(path, "rb") as stream:
                for index, line in enumerate(stream):
                    full = json.loads(line)
                    rid = full["id"]
                    require(rid in by_id and rid not in seen, f"Unexpected/repeated full-vector record: {rid}")
                    seen.add(rid)
                    scalar, source = by_id[rid], source_by_id[rid]
                    pointer = scalar["full_vector_record"]
                    require(Path(pointer["path"]) == path and pointer["row_index"] == index and
                            hashlib.sha256(line).hexdigest() == pointer["uncompressed_record_sha256"], f"{rid} vector lineage differs")
                    require(full["input_token_ids"] == source["tokens"][:-1] and full["target_token_ids"] == source["tokens"][1:] and
                            full["original_token_ids"] == source["tokens"], f"{rid} exact original/input/target IDs differ")
                    for key, value in scalar.items():
                        if key == "full_vector_record":
                            continue
                        if key in ("variants", "pandora_comparators"):
                            for variant, fields in value.items():
                                require(all(full[key][variant][field] == observation for field, observation in fields.items()),
                                        f"{rid} compact/full {key}/{variant} differs")
                        else:
                            require(full[key] == value, f"{rid} compact/full metadata {key} differs")
                    verify_required(full, with_vectors=True)
                    vector_count += 1
        require(vector_count == 3000 and seen == set(by_id), "Full vectors do not cover all3000rows")
    require(file_hash(execution_path) == execution_hash, "Execution manifest changed during verification")
    return {"status": "passed", "verifier_sha256": file_hash(Path(__file__)), "execution_sha256": execution_hash,
            "frozen_analysis_sha256": execution["source_sha256"]["checkmia_real_analysis.py"],
            "scalar_records_verified": len(records), "required_scored_targets_per_record": TARGETS,
            "explicit_Pandora_FP32_full2048_support_verified": True,
            "requested_CPU64_full2048_support_verified": gold_ids,
            "optional_failure_records_preserved": len(optional), "optional_failures_by_split": dict(Counter(r["split"] for r in optional)),
            "vector_records_verified": vector_count,
            "full_vectors": "all_hashes_exact_tokens_and_required_raw_masks_checked" if args.verify_vectors else "not_checked_use_verify_vectors_for_full_raw_support_audit",
            "target_model_calls": 0, "study_outcomes_computed": False,
            "limits": "Support/lineage checks only. No AUC, tail outcome, historical-run replication, independent corpus-membership proof, calibration or novelty claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoring", type=Path, required=True)
    parser.add_argument("--sealed-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON result file; never overwrites")
    parser.add_argument("--verify-vectors", action="store_true", help="Also hash/decompress every retained raw-vector shard and inspect exact masks/IDs")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = verify(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
