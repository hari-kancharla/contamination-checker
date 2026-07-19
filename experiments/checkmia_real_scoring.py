"""Execute the sealed real CheckMIABench normalization-definition study.

Full token evidence is sharded separately from compact score records. No
upstream code, remote calls, fitting, or threshold selection occurs here.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time


PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from experiments.real_score_engine import load_local_checkpoint, prepare_checkmiabench_tokens, score_prepared  # noqa: E402


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def sha_file(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-data", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.vectors.exists():
        parser.error("Output/vector directories must be new; existing runs are preserved.")
    seal_path = args.sealed_data / "seal.json"
    seal = json.loads(seal_path.read_text())
    for name, expected in seal["files_sha256"].items():
        if sha_file(args.sealed_data / name) != expected:
            raise ValueError(f"Sealed data changed: {name}.")
    protocol = json.loads((args.sealed_data / "protocol.json").read_text())
    if protocol["protocol_id"] != "checkmia-weighted-vs-uniform-v1":
        raise ValueError("Unknown protocol.")
    if protocol["scoring"]["model_dtype"] != "float16" or protocol["scoring"]["k_percent"] != 10:
        raise ValueError("This runner implements only the frozen FP16/k10 study.")
    amendment = json.loads(args.amendment.read_text())
    if amendment["amendment_id"] != "checkmia-weighted-vs-uniform-v1-amendment-1":
        raise ValueError("Unknown operational amendment.")
    prior_path = Path(amendment["pretest_evidence"]["failed_execution"])
    if sha_file(prior_path) != amendment["pretest_evidence"]["failed_execution_sha256"]:
        raise ValueError("Prior failure evidence changed.")
    prior = json.loads(prior_path.read_text())
    prior_rows = [json.loads(line) for line in (prior_path.parent / "scores.jsonl").read_text().splitlines()]
    if prior["status"] != "failed" or len(prior_rows) != 1 or any(row["source"]["study_split"] != "dev" for row in prior_rows):
        raise ValueError("Amendment must precede every heldout model score.")
    receipts = json.loads((args.sealed_data / "download_receipts.json").read_text())
    matches = [r for r in receipts if r["repo_type"] == "model" and r["repo_id"] == protocol["source"]["target"]
               and r["revision"] == protocol["source"]["target_revision"]]
    if len(matches) != 1:
        raise ValueError("Matching pinned checkpoint receipt is missing or ambiguous.")
    receipt = matches[0]
    model_verified = {}
    for name, expected in receipt["files"].items():
        path = args.model_path / name
        actual = {"bytes": path.stat().st_size, "sha256": sha_file(path)}
        if actual != expected:
            raise ValueError(f"Checkpoint file does not match pinned receipt: {name}.")
        model_verified[name] = actual
    with gzip.open(args.sealed_data / "inputs.jsonl.gz", "rt") as handle:
        original = [json.loads(line) for line in handle]
    records = [row for row in original if row["study_split"] == "test" or row["official_label"] == 0]
    assert len(records) == 3000
    assert sum(row["study_split"] == "dev" for row in records) == 1000
    assert all(row["official_label"] == 0 for row in records if row["study_split"] == "dev")
    assert len({row["component_id"] for row in records}) == len(records)
    args.output.mkdir(parents=True, exist_ok=False)
    args.vectors.mkdir(parents=True, exist_ok=False)
    frozen_sources = ("checkmia_real_scoring.py", "real_score_engine.py", "checkmia_real_analysis.py")
    for name in frozen_sources:
        shutil.copy2(PROJECT / "experiments" / name, args.output / name)
    shutil.copy2(args.sealed_data / "protocol.json", args.output / "protocol.json")
    shutil.copy2(args.amendment, args.output / "amendment.json")
    manifest = {
        "status": "sealed_before_model_load", "protocol_id": protocol["protocol_id"],
        "seal_path": str(seal_path.resolve()), "seal_sha256": sha_file(seal_path),
        "protocol_sha256": sha_file(args.output / "protocol.json"),
        "amendment_id": amendment["amendment_id"], "amendment_sha256": sha_file(args.output / "amendment.json"),
        "source_sha256": {name: sha_file(args.output / name) for name in frozen_sources},
        "failure_accounting": amendment["failure_accounting"],
        "model_files_verified": model_verified, "model_repo": receipt["repo_id"], "model_revision": receipt["revision"],
        "settings": protocol["scoring"], "planned_scored_records": len(records),
        "vector_directory": str(args.vectors.resolve()), "vector_records_per_shard": 100, "gzip_compresslevel": 1,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "numpy", "matplotlib", "scikit-learn", "pyarrow")},
        "python": sys.version, "platform": platform.platform(),
        "scope": "New fixed-data CheckMIABench weighted-vs-uniform and precision comparison, not exact historical table replication or a causal effect study.",
    }
    execution = args.output / "execution.json"
    execution.write_bytes(encoded(manifest))
    started = time.perf_counter()
    count = 0
    shard_handle = None
    shard_path = None
    shard_index = -1
    vector_shards = []
    optional_failures = []

    def close_shard():
        nonlocal shard_handle
        if shard_handle is not None:
            shard_handle.close()
            shard_handle = None
            vector_shards.append({"path": str(shard_path.resolve()), "bytes": shard_path.stat().st_size,
                                  "sha256": sha_file(shard_path)})

    try:
        checkpoint = load_local_checkpoint(args.model_path, device="mps", model_dtype="float16", attention_implementation="sdpa")
        manifest["model_load_seconds"] = time.perf_counter() - started
        manifest["checkpoint_identity"] = checkpoint.identity
        manifest["status"] = "running"
        execution.write_bytes(encoded(manifest))
        with (args.output / "scores.jsonl").open("xb") as scalar_handle:
            for index, row in enumerate(records):
                if index // 100 != shard_index:
                    close_shard()
                    shard_index = index // 100
                    shard_path = args.vectors / f"tokens-{shard_index:04d}.jsonl.gz"
                    shard_handle = gzip.open(shard_path, "xb", compresslevel=1)
                prepared = prepare_checkmiabench_tokens(row["tokens"], row["id"], text=row["text"])
                assert len(prepared.input_ids) == 2048 and len(prepared.target_ids) == 2048
                result = score_prepared(checkpoint, prepared, chunk_tokens=64, k_percent=10.,
                                        include_pandora=True, pandora_k_percent=10., include_cpu_float64=index < 2)
                assert result["input_token_ids"] == row["tokens"][:-1]
                assert result["target_token_ids"] == row["tokens"][1:]
                assert result["original_token_ids"] == row["tokens"]
                source = {key: value for key, value in row.items() if key not in {"tokens", "text"}}
                result["source"] = source
                required = amendment["failure_accounting"]
                primary_complete = all(result["variants"][name]["status"] == "complete"
                                       for name in required["required_complete_variants"])
                primary_complete = primary_complete and all(result["pandora_comparators"][name]["status"] == "complete"
                                       for name in required["required_complete_pandora_comparators"])
                if index < 2:
                    primary_complete = primary_complete and result["variants"]["cpu_float64_centered"]["status"] == "complete"
                failed_optional = ([f"variants/{name}" for name, value in result["variants"].items()
                                    if value["status"] != "complete" and name not in required["required_complete_variants"]]
                                   + [f"pandora_comparators/{name}" for name, value in result["pandora_comparators"].items()
                                      if value["status"] != "complete" and name not in required["required_complete_pandora_comparators"]])
                result["runner_eligibility"] = {"primary_complete": primary_complete, "optional_failed_sections": failed_optional,
                                                 "amendment_id": amendment["amendment_id"]}
                if failed_optional:
                    optional_failures.append({"id": row["id"], "split": row["study_split"], "official_label": row["official_label"],
                                              "failed_sections": failed_optional})
                raw = encoded(result)
                shard_handle.write(raw)
                shard_handle.flush()
                scalar = {key: value for key, value in result.items() if key not in
                          {"original_token_ids", "input_token_ids", "target_token_ids", "target_input_positions", "target_sequence_positions", "logit_positions", "variants", "pandora_comparators"}}
                scalar["variants"] = {name: {key: value for key, value in value.items() if key not in
                                            {"per_token", "mink_selected_indices", "minkpp_selected_indices"}}
                                      for name, value in result["variants"].items()}
                scalar["pandora_comparators"] = {name: {key: value for key, value in value.items()
                                                       if key not in {"per_token", "selected_indices"}}
                                                 for name, value in result["pandora_comparators"].items()}
                scalar["full_vector_record"] = {"path": str(shard_path.resolve()), "row_index": index % 100,
                                                "uncompressed_record_sha256": hashlib.sha256(raw).hexdigest()}
                scalar_handle.write(encoded(scalar))
                scalar_handle.flush()
                count += 1
                if not primary_complete:
                    raise ValueError(f"Incomplete primary numerical support for {row['id']}; full vectors retained.")
                if count <= 2 or count % 25 == 0:
                    manifest.update(scored_records=count, elapsed_seconds=time.perf_counter() - started,
                                    completed_vector_shards=vector_shards)
                    execution.write_bytes(encoded(manifest))
                    print(json.dumps({"scored": count, "planned": len(records), "current_split": row["study_split"],
                                      "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
        close_shard()
        for name, expected in manifest["source_sha256"].items():
            if sha_file(PROJECT / "experiments" / name) != expected:
                raise ValueError(f"Source changed during run: {name}; retain frozen source and diagnose before interpretation.")
        if sha_file(seal_path) != manifest["seal_sha256"]:
            raise ValueError("Data seal changed during scoring.")
        manifest.update(status="complete", scored_records=count, wall_seconds=time.perf_counter() - started)
    except BaseException as error:
        manifest.update(status="failed", scored_records=count, wall_seconds=time.perf_counter() - started,
                        error_type=type(error).__name__, error=str(error))
        raise
    finally:
        close_shard()
        manifest["vector_shards"] = vector_shards
        manifest["optional_failure_records"] = optional_failures
        manifest["optional_failure_record_count"] = len(optional_failures)
        if (args.output / "scores.jsonl").exists():
            manifest["scores_sha256"] = sha_file(args.output / "scores.jsonl")
            manifest["scores_bytes"] = (args.output / "scores.jsonl").stat().st_size
        execution.write_bytes(encoded(manifest))


if __name__ == "__main__":
    main()
