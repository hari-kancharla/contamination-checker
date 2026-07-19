"""Audit original CheckMIABench tokens and seal score-independent real splits.

No model is loaded and no upstream program is run. This audits the released
before/after partition, not the original training-stream consumption itself.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pyarrow.parquet as pq


NAMESPACE = "contamcheckr/checkmia-weighted-vs-uniform/20261107/v1"
DATASET_REVISION = "236fd818e7c877518d588eacc44658bdbf6b3a1b"
MODEL_REVISION = "29376c0a610279f1d79507fc887c6170082f8839"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def hash_file(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def source_rows(path):
    index = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=["tokens", "text"]):
        for row in batch.to_pylist():
            yield index, row
            index += 1


def audit_and_seal(dataset, output, protocol, per_arm_test=1000, per_arm_dev=1000):
    if output.exists():
        raise ValueError("Output already exists; use a new versioned directory.")
    assert per_arm_dev > 0 and per_arm_test > 0
    plan = json.loads(protocol.read_text())
    if (plan["split"]["namespace"] != NAMESPACE or plan["split"]["dev_per_arm"] != per_arm_dev
            or plan["split"]["test_per_arm"] != per_arm_test
            or plan["source"]["dataset_revision"] != DATASET_REVISION
            or plan["source"]["target_revision"] != MODEL_REVISION):
        raise ValueError("Protocol namespace, revisions and split sizes must match this sealed audit.")
    receipt = Path(__file__).resolve().parent.parent / ".cache/research/real_study_download_receipts.json"
    receipts = json.loads(receipt.read_text())
    matching = [row for row in receipts if row["repo_type"] == "dataset"
                and row["repo_id"] == plan["source"]["dataset"] and row["revision"] == DATASET_REVISION]
    if len(matching) != 1:
        raise ValueError("Exactly one matching pinned dataset download receipt is required.")
    source_receipt = matching[0]
    paths = {arm: dataset / "data" / f"{arm}-00000-of-00001.parquet" for arm in ("member", "nonmember")}
    verified = {}
    for arm, path in paths.items():
        relative = path.relative_to(dataset).as_posix()
        expected = source_receipt["files"][relative]
        actual = {"bytes": path.stat().st_size, "sha256": hash_file(path)}
        if actual != expected:
            raise ValueError(f"Source bytes do not match the pinned download receipt: {relative}.")
        verified[arm] = actual
    rows = []
    bank_summary = {}
    for arm, path in paths.items():
        counts = Counter()
        for index, row in source_rows(path):
            ids = row["tokens"]
            if len(ids) != 2049 or any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < 50304 for x in ids):
                raise ValueError(f"Invalid original Pythia tokens in {arm}:{index}.")
            text = row["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Invalid source text in {arm}:{index}.")
            array = np.asarray(ids, dtype="<u4")
            rows.append({"id": f"{arm}:{index}", "official_label": int(arm == "member"), "arm": arm,
                         "source_index": index, "token_count": len(ids),
                         "original_token_ids_sha256_uint32_le": digest(array.tobytes()),
                         "model_input_ids_sha256_uint32_le": digest(array[:-1].tobytes()),
                         "raw_text_sha256": digest(text.encode()),
                         "normalized_text_sha256": digest(" ".join(text.casefold().split()).encode())})
            counts.update({"records": 1, "original_tokens": len(ids), "zero_token_ids": ids.count(0),
                           "eos_token_ids": ids.count(0)})
        bank_summary[arm] = {**dict(counts), "parquet_sha256": verified[arm]["sha256"], "parquet_bytes": verified[arm]["bytes"]}
    # Union exact token/input/text duplicates across all records before splitting.
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    collisions = {}
    for key in ("original_token_ids_sha256_uint32_le", "model_input_ids_sha256_uint32_le", "normalized_text_sha256"):
        first = {}
        counts = Counter(row[key] for row in rows)
        collisions[key] = {"repeated_hashes": sum(v > 1 for v in counts.values()),
                           "records_in_repeated_hashes": sum(v for v in counts.values() if v > 1)}
        for index, row in enumerate(rows):
            if row[key] in first:
                union(index, first[row[key]])
            else:
                first[row[key]] = index
    components = defaultdict(list)
    for index, row in enumerate(rows):
        components[find(index)].append(row)
    quarantined = []
    representatives = defaultdict(list)
    repeated_components = []
    for component in components.values():
        component_id = min(row["id"] for row in component)
        arms = {row["arm"] for row in component}
        if len(arms) > 1:
            quarantined.append({"component_id": component_id, "reason": "duplicate_component_crosses_official_labels",
                                "record_ids": sorted(row["id"] for row in component)})
            continue
        representative = min(component, key=lambda row: (row["original_token_ids_sha256_uint32_le"], row["id"]))
        representative = {**representative, "component_id": component_id, "component_size": len(component)}
        representatives[representative["arm"]].append(representative)
        if len(component) > 1:
            repeated_components.append({"component_id": component_id, "retained_id": representative["id"],
                                        "record_ids": sorted(row["id"] for row in component)})
    assignment = []
    for arm in paths:
        ranked = sorted(representatives[arm], key=lambda row: digest(f"{NAMESPACE}|{arm}|{row['component_id']}".encode()))
        if len(ranked) < per_arm_dev + per_arm_test:
            raise ValueError("Insufficient eligible unique components for declared dev/test sizes.")
        for index, row in enumerate(ranked[:per_arm_dev + per_arm_test]):
            assignment.append({**row, "study_split": "dev" if index < per_arm_dev else "test",
                               "selection_rank": index})
    # Interleave arms for timing and memory diagnostics without changing split identity.
    assignment.sort(key=lambda row: (row["study_split"], row["selection_rank"], row["arm"]))
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), output / "checkmia_data_audit.py")
    shutil.copy2(protocol, output / "protocol.json")
    shutil.copy2(receipt, output / "download_receipts.json")
    (output / "record_index.jsonl").write_bytes(b"".join(json_bytes(row) for row in rows))
    (output / "assignment.jsonl").write_bytes(b"".join(json_bytes(row) for row in assignment))
    (output / "quarantine.json").write_bytes(json_bytes({"cross_label": quarantined, "within_label_duplicates": repeated_components}))
    lookup = {(row["arm"], row["source_index"]): row for row in assignment}
    written = Counter()
    selected_payloads = {}
    for arm, path in paths.items():
        for index, record in source_rows(path):
            selected = lookup.get((arm, index))
            if selected is None:
                continue
            assert digest(np.asarray(record["tokens"], dtype="<u4").tobytes()) == selected["original_token_ids_sha256_uint32_le"]
            assert digest(record["text"].encode()) == selected["raw_text_sha256"]
            selected_payloads[(arm, index)] = record
    with gzip.open(output / "inputs.jsonl.gz", "xb") as handle:
        for selected in assignment:
            record = selected_payloads[(selected["arm"], selected["source_index"])]
            handle.write(json_bytes({**selected, "tokens": record["tokens"], "text": record["text"]}))
            written[(selected["arm"], selected["study_split"])] += 1
    for arm, path in paths.items():
        if hash_file(path) != verified[arm]["sha256"]:
            raise ValueError(f"Source changed during audit: {path}.")
    manifest = {
        "status": "sealed_before_model_scores", "namespace": NAMESPACE,
        "dataset": "jeffreygwang/pythia_dedupe_mia_0-97000_97000-98500", "dataset_revision": DATASET_REVISION,
        "planned_target_model": "EleutherAI/pythia-1.4b-deduped", "planned_target_model_revision": MODEL_REVISION,
        "planned_target_checkpoint": "step97000", "source_banks": bank_summary, "duplicate_checks": collisions,
        "source_bytes_verified_against_pinned_download_receipt": True,
        "scoring_order": "Exactly assignment.jsonl order, interleaving arms within dev then test.",
        "total_components": len(components), "quarantined_cross_label_components": len(quarantined),
        "quarantined_cross_label_records": sum(len(row["record_ids"]) for row in quarantined),
        "eligible_component_representatives": {arm: len(value) for arm, value in representatives.items()},
        "assignment_counts": {f"{arm}/{split}": count for (arm, split), count in written.items()},
        "source_dependence_limit": "No document/source identity is released per packed row. Exact full-token/input/normalized-text duplicates are grouped; semantic clones and adjacent chunks may still depend.",
        "membership_limit": "Coarse publisher before/after-checkpoint partition; original training stream not independently replayed. Per-record source_steps bookkeeping is not used.",
        "zero_token_attention": "All original token positions remain context and targets; if token0 occurs, this differs from Pandora default input_ids>0 mask and is explicitly reported.",
        "token_hash_encoding": "SHA256 of original integers encoded as contiguous little-endian uint32, no length prefix; every original row has2049IDs.",
        "files_sha256": {name: hash_file(output / name) for name in
                         ("checkmia_data_audit.py", "protocol.json", "download_receipts.json", "record_index.jsonl",
                          "assignment.jsonl", "quarantine.json", "inputs.jsonl.gz")},
        "python": sys.version, "numpy": np.__version__, "upstream_code_executed": False,
    }
    (output / "seal.json").write_bytes(json_bytes(manifest))
    print(json.dumps({"output": str(output), "source_banks": bank_summary,
                      "quarantined_cross_label_components": len(quarantined), "counts": manifest["assignment_counts"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-arm-test", type=int, default=1000)
    parser.add_argument("--per-arm-dev", type=int, default=1000)
    args = parser.parse_args()
    audit_and_seal(args.dataset, args.output, args.protocol, args.per_arm_test, args.per_arm_dev)


if __name__ == "__main__":
    main()
