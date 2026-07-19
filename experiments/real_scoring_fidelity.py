"""Run a fixed, real OLMo development slice for scoring fidelity and timing.

This is not a membership-performance evaluation. Selection is independent of
model scores: the first two dev records in each of five leaves and two arms.
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
from experiments.real_score_engine import (  # noqa: E402
    load_local_checkpoint, prepare_olmo_plaintext, score_prepared,
)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--audit-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a fresh directory.")
    benchmark = [json.loads(line) for line in args.audit_rows.read_text().splitlines() if line.strip()]
    selected = [row for row in benchmark if row["split"] == "dev"
                and (row["condition"] == "matched" or row["label"] == 0)
                and row["benchmark_row_index"] < 2]
    selected.sort(key=lambda row: (row["domain"], row["subset"], row["label"], row["benchmark_row_index"]))
    assert len(selected) == 20, f"Expected 20 fixed real records, got {len(selected)}."
    records = []
    for row in selected:
        path = args.snapshot / row["benchmark_path"]
        raw = path.read_bytes()
        source = json.loads(raw.decode().splitlines()[row["benchmark_row_index"]])
        text = source["text"].strip()
        assert hashlib.sha256(text.encode()).hexdigest() == row["text_hash"]
        records.append({**row, "text": text, "benchmark_file_sha256": hashlib.sha256(raw).hexdigest(),
                        "id": f"{row['domain']}/{row['subset']}/{row['condition']}/{row['benchmark_row_index']}"})
    assert len({row["text_hash"] for row in records}) == len(records)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "inputs.jsonl").write_bytes(b"".join(encoded(row) for row in records))
    receipt = PROJECT / ".cache/research/real_study_download_receipts.json"
    shutil.copy2(receipt, args.output / "download_receipts.json")
    for name in ("real_score_engine.py", "real_scoring_fidelity.py"):
        shutil.copy2(PROJECT / "experiments" / name, args.output / name)
    manifest = {
        "scope": "Fixed real dev slice for implementation fidelity, precision diagnostics and measured throughput; no membership-performance claim.",
        "selection": "First two dev rows per five source leaves and matched-member/nonmember arms; no outcome-dependent selection.",
        "input_records": len(records), "model_path": str(args.model_path.resolve()),
        "device": "mps", "model_dtype": "bfloat16", "attention_implementation": "sdpa",
        "max_tokens": 4096, "batch_size": 1, "chunk_tokens": 64, "k_percent": 20.,
        "cpu_float64_diagnostic": "First selected record only, same actual forward logits transferred to CPU.",
        "audit_rows_sha256": hashlib.sha256(args.audit_rows.read_bytes()).hexdigest(),
        "inputs_sha256": hashlib.sha256((args.output / "inputs.jsonl").read_bytes()).hexdigest(),
        "download_receipts_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
                          for name in ("real_score_engine.py", "real_scoring_fidelity.py")},
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "numpy", "matplotlib", "scikit-learn")},
        "python": sys.version, "platform": platform.platform(), "status": "sealed_before_model_load",
    }
    manifest_path = args.output / "execution.json"
    manifest_path.write_bytes(encoded(manifest))
    started = time.perf_counter()
    count = 0
    try:
        checkpoint = load_local_checkpoint(args.model_path, device="mps", model_dtype="bfloat16")
        manifest["model_load_seconds"] = time.perf_counter() - started
        manifest["checkpoint_identity"] = checkpoint.identity
        manifest["status"] = "running"
        manifest_path.write_bytes(encoded(manifest))
        with gzip.open(args.output / "scores.jsonl.gz", "xb") as handle:
            for index, row in enumerate(records):
                prepared = prepare_olmo_plaintext(checkpoint.tokenizer, row["text"], row["id"], max_tokens=4096)
                result = score_prepared(checkpoint, prepared, chunk_tokens=64, include_cpu_float64=index == 0)
                result["source"] = {key: value for key, value in row.items() if key != "text"}
                handle.write(encoded(result))
                handle.flush()
                count += 1
                print(json.dumps({"scored": count, "total": len(records), "id": row["id"],
                                  "targets": result["target_count"], "seconds": result["timing"]["total_seconds"],
                                  "native_minkpp": result["variants"]["author_native"]["scores"]["minkpp"],
                                  "fp32_centered_minkpp": result["variants"]["fp32_centered"]["scores"]["minkpp"]}), flush=True)
                if result["status"] != "complete":
                    raise RuntimeError(f"Invalid target support for {row['id']}; vectors retained.")
        manifest.update(status="complete", scored_records=count, wall_seconds=time.perf_counter() - started)
    except BaseException as error:
        manifest.update(status="failed", scored_records=count, wall_seconds=time.perf_counter() - started,
                        error_type=type(error).__name__, error=str(error))
        raise
    finally:
        manifest_path.write_bytes(encoded(manifest))


if __name__ == "__main__":
    main()
