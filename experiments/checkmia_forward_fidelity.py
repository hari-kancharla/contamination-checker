"""One observed CheckMIA record: CPU-FP16 versus retained MPS score vectors.

Run only after the fixed 3,000-record process has exited. This program requires
its complete manifest, verifies pinned local artifacts, and loads one CPU model.
It never starts an MPS forward, downloads a model, changes hardware settings,
or interprets the result as membership evidence. --help does not load a model.

Example, after the main process has exited:
    .venv/bin/python experiments/checkmia_forward_fidelity.py \
      --study-run experiments/results/checkmia-real-v1/scoring-amendment-1 \
      --sealed-data experiments/results/checkmia-real-v1/sealed-data \
      --model-path /absolute/path/to/the/pinned/pythia/checkpoint \
      --output experiments/results/checkmia-real-v1/cpu-forward-first-dev

FP32 comparisons include model-forward and device-specific score arithmetic.
Both forwards also receive the same pinned CPU-FP64 statistics function, using
the already-retained CPU-FP64 statistics of the MPS logits as the reference.
That comparison holds statistics device/dtype/function fixed and measures
forward-induced differences through these observables, not full-logit equality.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import shutil
import statistics
import struct
import sys
import time
from typing import Any


PROJECT = Path(__file__).resolve().parent.parent
ENGINE_SHA256 = "91cc8640f4fcf0aeb94a1454f7cda27e7d7dcc406f8eada9329b1b742d4ba6ff"
MODEL_REPO = "EleutherAI/pythia-1.4b-deduped"
MODEL_REVISION = "29376c0a610279f1d79507fc887c6170082f8839"
MODEL_WEIGHT_SHA256 = "852d5bc740615a542c02c72a049d369a089e64c0709275498f9c3e0d6d29054a"
DATASET_REPO = "jeffreygwang/pythia_dedupe_mia_0-97000_97000-98500"
DATASET_REVISION = "236fd818e7c877518d588eacc44658bdbf6b3a1b"
RECORD_ID = "nonmember:6834"
REQUIRED_VARIANTS = ("fp32_second_moment", "fp32_centered")
CPU64_VARIANT = "cpu_float64_centered"


def encoded(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def sha_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_model(path: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    require(receipt["files"]["model.safetensors"]["sha256"] == MODEL_WEIGHT_SHA256,
            "Receipt does not identify the pinned study weights.")
    actual = {}
    for name, expected in receipt["files"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts,
                "Receipt must contain checkpoint-relative file paths.")
        source = path / relative
        observed = {"bytes": source.stat().st_size, "sha256": sha_file(source)}
        require(observed == expected, f"Checkpoint bytes differ from pinned receipt: {name}.")
        actual[name] = observed
    return actual


def full_fp32_support(record: dict[str, Any]) -> dict[str, Any]:
    sections = {f"variants/{name}": record["variants"][name] for name in REQUIRED_VARIANTS}
    sections["pandora_comparators/fp32"] = record["pandora_comparators"]["fp32"]
    result = {}
    for name, section in sections.items():
        support = section["per_token"]["support"]
        complete = (section["status"] == "complete" and len(support) == 2048
                    and all(value is True for value in support)
                    and section["supported_target_count"] == 2048)
        result[name] = {"complete": complete, "status": section["status"],
                        "supported_target_count": section["supported_target_count"],
                        "unsupported_positions": [i for i, value in enumerate(support) if value is not True]}
    return result


def full_cpu64_support(record: dict[str, Any]) -> dict[str, Any]:
    section = record["variants"].get(CPU64_VARIANT)
    if section is None:
        return {"complete": False, "status": "not_requested", "supported_target_count": 0,
                "statistics_convention_matches": False, "unsupported_positions": list(range(2048))}
    support = section["per_token"]["support"]
    dtypes = section["component_dtypes"]
    convention = (str(dtypes.get("computation_device", "")).startswith("cpu") and
                  all(dtypes.get(key) == "torch.float64" for key in
                      ("log_softmax", "target_logp", "moment_product", "moment_reduction",
                       "moment_result", "variance", "sigma", "normalized_score")))
    complete = (section["status"] == "complete" and len(support) == 2048
                and all(value is True for value in support)
                and section["supported_target_count"] == 2048 and convention)
    return {"complete": complete, "status": section["status"],
            "supported_target_count": section["supported_target_count"],
            "statistics_convention_matches": convention,
            "unsupported_positions": [i for i, value in enumerate(support) if value is not True]}


def vector_difference(cpu: list[float], mps: list[float]) -> dict[str, Any]:
    require(len(cpu) == len(mps) == 2048, "A required vector does not cover all 2,048 targets.")
    require(all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                for x in cpu + mps), "Required vector has missing/nonfinite values; no subset comparison is allowed.")
    absolute = [abs(float(a) - float(b)) for a, b in zip(cpu, mps)]
    maximum = max(absolute)
    return {"target_count": len(absolute), "max_absolute_difference": maximum,
            "median_absolute_difference": statistics.median(absolute),
            "mean_signed_cpu_minus_mps": math.fsum(float(a) - float(b) for a, b in zip(cpu, mps)) / len(cpu),
            "first_max_difference_aligned_target_index": absolute.index(maximum)}


def compare_records(cpu: dict[str, Any], mps: dict[str, Any]) -> dict[str, Any]:
    require(cpu["original_token_ids"] == mps["original_token_ids"]
            and cpu["input_token_ids"] == mps["input_token_ids"]
            and cpu["target_token_ids"] == mps["target_token_ids"], "Device records have different causal inputs/targets.")
    support = {"cpu": full_fp32_support(cpu), "mps": full_fp32_support(mps)}
    complete_fp32 = all(value["complete"] for device in support.values() for value in device.values())
    for device, record in (("cpu", cpu), ("mps", mps)):
        support[device][f"variants/{CPU64_VARIANT}"] = full_cpu64_support(record)
    complete_cpu64 = all(device[f"variants/{CPU64_VARIANT}"]["complete"] for device in support.values())
    complete = complete_fp32 and complete_cpu64
    output: dict[str, Any] = {
        "record_id": RECORD_ID, "required_support": support,
        "required_fp32_complete": complete_fp32, "required_cpu64_complete": complete_cpu64,
        "required_comparison_complete": complete,
        "interpretation": "One-record device execution comparison; no pass threshold, full-logit parity or membership claim.",
        "difference_scopes": {
            "fp32_sections": "Includes CPU/MPS model-forward differences and their device-specific FP32 scoring kernels.",
            CPU64_VARIANT: "Both statistic arrays use the same hash-pinned CPU-FP64 function/software on their respective FP16 forward logits. Statistics device/dtype/function are held fixed; differences reflect forward outputs through these score observables, not full-logit equality.",
        },
        "all_variant_statuses": {
            device: {"variants": {k: v["status"] for k, v in record["variants"].items()},
                     "pandora_comparators": {k: v["status"] for k, v in record["pandora_comparators"].items()}}
            for device, record in (("cpu", cpu), ("mps", mps))
        },
        "sections": {},
    }
    if not complete:
        output["status"] = "incomplete_required_support_no_difference_statistics"
        return output
    for name in REQUIRED_VARIANTS + (CPU64_VARIANT, "pandora_fp32"):
        if name == "pandora_fp32":
            left, right = cpu["pandora_comparators"]["fp32"], mps["pandora_comparators"]["fp32"]
            components = ("logp", "mean", "std", "z")
            selections = ("selected_indices",)
        else:
            left, right = cpu["variants"][name], mps["variants"][name]
            components = ("logp", "mu", "variance", "sigma", "z")
            selections = ("mink_selected_indices", "minkpp_selected_indices")
        aggregate = {}
        require(set(left["scores"]) == set(right["scores"]), "Device aggregate definitions differ.")
        for metric, a in left["scores"].items():
            b = right["scores"][metric]
            require(a is not None and b is not None and math.isfinite(a) and math.isfinite(b),
                    f"Required aggregate unavailable: {name}/{metric}.")
            aggregate[metric] = {"cpu": a, "mps": b, "cpu_minus_mps": a - b,
                                 "absolute_difference": abs(a - b)}
        tails = {}
        for key in selections:
            a, b = set(left[key]), set(right[key])
            require(len(a) == len(b) == 204, "Unexpected tail support or duplicate selected indices.")
            tails[key] = {"cpu_count": len(a), "mps_count": len(b), "intersection_count": len(a & b),
                          "union_count": len(a | b), "jaccard": len(a & b) / len(a | b)}
        output["sections"][name] = {
            "comparison_scope": output["difference_scopes"][CPU64_VARIANT if name == CPU64_VARIANT else "fp32_sections"],
            "per_token_differences": {key: vector_difference(left["per_token"][key], right["per_token"][key])
                                      for key in components},
            "aggregate_differences": aggregate, "selected_tail_overlap": tails,
        }
    output["status"] = "compared_no_acceptance_threshold"
    return output


def verified_reference(study_run: Path, sealed_data: Path) -> dict[str, Any]:
    """Read only the already-observed first dev score, never interpret test scores."""
    execution = read_json(study_run / "execution.json")
    require(execution["status"] == "complete" and execution["scored_records"] == 3000
            and execution["planned_scored_records"] == 3000,
            "The fixed 3,000-record run must be complete and its process exited before this CPU run.")
    require(execution["model_repo"] == MODEL_REPO and execution["model_revision"] == MODEL_REVISION,
            "Study checkpoint differs from the fixed Pythia checkpoint.")
    require(execution["source_sha256"]["real_score_engine.py"] == ENGINE_SHA256,
            "Study did not use the reviewed stable v3 engine.")
    require(sha_file(study_run / "real_score_engine.py") == ENGINE_SHA256
            and sha_file(PROJECT / "experiments/real_score_engine.py") == ENGINE_SHA256,
            "Frozen/live engine differs from the reviewed v3 bytes.")
    for name, expected in execution["source_sha256"].items():
        require(Path(name).name == name and sha_file(study_run / name) == expected,
                f"Frozen study source changed: {name}.")
    require(sha_file(sealed_data / "seal.json") == execution["seal_sha256"], "Data seal changed.")
    seal = read_json(sealed_data / "seal.json")
    require(seal["dataset"] == DATASET_REPO and seal["dataset_revision"] == DATASET_REVISION,
            "Wrong dataset identity.")
    for name, expected in seal["files_sha256"].items():
        require(sha_file(sealed_data / name) == expected, f"Sealed artifact changed: {name}.")
    require(sha_file(sealed_data / "protocol.json") == execution["protocol_sha256"], "Protocol changed.")
    require(sha_file(study_run / "scores.jsonl") == execution["scores_sha256"], "MPS scalar artifact changed.")
    with (study_run / "scores.jsonl").open("rb") as handle:
        scalar = json.loads(handle.readline())
    require(scalar["id"] == RECORD_ID and scalar["source"]["study_split"] == "dev"
            and scalar["source"]["official_label"] == 0 and scalar["source"]["selection_rank"] == 0,
            "Reference is not the already-observed first dev nonmember.")
    link = scalar["full_vector_record"]
    shard = Path(link["path"]).resolve()
    matching = [x for x in execution["vector_shards"] if Path(x["path"]).resolve() == shard]
    require(len(matching) == 1 and link["row_index"] == 0, "Ambiguous first-reference shard.")
    require(shard.stat().st_size == matching[0]["bytes"] and sha_file(shard) == matching[0]["sha256"],
            "MPS vector shard changed.")
    with gzip.open(shard, "rb") as handle:
        raw = handle.readline()
    require(hashlib.sha256(raw).hexdigest() == link["uncompressed_record_sha256"], "MPS vector record changed.")
    reference = json.loads(raw)
    require(reference["id"] == RECORD_ID and reference["source"] == scalar["source"], "Reference join differs.")
    require(reference["checkpoint_identity"]["model_dtype"] == "torch.float16"
            and reference["checkpoint_identity"]["device"].startswith("mps")
            and reference["checkpoint_identity"]["resolved_attention_implementation"] == "sdpa",
            "Reference is not the declared MPS/FP16/SDPA execution.")
    require(reference["target_count"] == 2048 and reference["selection"]["k_percent"] == 10
            and reference["statistics_chunk_tokens"] == 64 and reference["pandora_convention"]["k_percent"] == 10,
            "Reference scoring convention differs.")
    source = None
    with gzip.open(sealed_data / "inputs.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            candidate = json.loads(line)
            if candidate["id"] == RECORD_ID:
                source = candidate
                break
    require(source is not None and source["study_split"] == "dev" and source["official_label"] == 0,
            "Observed dev record is missing from the fixed input cohort.")
    ids = source["tokens"]
    require(len(ids) == 2049 and reference["original_token_ids"] == ids
            and reference["input_token_ids"] == ids[:-1] and reference["target_token_ids"] == ids[1:],
            "Reference does not preserve the exact 2,049 source / 2,048 input / 2,048 target IDs.")
    for values, key in ((ids, "original_token_ids_sha256_uint32_le"),
                        (ids[:-1], "model_input_ids_sha256_uint32_le")):
        require(hashlib.sha256(struct.pack(f"<{len(values)}I", *values)).hexdigest() == source[key],
                f"Source token identity changed: {key}.")
    require(hashlib.sha256(source["text"].encode()).hexdigest() == source["raw_text_sha256"], "Source text changed.")
    require(reference["source"] == {key: value for key, value in source.items() if key not in {"tokens", "text"}}
            and reference["source_text_sha256"] == source["raw_text_sha256"],
            "MPS reference source metadata/text differs from the sealed input.")
    support = full_fp32_support(reference)
    require(all(x["complete"] for x in support.values()), "Retained MPS reference has incomplete required FP32 support.")
    support[f"variants/{CPU64_VARIANT}"] = full_cpu64_support(reference)
    require(support[f"variants/{CPU64_VARIANT}"]["complete"],
            "Retained first MPS record lacks complete matching CPU-FP64 statistics.")
    receipts = json.loads((sealed_data / "download_receipts.json").read_text())
    matches = [x for x in receipts if x["repo_type"] == "model" and x["repo_id"] == MODEL_REPO
               and x["revision"] == MODEL_REVISION]
    require(len(matches) == 1, "Pinned model receipt is missing or ambiguous.")
    receipt = matches[0]
    require(receipt["files"] == execution["model_files_verified"], "Study model verification differs from receipt.")
    return {"execution": execution, "reference": reference, "reference_raw": raw,
            "source": source, "receipt": receipt, "mps_required_support": support,
            "reference_shard": matching[0], "reference_record_sha256": link["uncompressed_record_sha256"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-run", type=Path, required=True, help="Complete amended 3,000-record run; its process must have exited.")
    parser.add_argument("--sealed-data", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing evidence is never overwritten.")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output must be a new directory.")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "status": "preflight", "record_id": RECORD_ID, "learned_model_forwards_planned": 1,
        "learned_model_forwards_attempted": 0, "device": "cpu", "model_dtype": "float16",
        "threshold": None, "python": sys.version, "platform": platform.platform(),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")},
        "scope": "One observed dev record; CPU/MPS execution fidelity, not membership or full-logit parity.",
        "statistics_controls": "FP32 device sections mix forward and scoring-kernel effects. CPU-FP64 centered sections hold statistics device/dtype/function fixed across both forward outputs.",
        "cpu_float64_diagnostic_requested": True,
        "timing_limit": "Both records include CPU-FP64 statistics, but this single first-record comparison is not a controlled CPU/MPS throughput benchmark.",
        "settings": {key: str(value.resolve()) for key, value in vars(args).items()},
        "files_sha256": {}, "timing": {},
    }
    execution_path = args.output / "execution.json"
    execution_path.write_bytes(encoded(manifest))
    started = time.perf_counter()
    checkpoint = None
    try:
        verified = verified_reference(args.study_run, args.sealed_data)
        for dependency in ("torch", "transformers", "numpy"):
            require(manifest["versions"][dependency] == verified["execution"]["versions"][dependency],
                    f"Device comparison environment drift: {dependency}.")
        manifest["timing"]["reference_preflight_seconds"] = time.perf_counter() - started
        for source, destination in (
            (Path(__file__), "checkmia_forward_fidelity.py"),
            (args.study_run / "real_score_engine.py", "real_score_engine.py"),
            (args.study_run / "checkmia_real_scoring.py", "study_scoring_source.py"),
            (args.study_run / "checkmia_real_analysis.py", "study_analysis_source.py"),
            (args.study_run / "execution.json", "mps_execution.json"),
            (args.sealed_data / "seal.json", "data_seal.json"),
            (args.sealed_data / "protocol.json", "protocol.json"),
            (args.sealed_data / "download_receipts.json", "download_receipts.json"),
            (PROJECT / "contamcheckr/hashing.py", "hashing_source.py"),
            (PROJECT / "docs/CHECKMIA_FORWARD_FIDELITY_PLAN.md", "fidelity_plan.md"),
        ):
            shutil.copy2(source, args.output / destination)
        require(sha_file(args.study_run / "checkmia_real_scoring.py") ==
                verified["execution"]["source_sha256"]["checkmia_real_scoring.py"], "Frozen study runner changed.")
        if "amendment_sha256" in verified["execution"]:
            amendment = args.study_run / "amendment.json"
            require(sha_file(amendment) == verified["execution"]["amendment_sha256"], "Amendment changed.")
            shutil.copy2(amendment, args.output / "amendment.json")
        (args.output / "input.jsonl").write_bytes(encoded(verified["source"]))
        with gzip.open(args.output / "mps_reference.jsonl.gz", "xb", compresslevel=1) as handle:
            handle.write(verified["reference_raw"])
        manifest.update(reference_record_sha256=verified["reference_record_sha256"],
                        reference_shard=verified["reference_shard"], model_repo=MODEL_REPO,
                        model_revision=MODEL_REVISION, engine_source_sha256=ENGINE_SHA256,
                        mps_checkpoint_identity=verified["reference"]["checkpoint_identity"],
                        mps_required_support=verified["mps_required_support"])
        phase = time.perf_counter()
        manifest["model_files_verified_before_load"] = verify_model(args.model_path, verified["receipt"])
        manifest["timing"]["model_hash_verification_seconds"] = time.perf_counter() - phase
        manifest["status"] = "verified_before_cpu_model_load"
        execution_path.write_bytes(encoded(manifest))
        print(json.dumps({"stage": "loading_single_cpu_checkpoint", "record_id": RECORD_ID}), flush=True)
        if str(PROJECT) not in sys.path:
            sys.path.insert(0, str(PROJECT))
        # Execute only our already reviewed, hash-pinned local engine copy.
        spec = importlib.util.spec_from_file_location("checkmia_frozen_fidelity_engine", args.output / "real_score_engine.py")
        require(spec is not None and spec.loader is not None, "Cannot load the frozen local scoring engine.")
        engine = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = engine
        spec.loader.exec_module(engine)
        phase = time.perf_counter()
        checkpoint = engine.load_local_checkpoint(args.model_path, device="cpu", model_dtype="float16",
                                                 attention_implementation="sdpa", seed=42)
        manifest["timing"]["cpu_model_load_seconds"] = time.perf_counter() - phase
        identity = checkpoint.identity
        require(identity["device"].startswith("cpu") and identity["model_dtype"] == "torch.float16"
                and identity["context_limit"] == 2048 and identity["resolved_attention_implementation"] == "sdpa",
                "CPU model did not resolve the fixed dtype/context/attention convention.")
        require(identity["tokenizer_fingerprint"] == verified["reference"]["checkpoint_identity"]["tokenizer_fingerprint"],
                "CPU and MPS tokenizer identity differs.")
        manifest["cpu_checkpoint_identity"] = identity
        manifest["status"] = "scoring_single_cpu_record"
        manifest["learned_model_forwards_attempted"] = 1
        execution_path.write_bytes(encoded(manifest))
        print(json.dumps({"stage": "scoring_single_cpu_dev_record", "record_id": RECORD_ID}), flush=True)
        row = verified["source"]
        prepared = engine.prepare_checkmiabench_tokens(row["tokens"], RECORD_ID, text=row["text"])
        require(len(prepared.input_ids) == len(prepared.target_ids) == 2048, "Unexpected forwarded input/target count.")
        cpu = engine.score_prepared(checkpoint, prepared, chunk_tokens=64, k_percent=10,
                                    include_pandora=True, pandora_k_percent=10, include_cpu_float64=True)
        cpu["source"] = {key: value for key, value in row.items() if key not in {"text", "tokens"}}
        with gzip.open(args.output / "cpu_result.jsonl.gz", "xb", compresslevel=1) as handle:
            handle.write(encoded(cpu))
        require(cpu["forward_logit_dtype"] == verified["reference"]["forward_logit_dtype"] == "torch.float16",
                "Returned CPU/MPS forward-logit dtypes differ from the fixed FP16 convention; CPU result retained.")
        manifest["cpu_required_support"] = full_fp32_support(cpu)
        manifest["cpu_required_support"][f"variants/{CPU64_VARIANT}"] = full_cpu64_support(cpu)
        comparison = compare_records(cpu, verified["reference"])
        (args.output / "comparison.json").write_bytes(encoded(comparison))
        require(comparison["required_comparison_complete"],
                "CPU result has incomplete required FP32/CPU-FP64 support or convention; full record retained.")
        checkpoint = None
        manifest["model_files_verified_after_score"] = verify_model(args.model_path, verified["receipt"])
        require(sha_file(args.study_run / "execution.json") == sha_file(args.output / "mps_execution.json"),
                "Reference execution changed during the CPU comparison.")
        require(sha_file(args.output / "real_score_engine.py") == ENGINE_SHA256
                and sha_file(PROJECT / "experiments/real_score_engine.py") == ENGINE_SHA256,
                "Scoring engine changed during the CPU comparison.")
        require(sha_file(Path(__file__)) == sha_file(args.output / "checkmia_forward_fidelity.py"),
                "Fidelity source changed during execution.")
        require(sha_file(PROJECT / "contamcheckr/hashing.py") == sha_file(args.output / "hashing_source.py"),
                "Identity helper changed during execution.")
        require(sha_file(PROJECT / "docs/CHECKMIA_FORWARD_FIDELITY_PLAN.md") == sha_file(args.output / "fidelity_plan.md"),
                "Fidelity interpretation plan changed during execution.")
        manifest["status"] = "complete_required_fp32_and_cpu64_no_acceptance_threshold"
        print(json.dumps({"output": str(args.output), "record_id": RECORD_ID,
                          "status": manifest["status"], "comparison": str(args.output / "comparison.json")}), flush=True)
        return 0
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        checkpoint = None
        manifest["timing"]["total_wall_seconds"] = time.perf_counter() - started
        manifest["files_sha256"] = {path.name: sha_file(path) for path in sorted(args.output.iterdir())
                                    if path.is_file() and path.name != "execution.json"}
        execution_path.write_bytes(encoded(manifest))


if __name__ == "__main__":
    raise SystemExit(main())
