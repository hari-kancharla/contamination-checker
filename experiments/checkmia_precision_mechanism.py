"""One already observed development position: controlled precision diagnostics.

Do not run until the fixed 3k process AND the one-record CPU helper have exited.
Their completed manifests are required. Exactly one new MPS model forward is
planned. No downloads, training, threshold tuning or membership/AUC inference.
All outputs are exclusive; full-vocabulary logits are retained only for the
same 64-row statistics block as v3, plus its selected row.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import struct
import sys
import time
from typing import Any


PROJECT = Path(__file__).resolve().parent.parent
RECORD_ID = "nonmember:6834"
POSITION = 353
BLOCK_START, BLOCK_STOP, BLOCK_ROW = 320, 384, 33
VOCABULARY = 50304
ENGINE_SHA256 = "91cc8640f4fcf0aeb94a1454f7cda27e7d7dcc406f8eada9329b1b742d4ba6ff"
ORIGINAL_EXECUTION_SHA256 = "6f90df27b01349a095ef25084deea5b92f953f3eced31747adbe8b07fe160cbd"
ORIGINAL_SCORES_SHA256 = "55b31c9e7f55e20c78e4a9193361330af340ba00f559ca6b6022c9e09cfcacc6"
FLOOR = EPSILON = 1e-9


def encoded(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(encoded(value))


def read_json(path):
    return json.loads(Path(path).read_text())


def load_local_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Cannot import the reviewed local helper.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def scalar(value):
    """Keep finite signed values and distinguish every nonfinite class."""
    value = float(value.item() if hasattr(value, "item") else value)
    state = "nan" if math.isnan(value) else "positive_infinity" if value == math.inf else "negative_infinity" if value == -math.inf else "finite"
    return {"value": value if state == "finite" else None, "state": state,
            "signbit": math.copysign(1., value) < 0}


def write_array(directory, name, value):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.ascontiguousarray(value)
    path = directory / f"{name}.npy"
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
    return {"path": path.name, "sha256": sha(path), "dtype": value.dtype.str,
            "shape": list(value.shape), "data_bytes": value.nbytes,
            "data_sha256_c_order": hashlib.sha256(value.tobytes(order="C")).hexdigest()}


def fp64_products(logp, probability, targets):
    """CPU FP64 operands/products/sums; no probability renormalization."""
    import torch
    require(logp.device.type == probability.device.type == "cpu" and
            logp.dtype == probability.dtype == torch.float64,
            "FP64 calculations must start with CPU FP64 arrays.")
    first = probability * logp
    second = probability * logp.square()
    return first, second, first.sum(dim=-1), second.sum(dim=-1), logp.gather(-1, targets[:, None]).squeeze(-1)


def postprocess(target, mean, second, *, supplied_variance=None):
    variance = second - mean.square() if supplied_variance is None else supplied_variance
    clamped = variance.clamp(min=FLOOR)
    sigma = clamped.sqrt()
    numerator = target - mean
    denominator = sigma + EPSILON
    z = numerator / denominator
    return {"target_logp": target, "mean": mean, "first_product_sum": mean,
            "second_product_sum": second, "mean_squared": mean.square(),
            "variance": variance, "clamped_variance": clamped, "sigma": sigma,
            "numerator": numerator, "denominator": denominator, "z": z}


def summarize_stage(directory, name, logp, probability, first, second, readout,
                    reference_probability, conventions, *, row=BLOCK_ROW):
    import torch
    p = probability[row].detach().cpu().to(dtype=torch.float64)
    zero = p == 0
    delta = reference_probability - p
    observed = {key: scalar(value[row]) for key, value in readout.items()}
    supported = all(observed[key]["state"] == "finite" for key in ("target_logp", "mean", "variance", "sigma", "z"))
    arrays = {key: write_array(directory, f"{name}_{key}", value[row]) for key, value in
              (("logp", logp), ("probability", probability), ("first_product", first), ("second_product", second))}
    arrays["probability_zero_mask"] = write_array(directory, f"{name}_probability_zero_mask", zero)
    nonfinite_counts = {key: int((~torch.isfinite(value[row])).sum().item()) for key, value in
                        (("logp", logp), ("probability", probability), ("first_product", first), ("second_product", second))}
    return {"status": "finite_selected_target" if supported else "nonfinite_selected_target",
            "strict_v3_selected_target_support": supported, "observed": observed,
            "conventions": conventions, "component_dtypes": {key: str(value.dtype) for key, value in
                (("logp", logp), ("probability", probability), ("first_product", first), ("second_product", second),
                 ("mean", readout["mean"]), ("variance", readout["variance"]), ("sigma", readout["sigma"]), ("z", readout["z"]))},
            "probabilities_renormalized": False, "array_directory": "arrays", "arrays": arrays,
            "nonfinite_component_counts": nonfinite_counts,
            "probability_zero_count": int(zero.sum().item()),
            "probability_sum": scalar(p.sum()),
            "reference_probability_sum": scalar(reference_probability.sum()),
            "reference_mass_at_stage_zero_positions": scalar(reference_probability[zero].sum()),
            "signed_reference_minus_stage_total_probability": scalar(delta.sum()),
            "positive_probability_deficit_vs_reference": scalar(delta.clamp(min=0).sum()),
            "positive_probability_surplus_vs_reference": scalar((-delta).clamp(min=0).sum())}


def analyze_block(block, targets, directory):
    """Run controlled arithmetic on the actual MPS [64,V] logit view."""
    import torch
    require(block.device.type == "mps" and block.dtype == torch.float16 and
            tuple(block.shape) == (64, VOCABULARY) and len(targets) == 64,
            "Expected the exact 64×50304 FP16 MPS statistics block.")
    target_mps = torch.tensor(targets, dtype=torch.long, device=block.device)
    target_cpu = target_mps.detach().cpu()
    # Full native intermediates retain the same row count, vocabulary axis and
    # expression order as v3. No contiguous() or one-row softmax substitution.
    native_lp = torch.log_softmax(block, dim=-1)
    native_p = native_lp.exp()
    native_first = native_p * native_lp
    native_second = native_p * native_lp.square()
    native_mean = native_first.sum(dim=-1)
    native_moment2 = native_second.sum(dim=-1)
    native_target = native_lp.gather(-1, target_mps[:, None]).squeeze(-1)
    native = postprocess(native_target, native_mean, native_moment2)

    # Transfer BEFORE changing dtype. FP64 tensors are never allocated on MPS.
    cpu_logits = block.detach().cpu()
    lp64 = torch.log_softmax(cpu_logits.to(dtype=torch.float64), dim=-1)
    p64 = lp64.exp()
    reference_p = p64[BLOCK_ROW].clone()
    first, second, mean, moment2, target = fp64_products(lp64, p64, target_cpu)
    reference = postprocess(target, mean, moment2)
    centered_var = (p64 * (lp64 - mean[:, None]).square()).sum(dim=-1)
    centered = postprocess(target, mean, moment2, supplied_variance=centered_var)
    stages = {"cpu_fp64_reference": summarize_stage(directory, "cpu_fp64_reference", lp64, p64, first, second, reference, reference_p,
        "Returned FP16 MPS logits→CPU transfer→FP64 cast→CPU FP64 log_softmax/exp/products/sums/second-moment variance/guards/z; centered readout saved separately.")}
    reference_centered = {key: scalar(value[BLOCK_ROW]) for key, value in centered.items()}
    del lp64, p64, first, second, reference, centered

    lp_native64 = native_lp.detach().cpu().to(dtype=torch.float64)
    p_from_lp64 = lp_native64.exp()
    first, second, mean, moment2, target = fp64_products(lp_native64, p_from_lp64, target_cpu)
    stages["native_lp_fp64_rest"] = summarize_stage(directory, "native_lp_fp64_rest", lp_native64, p_from_lp64, first, second,
        postprocess(target, mean, moment2), reference_p,
        "Native [64,V] MPS FP16 log_softmax→CPU transfer→FP64 cast; exp/products/sums/variance/guards/z on CPU FP64. Native log-probabilities are not recomputed or renormalized.")
    del p_from_lp64, first, second

    p_native64 = native_p.detach().cpu().to(dtype=torch.float64)
    first, second, mean, moment2, target = fp64_products(lp_native64, p_native64, target_cpu)
    stages["native_lp_probability_fp64_products_sums"] = summarize_stage(directory, "native_lp_probability_fp64_products_sums", lp_native64, p_native64, first, second,
        postprocess(target, mean, moment2), reference_p,
        "Native MPS FP16 log_softmax AND exp results→CPU transfer→FP64 cast BEFORE products/square/sums/variance/guards/z.")
    stage2_first_row = first[BLOCK_ROW].clone()
    stage2_second_row = second[BLOCK_ROW].clone()
    del first, second

    first64 = native_first.detach().cpu().to(dtype=torch.float64)
    second64 = native_second.detach().cpu().to(dtype=torch.float64)
    mean64, moment264 = first64.sum(dim=-1), second64.sum(dim=-1)
    stages["native_products_fp64_sums"] = summarize_stage(directory, "native_products_fp64_sums", lp_native64, p_native64, first64, second64,
        postprocess(target, mean64, moment264), reference_p,
        "Native MPS FP16 log_softmax/exp/square/products→CPU transfer→FP64 cast BEFORE sums; variance/guards/z CPU FP64. This changes square and product-result arithmetic together relative to the previous stage.")
    product_loss = {
        "signed_first_product_sum_native_products_minus_fp64_products_at_same_native_lp_p": scalar((first64[BLOCK_ROW] - stage2_first_row).sum()),
        "signed_second_product_sum_native_products_minus_fp64_products_at_same_native_lp_p": scalar((second64[BLOCK_ROW] - stage2_second_row).sum()),
        "native_first_product_zero_count": int((first64[BLOCK_ROW] == 0).sum().item()),
        "native_second_product_zero_count": int((second64[BLOCK_ROW] == 0).sum().item()),
        "fp64_second_product_mass_at_native_product_zeros": scalar(stage2_second_row[second64[BLOCK_ROW] == 0].sum())}
    del first64, second64, mean64, moment264

    stages["fully_native_mps"] = summarize_stage(directory, "fully_native_mps", native_lp, native_p, native_first, native_second,
        native, reference_p,
        "Native MPS FP16 log_softmax/exp/square/products/reduction RESULTS/variance/clamp/sqrt/numerator/denominator/z, same expression order and [64,V] shape as v3. Internal accumulator precision is unobserved.")
    # Additional named readouts separate reduction-result changes from later
    # arithmetic. Neither is a repaired aggregate or proposed detector.
    mean_result64 = native_mean.detach().cpu().to(dtype=torch.float64)
    second_result64 = native_moment2.detach().cpu().to(dtype=torch.float64)
    target_result64 = native_target.detach().cpu().to(dtype=torch.float64)
    lifted = postprocess(target_result64, mean_result64, second_result64)
    guarded = postprocess(target_result64, mean_result64, second_result64,
                          supplied_variance=native["variance"].detach().cpu().to(dtype=torch.float64))
    counterfactuals = {
        "native_moment_results_fp64_postprocessing": {key: scalar(value[BLOCK_ROW]) for key, value in lifted.items()},
        "native_variance_fp64_guard_and_normalization": {key: scalar(value[BLOCK_ROW]) for key, value in guarded.items()},
        "scope": "Selected-target diagnostic readouts only. First lifts native mean/second-moment results before variance arithmetic; second additionally retains native raw variance. Both use FP64 guards/normalization and unchanged native target LP; no tail/aggregate score is defined."}
    guard = {"requested_variance_floor": FLOOR, "requested_denominator_epsilon": EPSILON,
             "native_floor_cast": scalar(torch.tensor(FLOOR, dtype=torch.float16, device=block.device)),
             "native_epsilon_cast": scalar(torch.tensor(EPSILON, dtype=torch.float16, device=block.device)),
             "native_variance_less_than_requested_floor_runtime": bool((native["variance"][BLOCK_ROW] < FLOOR).item())}
    return {"stages": stages, "cpu_fp64_centered_readout": reference_centered,
            "native_product_diagnostics": product_loss, "postmoment_counterfactual_readouts": counterfactuals,
            "guard_representation": guard,
            "scope": "One selected already-observed development target, same returned logit block. No aggregate membership score, tail, AUC, frequency estimate or internal-accumulator claim."}


def old_point(record, variant):
    s = record["variants"][variant]
    return {"variant_status": s["status"], "selected_target_support": s["per_token"]["support"][POSITION],
            "observed": {key: s["per_token"][key][POSITION] for key in ("logp", "mu", "variance", "sigma", "z")}}


def compare_old_point(old, new):
    mapping = {"logp": "target_logp", "mu": "mean", "variance": "variance", "sigma": "sigma", "z": "z"}
    result = {}
    for key, mapped in mapping.items():
        before, after = old["observed"][key], new[mapped]
        comparable = before is not None and math.isfinite(before) and after["state"] == "finite"
        result[key] = {"retained_old": before, "new": after,
                       "new_minus_old": after["value"] - before if comparable else None,
                       "absolute_difference": abs(after["value"] - before) if comparable else None,
                       "comparison_status": "finite_difference" if comparable else "nonfinite_or_old_null_no_difference_defined"}
    return result


def original_failed_reference(path, source):
    require(sha(path / "execution.json") == ORIGINAL_EXECUTION_SHA256 and sha(path / "scores.jsonl") == ORIGINAL_SCORES_SHA256,
            "Original development-failure artifacts differ from their observed immutable receipt.")
    e = read_json(path / "execution.json")
    require(e["status"] == "failed" and e["scored_records"] == 1, "Original observed run must contain only its failed first development record.")
    with (path / "scores.jsonl").open("rb") as stream:
        scalar_row = json.loads(stream.readline())
    link = scalar_row["full_vector_record"]
    shard = Path(link["path"])
    matching = [s for s in e["vector_shards"] if Path(s["path"]).resolve() == shard.resolve()]
    require(len(matching) == 1 and link["row_index"] == 0 and sha(shard) == matching[0]["sha256"], "Original observed vector shard changed.")
    with gzip.open(shard, "rb") as stream:
        raw = stream.readline()
    require(hashlib.sha256(raw).hexdigest() == link["uncompressed_record_sha256"], "Original observed vector row changed.")
    record = json.loads(raw)
    require(record["id"] == RECORD_ID and record["original_token_ids"] == source["tokens"] and
            record["input_token_ids"] == source["tokens"][:-1] and record["target_token_ids"] == source["tokens"][1:] and
            record["source"]["study_split"] == "dev" and record["source"]["official_label"] == 0 and
            record["variants"]["author_native"]["per_token"]["support"][POSITION] is False,
            "Original observed position/input identity or known development failure changed.")
    return record, raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-run", type=Path, required=True)
    parser.add_argument("--cpu-forward-run", type=Path, required=True, help="Completed one-record CPU helper; its process must already have exited.")
    parser.add_argument("--observed-failure-run", type=Path, default=PROJECT / "experiments/results/checkmia-real-v1/scoring")
    parser.add_argument("--sealed-data", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Output must be a fresh exclusive directory.")
    args.output.mkdir(parents=True, exist_ok=False)
    execution_path = args.output / "execution.json"
    manifest = {"schema_version": "checkmia_selected_precision_mechanism_v1", "status": "preflight", "record_id": RECORD_ID, "aligned_target_index": POSITION,
                "original_target_position": POSITION + 1, "block_start": BLOCK_START, "block_stop": BLOCK_STOP,
                "selected_block_row": BLOCK_ROW, "statistics_shape": [64, VOCABULARY],
                "full_forward_input_tokens": 2048, "model_forwards_planned": 1, "model_forwards_attempted": 0,
                "device": "mps", "model_dtype": "float16", "attention_implementation": "sdpa", "seed": 42,
                "scope": "One observed dev-target mechanism diagnostic; no aggregate/performance/published-program-crash claim.",
                "acceptance_threshold": None, "python": sys.version, "platform": platform.platform(),
                "versions": {x: importlib.metadata.version(x) for x in ("torch", "transformers", "numpy")},
                "settings": {k: str(v.resolve()) for k, v in vars(args).items()},
                "environment_whitelist": {k: os.environ.get(k) for k in
                    ("PYTORCH_MPS_FAST_MATH", "PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_PREFER_METAL",
                     "PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO")},
                "source_files_sha256": {}, "timing": {}}
    execution_path.write_bytes(encoded(manifest))
    checkpoint = None
    started = time.perf_counter()
    try:
        helper_path = PROJECT / "experiments/checkmia_forward_fidelity.py"
        sources = [Path(__file__), helper_path, PROJECT / "contamcheckr/hashing.py",
                   PROJECT / "docs/CHECKMIA_PRECISION_MECHANISM_PLAN.md"]
        manifest["source_files_sha256"] = {str(p.resolve()): sha(p) for p in sources}
        if str(PROJECT) not in sys.path:
            sys.path.insert(0, str(PROJECT))
        helper = load_local_module(helper_path, "checkmia_mechanism_preflight")
        verified = helper.verified_reference(args.study_run, args.sealed_data)
        cpu = read_json(args.cpu_forward_run / "execution.json")
        require(cpu["status"] == "complete_required_fp32_and_cpu64_no_acceptance_threshold" and
                cpu["record_id"] == RECORD_ID and cpu["engine_source_sha256"] == ENGINE_SHA256 and
                cpu["learned_model_forwards_attempted"] == 1,
                "The one-record CPU helper must have completed before this MPS mechanism run.")
        for name, expected in cpu["files_sha256"].items():
            require(sha(args.cpu_forward_run / name) == expected, f"Completed CPU-helper artifact changed: {name}.")
        require(sha(args.cpu_forward_run / "mps_execution.json") == sha(args.study_run / "execution.json"), "CPU helper used a different completed study.")
        for dependency, version in manifest["versions"].items():
            require(version == verified["execution"]["versions"][dependency] == cpu["versions"][dependency], f"Environment changed: {dependency}.")
        old, old_raw = original_failed_reference(args.observed_failure_run, verified["source"])
        manifest["original_observed_reference_sha256"] = hashlib.sha256(old_raw).hexdigest()
        manifest["amended_first_dev_reference_sha256"] = verified["reference_record_sha256"]
        manifest["cpu_helper_execution_sha256"] = sha(args.cpu_forward_run / "execution.json")
        manifest["completed_study_execution_sha256"] = sha(args.study_run / "execution.json")
        frozen = args.output / "source"
        frozen.mkdir()
        for p in sources:
            shutil.copy2(p, frozen / p.name)
        for source, name in ((args.study_run / "real_score_engine.py", "real_score_engine.py"),
                             (args.study_run / "execution.json", "completed_study_execution.json"),
                             (args.cpu_forward_run / "execution.json", "completed_cpu_execution.json"),
                             (args.observed_failure_run / "execution.json", "original_failure_execution.json"),
                             (args.sealed_data / "seal.json", "data_seal.json"),
                             (args.sealed_data / "protocol.json", "protocol.json"),
                             (args.sealed_data / "download_receipts.json", "download_receipts.json")):
            shutil.copy2(source, frozen / name)
        if "amendment_sha256" in verified["execution"]:
            amendment = args.study_run / "amendment.json"
            require(sha(amendment) == verified["execution"]["amendment_sha256"], "Frozen operational amendment changed.")
            manifest["source_files_sha256"][str(amendment.resolve())] = sha(amendment)
            shutil.copy2(amendment, frozen / "amendment.json")
        require(sha(frozen / "real_score_engine.py") == ENGINE_SHA256, "Wrong v3 engine source.")
        write_json(args.output / "input.json", verified["source"])
        for name, raw in (("original_observed_reference.jsonl.gz", old_raw), ("amended_first_dev_reference.jsonl.gz", verified["reference_raw"])):
            with gzip.open(args.output / name, "xb", compresslevel=1) as stream:
                stream.write(raw)
        manifest["model_files_verified_before_load"] = helper.verify_model(args.model_path, verified["receipt"])
        manifest["status"] = "verified_before_single_mps_load"
        execution_path.write_bytes(encoded(manifest))
        engine = load_local_module(frozen / "real_score_engine.py", "checkmia_mechanism_v3_engine")
        phase = time.perf_counter()
        checkpoint = engine.load_local_checkpoint(args.model_path, device="mps", model_dtype="float16",
                                                 attention_implementation="sdpa", seed=42)
        manifest["timing"]["model_load_seconds"] = time.perf_counter() - phase
        identity = checkpoint.identity
        require(identity["device"].startswith("mps") and identity["model_dtype"] == "torch.float16" and
                identity["context_limit"] == 2048 and identity["resolved_attention_implementation"] == "sdpa" and
                identity["tokenizer_fingerprint"] == verified["reference"]["checkpoint_identity"]["tokenizer_fingerprint"],
                "Model resolved to a different dtype/device/context/tokenizer/attention convention.")
        manifest["checkpoint_identity"] = identity
        import torch
        prepared = engine.prepare_checkmiabench_tokens(verified["source"]["tokens"], RECORD_ID, text=verified["source"]["text"])
        require(prepared.input_ids == tuple(old["input_token_ids"]) and prepared.target_ids == tuple(old["target_token_ids"]) and
                len(prepared.input_ids) == len(prepared.target_ids) == 2048,
                "Prepared causal IDs differ from the observed record.")
        manifest["input_identities_sha256_uint32_le"] = {
            key: hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()
            for key, ids in (("original", verified["source"]["tokens"]), ("forwarded_input", prepared.input_ids), ("targets", prepared.target_ids))}
        tensors = torch.tensor([prepared.input_ids], dtype=torch.long, device=checkpoint.device)
        targets = list(prepared.target_ids[BLOCK_START:BLOCK_STOP])
        require(targets == verified["source"]["tokens"][BLOCK_START + 1:BLOCK_STOP + 1] and
                targets[BLOCK_ROW] == verified["source"]["tokens"][POSITION + 1] and POSITION == BLOCK_START + BLOCK_ROW,
                "The selected target or 64-target block has shifted.")
        write_json(args.output / "block_target_ids.json", {"ids": targets, "aligned_logit_positions": list(range(BLOCK_START, BLOCK_STOP)),
                   "original_target_positions": list(range(BLOCK_START + 1, BLOCK_STOP + 1)),
                   "selected_row": BLOCK_ROW, "selected_target_id": targets[BLOCK_ROW]})
        manifest["status"] = "capturing_single_mps_forward"
        manifest["model_forwards_attempted"] = 1
        execution_path.write_bytes(encoded(manifest))
        phase = time.perf_counter()
        with torch.inference_mode():
            output = checkpoint.model(input_ids=tensors, attention_mask=torch.ones_like(tensors), use_cache=False)
            torch.mps.synchronize()
            manifest["timing"]["full_forward_seconds"] = time.perf_counter() - phase
            require(tuple(output.logits.shape) == (1, 2048, VOCABULARY) and output.logits.dtype == torch.float16,
                    "Model did not return the fixed full FP16 logit tensor.")
            block = output.logits[0, BLOCK_START:BLOCK_STOP, :]
            arrays = args.output / "arrays"
            arrays.mkdir()
            manifest["captured_array_directory"] = "arrays"
            manifest["captured_arrays"] = {"raw_logit_block": write_array(arrays, "raw_logit_block_fp16", block),
                                            "selected_raw_logit_row": write_array(arrays, "selected_raw_logit_row_fp16", block[BLOCK_ROW])}
            manifest["native_tensor_layout"] = {"full_logits_stride": list(output.logits.stride()),
                                                 "block_stride": list(block.stride()), "block_contiguous": block.is_contiguous()}
            phase = time.perf_counter()
            report = analyze_block(block, targets, arrays)
            torch.mps.synchronize()
            manifest["timing"]["controlled_statistics_seconds"] = time.perf_counter() - phase
        native_new = report["stages"]["fully_native_mps"]["observed"]
        report["reproduction_differences"] = {}
        for name, record in (("original_observed_failure", old), ("completed_amendment_first_dev", verified["reference"])):
            report["reproduction_differences"][name] = {
                "native": compare_old_point(old_point(record, "author_native"), native_new),
                "cpu_fp64_centered": compare_old_point(old_point(record, "cpu_float64_centered"), report["cpu_fp64_centered_readout"]),
                "scope": "Differences between newly returned logits' observable statistics and retained old statistics; old full logits were not saved. Matching summaries would not prove raw-logit equality."}
        report.update(record_id=RECORD_ID, aligned_target_index=POSITION, original_target_position=POSITION + 1,
                      selected_block_row=BLOCK_ROW, statistics_shape=[64, VOCABULARY],
                      selected_target_id=targets[BLOCK_ROW], model_forwards=1)
        write_json(args.output / "mechanism.json", report)
        require(report["stages"]["cpu_fp64_reference"]["strict_v3_selected_target_support"],
                "CPU-FP64 reference is nonfinite; raw arrays and stage report retained.")
        del output, block, tensors
        checkpoint = None
        manifest["model_files_verified_after_capture"] = helper.verify_model(args.model_path, verified["receipt"])
        for path, expected in manifest["source_files_sha256"].items():
            require(sha(path) == expected, f"Mechanism source/plan changed during execution: {path}.")
        require(sha(args.study_run / "execution.json") == manifest["completed_study_execution_sha256"] and
                sha(args.cpu_forward_run / "execution.json") == manifest["cpu_helper_execution_sha256"] and
                sha(PROJECT / "experiments/real_score_engine.py") == ENGINE_SHA256,
                "A completed prerequisite or the frozen v3 engine changed during capture.")
        manifest.update(status="complete_selected_target_stages_recorded_no_acceptance_threshold",
                        native_selected_target_support=report["stages"]["fully_native_mps"]["strict_v3_selected_target_support"],
                        native_support_failure_is_permitted_outcome=True)
        return 0
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        checkpoint = None
        manifest["timing"]["wall_seconds"] = time.perf_counter() - started
        manifest["files_sha256"] = {str(p.relative_to(args.output)): sha(p) for p in sorted(args.output.rglob("*"))
                                    if p.is_file() and p != execution_path}
        execution_path.write_bytes(encoded(manifest))


if __name__ == "__main__":
    raise SystemExit(main())
