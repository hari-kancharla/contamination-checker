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
