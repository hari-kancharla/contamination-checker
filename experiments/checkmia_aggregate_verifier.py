"""Independently reconstruct aggregates from a COMPLETED fixed CheckMIA run.

Standard library only: no model, project metric, accelerator, AUC, or threshold
selection. Run only after the parent-owned 3,000-record process has exited.
The arithmetic bound is declared below before inspecting any study outcomes.
Existing artifacts are read-only; --output must be a new directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import struct
import sys
import time
import zlib


ENGINE_SHA256 = "91cc8640f4fcf0aeb94a1454f7cda27e7d7dcc406f8eada9329b1b742d4ba6ff"
MODEL_REPO = "EleutherAI/pythia-1.4b-deduped"
MODEL_REVISION = "29376c0a610279f1d79507fc887c6170082f8839"
MODEL_WEIGHT_SHA256 = "852d5bc740615a542c02c72a049d369a089e64c0709275498f9c3e0d6d29054a"
DATASET_REPO = "jeffreygwang/pythia_dedupe_mia_0-97000_97000-98500"
DATASET_REVISION = "236fd818e7c877518d588eacc44658bdbf6b3a1b"
REQUIRED = ("fp32_second_moment", "fp32_centered")
N, TARGETS, TAIL = 3000, 2048, 204
U32, U64, MIN_NORMAL32 = 2.0 ** -24, 2.0 ** -53, 2.0 ** -126


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def reject_constant(value):
    raise ValueError(f"Nonstandard JSON numeric constant: {value}")


def decode(value):
    return json.loads(value, parse_constant=reject_constant)


def read_json(path):
    return decode(Path(path).read_bytes())


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def relative_file(directory, name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, "Unsafe artifact-relative path.")
    return directory / path


def finite_number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def binary32(value):
    require(finite_number(value), "An FP32 vector/scalar contains a missing or nonfinite number.")
    observed = struct.unpack("<f", struct.pack("<f", value))[0]
    require(float(value).hex() == observed.hex(), "A declared FP32 value is not exactly representable in binary32.")
    return float(value)


def gamma(count, unit_roundoff):
    require(0 <= count * unit_roundoff < 1, "Invalid forward-error bound dimension.")
    return count * unit_roundoff / (1 - count * unit_roundoff)


def reconstruct(values, divisor=1, *, division="none"):
    """FSUM reference and conservative bound, independent of observed scores.

    Reduction: gamma_n*sumabs + n*2^-126/(1-n*u32). The additive term conservatively
    covers gradual underflow or flushing a subnormal FP32 reduction result.
    FP32 mean/division: allow two roundings (rounded reciprocal then product),
    also covering one correctly-rounded division. Zlib uses a Python FP64
    division after the stored FP32 sum. Backend accumulator precision is not
    inferred; the bound assumes operations are at least FP32, no overflow,
    and at most this rounding/underflow model. It is not an accuracy guarantee
    for log-softmax, moments, per-token z, model logits, or upstream labels.
    """
    require(values and divisor > 0, "Reduction needs values and a positive divisor.")
    require(division in {"none", "fp32", "python_fp64"}, "Unknown division convention.")
    require(division != "none" or divisor == 1, "An undivided sum must have divisor one.")
    total = math.fsum(values)
    sumabs = math.fsum(abs(value) for value in values)
    reduction_error = gamma(len(values), U32) * sumabs + len(values) * MIN_NORMAL32 / (1 - len(values) * U32)
    quotient = total / divisor
    division_error = 0.0
    if division != "none":
        unit = U32 if division == "fp32" else U64
        division_error = gamma(2 if division == "fp32" else 1, unit) * (abs(total) + reduction_error) / divisor
        if division == "fp32":
            division_error += 2 * MIN_NORMAL32 / (1 - 2 * U32)
    # Explicitly cover rounding of the host FSUM/reference/bound bookkeeping.
    reference_error = 4 * math.ulp(total) / divisor + 4 * math.ulp(quotient)
    bound = math.nextafter(reduction_error / divisor + division_error + reference_error, math.inf)
    return {"reconstructed": quotient, "absolute_error_bound": bound,
            "count": len(values), "divisor": divisor, "division": division,
            "sum_abs_inputs": sumabs, "reduction_error_bound": reduction_error,
            "final_division_error_bound": division_error, "host_reference_rounding_bound": reference_error}


def tail_values(values, indices, count):
    require(len(values) == TARGETS and len(indices) == count and
            all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < TARGETS for i in indices) and
            len(set(indices)) == count, "Invalid or repeated tail indices.")
    ordered = sorted(values)
    cutoff = ordered[count - 1]
    selected = set(indices)
    require(all(values[i] <= cutoff for i in indices) and
            all(i in selected for i, value in enumerate(values) if value < cutoff),
            "Saved indices do not select the exact k smallest values, allowing cutoff ties.")
    return ordered[:count], {"selected_count": count, "cutoff": cutoff,
                            "strictly_below_cutoff": sum(value < cutoff for value in values),
                            "equal_to_cutoff": sum(value == cutoff for value in values)}


def aggregate_check(summary, name, observed, result, record_id):
    require(finite_number(observed), f"Required aggregate missing/nonfinite: {record_id}/{name}.")
    difference = abs(float(observed) - result["reconstructed"])
    bound = result["absolute_error_bound"]
    within = difference <= bound
    detail = {"record_id": record_id, "observed": observed, **result,
              "absolute_difference": difference, "within_declared_bound": within}
    entry = summary.setdefault(name, {"count": 0, "outside_bound_count": 0,
                                     "max_absolute_difference": -1., "max_difference_over_bound": -1.})
    entry["count"] += 1
    entry["outside_bound_count"] += not within
    if difference > entry["max_absolute_difference"]:
        entry.update(max_absolute_difference=difference, max_difference_case=detail)
    ratio = difference / bound
    if ratio > entry["max_difference_over_bound"]:
        entry.update(max_difference_over_bound=ratio, max_ratio_case=detail)


def token_vector(section, key):
    values = section["per_token"][key]
    require(len(values) == TARGETS, "Required vector does not cover 2048 targets.")
    return [binary32(value) for value in values]


def validate_required(section, *, pandora=False):
    require(section["status"] == "complete" and section["supported_target_count"] == TARGETS and
            section["selected_token_count"] == TAIL and len(section["per_token"]["support"]) == TARGETS and
            all(value is True for value in section["per_token"]["support"]),
            "Required FP32 section lacks complete fixed target/tail support.")
    components = ("log_softmax", "mean", "std", "normalized_score") if pandora else (
        "log_softmax", "target_logp", "moment_product", "moment_reduction", "moment_result", "variance", "sigma", "normalized_score")
    require(all(section["component_dtypes"][key] == "torch.float32" for key in components) and
            section["component_dtypes"]["computation_device"].startswith("mps"),
            "Required section uses a different dtype/statistics device.")
    if not pandora:
        require(section["aggregate_valid"] is True, "Required weighted aggregate is unavailable.")


def verify_record(record, source, metric_summary, counts):
    ids = source["tokens"]
    require(len(ids) == TARGETS + 1 and record["original_token_ids"] == ids and
            record["input_token_ids"] == ids[:-1] and record["target_token_ids"] == ids[1:] and
            record["target_sequence_positions"] == list(range(1, TARGETS + 1)) and
            record["logit_positions"] == list(range(TARGETS)), "Source/input/target alignment differs.")
    require(record["source"] == {k: v for k, v in source.items() if k not in {"text", "tokens"}}, "Source join differs.")
    for values, key in ((ids, "original_token_ids_sha256_uint32_le"), (ids[:-1], "model_input_ids_sha256_uint32_le")):
        require(hashlib.sha256(struct.pack(f"<{len(values)}I", *values)).hexdigest() == source[key], "Sealed token-ID hash differs.")
    for values, key in ((ids[:-1], "input_token_ids_sha256"), (ids[1:], "target_token_ids_sha256")):
        digest = hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()
        require(digest == record[key], "Scored token-ID hash differs.")
    text = source["text"].encode("utf-8")
    require(hashlib.sha256(text).hexdigest() == source["raw_text_sha256"] == record["source_text_sha256"], "Source text hash differs.")
    compressed = len(zlib.compress(text))
    require(compressed == record["compressed_source_text_bytes"], "Exact source text Zlib byte count differs.")
    require(record["target_count"] == TARGETS and record["statistics_chunk_tokens"] == 64 and
            record["selection"] == {"k_percent": 10.0, "rounding": "max(1,floor(ratio*target_count))", "selected_count": TAIL} and
            record["pandora_convention"]["k_percent"] == 10 and record["pandora_convention"]["selected_count"] == TAIL and
            record["runner_eligibility"]["primary_complete"] is True, "Fixed scoring convention differs.")
    for name in REQUIRED:
        section = record["variants"][name]
        validate_required(section)
        lp, z = token_vector(section, "logp"), token_vector(section, "z")
        lp_tail, lp_info = tail_values(lp, section["mink_selected_indices"], TAIL)
        z_tail, z_info = tail_values(z, section["minkpp_selected_indices"], TAIL)
        counts["verified_tail_selections"] += 2
        counts["tail_selections_with_cutoff_ties"] += int(lp_info["equal_to_cutoff"] > 1) + int(z_info["equal_to_cutoff"] > 1)
        reductions = {"loss": reconstruct(lp, TARGETS, division="fp32"),
                      "sum_logp": reconstruct(lp), "mink": reconstruct(lp_tail, TAIL, division="fp32"),
                      "minkpp": reconstruct(z_tail, TAIL, division="fp32"),
                      "zlib": reconstruct(lp, compressed, division="python_fp64")}
        for metric, result in reductions.items():
            observed = section["scores"][metric]
            if metric != "zlib":
                binary32(observed)
            aggregate_check(metric_summary, f"{name}.{metric}", observed, result, record["id"])
        require(section["scores"]["zlib"] == section["scores"]["sum_logp"] / compressed,
                "Stored Zlib score is not the stored FP32 sum divided on Python FP64 by exact compressed source bytes.")
    section = record["pandora_comparators"]["fp32"]
    validate_required(section, pandora=True)
    z = token_vector(section, "z")
    selected, info = tail_values(z, section["selected_indices"], TAIL)
    counts["verified_tail_selections"] += 1
    counts["tail_selections_with_cutoff_ties"] += info["equal_to_cutoff"] > 1
    result = reconstruct(selected, TAIL, division="fp32")
    score, feature = (binary32(section["scores"][key]) for key in ("pandora_minkxx", "pandora_minkxx_feature"))
    require(feature.hex() == (-score).hex(), "Pandora original feature/high-member signs disagree.")
    aggregate_check(metric_summary, "pandora_fp32.pandora_minkxx", score, result, record["id"])
    aggregate_check(metric_summary, "pandora_fp32.pandora_minkxx_feature", feature,
                    {**result, "reconstructed": -result["reconstructed"]}, record["id"])
    unavailable = []
    for group, excluded in (("variants", set(REQUIRED)), ("pandora_comparators", {"fp32"})):
        for name, section in record[group].items():
            if name in excluded:
                continue
            counts[f"optional_status/{group}/{name}/{section['status']}"] += 1
            if section["status"] != "complete":
                require(all(value is None for value in section["scores"].values()), "An unavailable optional score has been filled.")
                index_keys = ("selected_indices",) if group == "pandora_comparators" else ("mink_selected_indices", "minkpp_selected_indices")
                require(all(section[key] == [] for key in index_keys), "Unavailable optional tail indices were filled.")
                unavailable.append(f"{group}/{name}")
    require(sorted(unavailable) == sorted(record["runner_eligibility"]["optional_failed_sections"]), "Optional failure ledger differs from vectors.")
    return unavailable


def scalar_projection(record):
    excluded = {"original_token_ids", "input_token_ids", "target_token_ids", "target_input_positions", "target_sequence_positions", "logit_positions", "variants", "pandora_comparators"}
    projection = {key: value for key, value in record.items() if key not in excluded}
    projection["variants"] = {name: {key: value for key, value in section.items() if key not in {"per_token", "mink_selected_indices", "minkpp_selected_indices"}} for name, section in record["variants"].items()}
    projection["pandora_comparators"] = {name: {key: value for key, value in section.items() if key not in {"per_token", "selected_indices"}} for name, section in record["pandora_comparators"].items()}
    return projection
