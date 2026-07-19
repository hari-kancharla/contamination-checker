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
