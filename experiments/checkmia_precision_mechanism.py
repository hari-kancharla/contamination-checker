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
