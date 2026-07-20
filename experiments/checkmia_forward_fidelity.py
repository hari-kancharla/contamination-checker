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
