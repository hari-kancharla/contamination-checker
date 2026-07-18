"""Sealed, randomized experiment for *additional fine-tuning* exposure.

Only the ``run`` command loads or trains a model.  It accepts an existing
local checkpoint and local JSONL pool, never downloads a model or dataset,
and creates a new run directory.  See docs/RANDOMIZED_EXPOSURE.md.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
from typing import Any, Callable


PROJECT = Path(__file__).resolve().parent.parent
SCHEMA = "contamcheckr.randomized-exposure.v1"
SCORES = ("loss", "mink", "minkpp")
ESTIMAND = (
    "Discrimination of randomized additional fine-tuning exposure within a "
    "fixed local pool; not unknown pretraining membership, false-positive "
    "calibration, or benchmark-performance inflation."
)


@dataclass(frozen=True)
class Settings:
    assignment_seed: int = 17
    training_seed: int = 17
    bootstrap_seed: int = 101
    exposed_fraction: float = 0.5
    group_field: str | None = None
    epochs: int = 1
    learning_rate: float = 0.00005
    training_batch_size: int = 2
    scoring_batch_size: int = 2
    max_tokens: int = 256
    k: float = 20.0
    bootstrap_draws: int = 2000
    device: str = "cpu"

    def validate(self) -> None:
        if not 0 < self.exposed_fraction < 1:
            raise ValueError("exposed_fraction must be strictly between zero and one.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite.")
        if not 0 < self.k <= 100:
            raise ValueError("k must be in (0, 100].")
        for name in ("epochs", "training_batch_size", "scoring_batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least one.")
        if self.max_tokens < 2 or self.bootstrap_draws < 20:
            raise ValueError("max_tokens must be >= 2 and bootstrap_draws >= 20.")
        if self.group_field in {"id", "text"}:
            raise ValueError("group_field must be a separate input metadata field.")
        if self.group_field is not None and not self.group_field.strip():
            raise ValueError("group_field cannot be empty.")
        if self.device not in {"cpu", "mps", "cuda", "auto"}:
            raise ValueError("Unsupported device.")


@dataclass
class Runtime:
    load: Callable[..., Any]
    train: Callable[..., dict[str, Any]]
    score: Callable[..., list[Any]]
    release: Callable[[Any], None]


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("xb") as handle:
        handle.write(_bytes(value))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("xb") as handle:
        for row in rows:
            handle.write(_bytes(row))


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_pool(path: Path, *, group_field: str | None = None) -> list[dict[str, str]]:
    """Validate one pool without inferring labels from text, IDs, or metadata."""
    return _read_pool_payload(path.read_bytes(), group_field=group_field)


def _read_pool_payload(payload: bytes, *, group_field: str | None) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Pool row {line_number} must be an object.")
        identifier, text = row.get("id"), row.get("text")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"Pool row {line_number} requires a nonempty string id.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Pool row {line_number} requires nonempty text.")
        if identifier in seen_ids:
            raise ValueError(f"Duplicate id: {identifier}")
        normalized = " ".join(text.split())
        if normalized in seen_texts:
            raise ValueError(f"Duplicate text (including whitespace variants): {identifier}")
        seen_ids.add(identifier)
        seen_texts.add(normalized)
        group = row.get(group_field) if group_field else identifier
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f"Pool row {line_number} requires a nonempty group string.")
        records.append({"id": identifier, "text": text, "group_id": group})
    if len(records) < 4:
        raise ValueError("Pool requires at least four distinct examples.")
    return sorted(records, key=lambda row: row["id"])


def assign_exposure(records: list[dict[str, str]], settings: Settings) -> list[dict[str, Any]]:
    settings.validate()
    groups = sorted({row["group_id"] for row in records})
    random.Random(settings.assignment_seed).shuffle(groups)
    count = int(len(groups) * settings.exposed_fraction)
    if count < 2 or len(groups) - count < 2:
        raise ValueError("Assignment needs at least two groups per arm; enlarge the pool or change fraction.")
    exposed = set(groups[:count])
    return [
        {
            "id": row["id"], "group_id": row["group_id"],
            "text_sha256": _sha(row["text"].encode("utf-8")),
            "exposed": row["group_id"] in exposed,
        }
        for row in records
    ]


def _source_payloads() -> dict[str, bytes]:
    paths = list((PROJECT / "contamcheckr").rglob("*.py"))
    paths += list((PROJECT / "experiments").glob("*.py"))
    paths += [PROJECT / "pyproject.toml"]
    return {str(path.relative_to(PROJECT)): path.read_bytes() for path in sorted(paths)}


def _source_manifest() -> dict[str, str]:
    return {name: _sha(payload) for name, payload in _source_payloads().items()}


def _model_manifest(path: Path) -> dict[str, Any]:
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError("--model must be an existing local checkpoint directory with config.json.")
    files = {
        str(item.relative_to(path)): {"sha256": _file_sha(item), "bytes": item.stat().st_size}
        for item in sorted(path.rglob("*")) if item.is_file()
    }
    if not any(name.endswith((".bin", ".safetensors")) for name in files):
        raise ValueError("Local checkpoint contains no .bin or .safetensors weights.")
    return {"path": str(path.resolve()), "files": files, "content_sha256": _sha(_bytes(files))}


def create_plan(input_path: Path, output_dir: Path, model_path: Path, settings: Settings) -> Path:
    """Seal a new plan before model loading. Never overwrite an existing run."""
    input_path, output_dir, model_path = (path.expanduser().resolve() for path in (input_path, output_dir, model_path))
    if output_dir.exists():
        raise FileExistsError(f"Run directory already exists: {output_dir}")
    if output_dir == model_path or model_path in output_dir.parents:
        raise ValueError("Run output must be outside the parent checkpoint directory.")
    input_payload = input_path.read_bytes()
    records = _read_pool_payload(input_payload, group_field=settings.group_field)
    assignment = assign_exposure(records, settings)
    model_manifest = _model_manifest(model_path)
    source_payloads = _source_payloads()
    sources = {name: _sha(payload) for name, payload in source_payloads.items()}
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, payload in source_payloads.items():
        destination = output_dir / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(payload)
    _write_rows(output_dir / "pool.jsonl", records)
    _write_rows(output_dir / "assignment.jsonl", assignment)
    _write_json(output_dir / "settings.json", asdict(settings))
    _write_json(output_dir / "parent_checkpoint.json", model_manifest)
    _write_json(output_dir / "source_manifest.json", sources)
    frozen_files = ("pool.jsonl", "assignment.jsonl", "settings.json", "parent_checkpoint.json", "source_manifest.json")
    plan = {
        "schema": SCHEMA, "estimand": ESTIMAND,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path), "input_file_sha256": _sha(input_payload),
        "assignment_unit": "group" if settings.group_field else "item",
        "assignment_mechanism": "Seeded simple random sample of groups without replacement, independent of item content.",
        "n_items": len(records), "n_exposed": sum(row["exposed"] for row in assignment),
        "n_groups": len({row["group_id"] for row in assignment}),
        "frozen_files_sha256": {name: _file_sha(output_dir / name) for name in frozen_files},
        "source_content_sha256": _sha(_bytes(sources)),
        "source_snapshot_directory": "source_snapshot",
        "parent_content_sha256": model_manifest["content_sha256"],
        "primary_comparison": "Adapted Min-K%++ AUC minus adapted mean-log-probability AUC; paired group bootstrap.",
        "secondary_comparisons": "All parent/adapted/change scores for loss, Min-K, Min-K%++; no post-hoc direction reversal or winner selection.",
        "tokenization_policy": (
            "Capture the tokenizer calls actually made by the scorer. Require unique "
            "effective inputs, zero skipped targets, and parity with the trainer's "
            "encode(add_special_tokens=False) text prefix. EOS appended by training "
            "is the sole allowed training/scoring suffix difference."
        ),
    }
    _write_json(output_dir / "plan.json", plan)
    _write_json(output_dir / "plan_seal.json", {"plan_sha256": _file_sha(output_dir / "plan.json")})
    return output_dir


def _verify_plan(run_dir: Path) -> tuple[dict[str, Any], Settings, list[dict[str, str]], list[dict[str, Any]], str]:
    plan_sha = _file_sha(run_dir / "plan.json")
    if json.loads((run_dir / "plan_seal.json").read_text())["plan_sha256"] != plan_sha:
        raise ValueError("Sealed plan was modified.")
    plan = json.loads((run_dir / "plan.json").read_text())
    if plan.get("schema") != SCHEMA:
        raise ValueError("Unsupported plan schema.")
    for name, expected in plan["frozen_files_sha256"].items():
        if _file_sha(run_dir / name) != expected:
            raise ValueError(f"Sealed file was modified: {name}")
    sources = json.loads((run_dir / "source_manifest.json").read_text())
    for name, expected in sources.items():
        snapshot = run_dir / "source_snapshot" / name
        if not snapshot.is_file() or _file_sha(snapshot) != expected:
            raise ValueError(f"Sealed source snapshot was modified: {name}")
    if _source_manifest() != sources:
        raise ValueError("Source changed after plan creation; create a new run.")
    checkpoint = json.loads((run_dir / "parent_checkpoint.json").read_text())
    if _model_manifest(Path(checkpoint["path"])) != checkpoint:
        raise ValueError("Parent checkpoint changed after plan creation.")
    settings = Settings(**json.loads((run_dir / "settings.json").read_text()))
    settings.validate()
    records = _read_rows(run_dir / "pool.jsonl")
    assignment = _read_rows(run_dir / "assignment.jsonl")
    if assign_exposure(records, settings) != assignment:
        raise ValueError("Assignment is inconsistent with the sealed pool/settings.")
    return plan, settings, records, assignment, plan_sha


def auc(positive: list[float], negative: list[float]) -> float:
    """Tie-aware AUC; all scores are oriented higher = more exposure-like."""
    if not positive or not negative or not all(math.isfinite(value) for value in positive + negative):
        raise ValueError("AUC needs nonempty finite scores in both arms.")
    ordered = sorted(negative)
    wins = sum(bisect_left(ordered, value) + 0.5 * (bisect_right(ordered, value) - bisect_left(ordered, value)) for value in positive)
    return wins / (len(positive) * len(negative))


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    location = probability * (len(ordered) - 1)
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (location - lower) * (ordered[upper] - ordered[lower])


def summarize(rows: list[dict[str, Any]], settings: Settings) -> dict[str, Any]:
    """Paired group bootstrap conditional on this pool and fitted checkpoints."""
    groups: dict[bool, dict[str, list[int]]] = {True: {}, False: {}}
    vectors = {f"{stage}/{score}": [row[stage][score] for row in rows] for stage in ("parent", "adapted", "delta") for score in SCORES}
    for index, row in enumerate(rows):
        groups[row["exposed"]].setdefault(row["group_id"], []).append(index)
    if set(groups[True]) & set(groups[False]):
        raise ValueError("A group cannot cross exposure arms.")
    if min(map(len, groups.values())) < 2:
        raise ValueError("Bootstrap requires at least two groups per arm.")
    arm_indices = {arm: [index for values in by_group.values() for index in values] for arm, by_group in groups.items()}

    def score_auc(values: list[float], indices: dict[bool, list[int]]) -> float:
        return auc([values[i] for i in indices[True]], [values[i] for i in indices[False]])

    points = {key: score_auc(values, arm_indices) for key, values in vectors.items()}
    draws: dict[str, list[float]] = {key: [] for key in vectors}
    differences: dict[str, list[float]] = {key: [] for key in vectors}
    rng = random.Random(settings.bootstrap_seed)
    for _ in range(settings.bootstrap_draws):
        indices: dict[bool, list[int]] = {}
        for arm, by_group in groups.items():
            keys = sorted(by_group)
            sampled = rng.choices(keys, k=len(keys))
            indices[arm] = [index for group in sampled for index in by_group[group]]
        replicate = {key: score_auc(values, indices) for key, values in vectors.items()}
        for key, value in replicate.items():
            draws[key].append(value)
            differences[key].append(value - replicate[key.split("/")[0] + "/loss"])
    metrics = {}
    for key, value in points.items():
        baseline = key.split("/")[0] + "/loss"
        metrics[key] = {
            "auc": value,
            "auc_ci_percentile_95": [_quantile(draws[key], .025), _quantile(draws[key], .975)],
            "auc_bootstrap_degenerate": min(draws[key]) == max(draws[key]),
            "comparison_baseline": baseline,
            "auc_difference_vs_loss": value - points[baseline],
            "difference_ci_percentile_95": [_quantile(differences[key], .025), _quantile(differences[key], .975)],
        }
    return {
        "estimand": ESTIMAND, "metrics": metrics,
        "uncertainty": {
            "method": "Paired percentile bootstrap, resampling groups separately within assigned arms.",
            "draws": settings.bootstrap_draws, "seed": settings.bootstrap_seed,
            "scope": "Conditional on this pool and fitted checkpoints; excludes training-seed and corpus variability. Saturated/degenerate intervals do not establish perfect population performance.",
        },
        "n_items": len(rows), "n_exposed": sum(row["exposed"] for row in rows),
        "n_exposed_groups": len(groups[True]), "n_held_out_groups": len(groups[False]),
        "score_orientation": "loss is mean token log-probability (higher is more likely); delta = adapted minus parent for each score.",
        "caveats": [
            "Held out means absent from this additional fine-tune, not proven absent from the parent training corpus.",
            "Random assignment removes systematic content-label construction; it does not make this corpus representative of deployment.",
            "Exact and whitespace-duplicate text is rejected; unlabelled near-duplicate/template families may still cross arms.",
            "Task performance and benchmark inflation are not measured. No contamination verdict, combined p-value, FDR claim, or calibrated false-positive rate is produced.",
            "Fine-tuning appends EOS; scoring uses the existing scorer without EOS. Truncation is recorded per item.",
            "The runner rejects scorer default-special-token changes relative to the trainer's raw text prefix, duplicate effective scored inputs, and any skipped target tokens.",
        ],
    }


def _get_runtime() -> Runtime:
    # Set before importing transformers through validation/model.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    from contamcheckr.model import load_model
    from contamcheckr.methods.minkpp import score_text_batch
    from experiments.validation import fine_tune, move_model_to_cpu
    return Runtime(load_model, fine_tune, score_text_batch, move_model_to_cpu)


class _ScoringTokenizerTrace:
    """Record inputs from the tokenizer calls actually used by the scorer.

    Re-encoding with encode(False) is not an identity trace: HF defaults can
    insert special tokens. This wrapper observes the returned input IDs and
    attention mask from the scorer's real batch calls instead.
    """

    def __init__(self, tokenizer: Any, max_tokens: int):
        self.original = tokenizer
        self.max_tokens = max_tokens
        self.inputs: dict[str, dict[str, Any]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"original", "max_tokens", "inputs"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.original, name, value)

    def __call__(self, texts: Any, *args: Any, **kwargs: Any) -> Any:
        encoded = self.original(texts, *args, **kwargs)
        submitted = [texts] if isinstance(texts, str) else list(texts)
        input_rows = encoded["input_ids"]
        input_rows = input_rows.tolist() if hasattr(input_rows, "tolist") else input_rows
        masks = encoded.get("attention_mask")
        masks = masks.tolist() if hasattr(masks, "tolist") else masks
        if submitted and input_rows and isinstance(input_rows[0], int):
            input_rows = [input_rows]
            masks = [masks] if masks is not None else None
        if len(input_rows) != len(submitted):
            raise ValueError("Cannot align actual scorer tokenizer inputs.")
        if kwargs.get("max_length") != self.max_tokens or not kwargs.get("truncation"):
            raise ValueError("Scorer tokenizer did not use the frozen token budget.")
        for index, text in enumerate(submitted):
            if text in self.inputs:
                raise ValueError("Scorer tokenized the same pool text more than once.")
            tokens = [int(value) for value in input_rows[index]]
            mask = [int(value) for value in masks[index]] if masks is not None else [1] * len(tokens)
            length = sum(mask)
            if len(mask) != len(tokens) or mask != [1] * length + [0] * (len(tokens) - length):
                raise ValueError("Runner requires contiguous right-padded scorer inputs.")
            # LocalCausalLM scores the first length tokens and their causal
            # targets input_ids[1:length]; padding is not a scored input.
            self.inputs[text] = {
                "token_ids": tokens[:length],
                "tokenizer_default_special_tokens": kwargs.get("add_special_tokens", True),
            }
        return encoded


def _validate_score_support(rows: list[dict[str, Any]], *, stage: str) -> None:
    for row in rows:
        if row["skipped_token_count"] != 0 or row["token_count"] != row["expected_target_count"]:
            raise ValueError(f"{stage} score support mismatch or skipped target tokens for {row['id']}.")
        if not row["training_text_prefix_matches"]:
            raise ValueError(f"{stage} scorer default special tokens/truncation differ from training text prefix for {row['id']}.")


def _reject_effective_duplicates(rows: list[dict[str, Any]]) -> None:
    seen: dict[str, str] = {}
    for row in rows:
        key = row["token_ids_sha256"]
        if key in seen:
            raise ValueError(
                f"Duplicate effective scored input: {seen[key]} and {row['id']}; "
                "distinct full texts can share the same retained token prefix."
            )
        seen[key] = row["id"]


def _score_checkpoint(runtime: Runtime, path: str, records: list[dict[str, str]], settings: Settings) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = runtime.load(path, device=settings.device, dtype="float32", seed=settings.training_seed, local_files_only=True)
    original_tokenizer = model.tokenizer
    trace = _ScoringTokenizerTrace(original_tokenizer, settings.max_tokens)
    model.tokenizer = trace
    try:
        scores = runtime.score(model, [row["text"] for row in records], k=settings.k, max_tokens=settings.max_tokens, batch_size=settings.scoring_batch_size)
        if len(scores) != len(records):
            raise ValueError("Scorer returned a different number of rows.")
        rows = []
        for record, score in zip(records, scores, strict=True):
            values = {key: float(getattr(score, key)) for key in SCORES}
            if not all(math.isfinite(value) for value in values.values()) or int(score.token_count) < 1:
                raise ValueError(f"Nonfinite or empty scores for {record['id']}.")
            captured = trace.inputs.get(record["text"])
            if captured is None:
                raise ValueError(f"Scorer did not expose an actual tokenizer call for {record['id']}.")
            token_ids = captured["token_ids"]
            if len(token_ids) < 2:
                raise ValueError(f"Scorer input has fewer than two tokens for {record['id']}.")
            training_prefix = original_tokenizer.encode(record["text"], add_special_tokens=False)[:settings.max_tokens]
            rows.append({
                "id": record["id"], **values,
                "token_count": int(score.token_count), "skipped_token_count": int(score.skipped_token_count),
                "truncated": bool(score.truncated), "token_ids_sha256": _sha(_bytes(token_ids)),
                "token_ids": token_ids, "expected_target_count": len(token_ids) - 1,
                "training_text_prefix_matches": list(training_prefix) == token_ids,
                "tokenizer_default_special_tokens": captured["tokenizer_default_special_tokens"],
                "token_identity_source": "Captured actual scorer tokenizer output and attention mask; causal targets are token_ids[1:].",
            })
        identity = {name: str(getattr(model, name, "unknown")) for name in ("model_id", "fingerprint", "tokenizer_fingerprint", "dtype", "device")}
        identity.update({name: getattr(model, name, None) for name in ("revision", "parameter_count", "context_limit")})
        return rows, identity
    finally:
        model.tokenizer = original_tokenizer
        runtime.release(model)
        del model
        gc.collect()


def _state(run_dir: Path, **values: Any) -> None:
    temporary = run_dir / "execution.tmp"
    _write_json(temporary, values)
    temporary.replace(run_dir / "execution.json")
