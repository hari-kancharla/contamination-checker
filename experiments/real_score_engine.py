"""Inspectible, local-only scoring of externally labeled real model inputs.

This module does not infer membership labels, train models, or execute upstream
code.  All variants consume one shared model forward pass.  See
``docs/REAL_SCORING.md`` for the exact conventions and their limits.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import importlib.metadata
import json
import math
from numbers import Integral
from pathlib import Path
import resource
import sys
import time
from typing import Any, Iterable


ENGINE_VERSION = "contamcheckr.real-score.v3"
AUTHOR_REVISION = "6eb88de34a09dea85c4b41deff4048e5a203e1b0"
AUTHOR_SOURCE = f"https://github.com/LuckyDaydreamer/OLMo-Detect/tree/{AUTHOR_REVISION}"
PANDORA_REVISION = "189a1f43fed580f002ebba3c894c8509f9da14b0"
PANDORA_SOURCE = f"https://github.com/safr-ai-lab/pandora_llm/tree/{PANDORA_REVISION}/src/pandora_llm/features"
VARIANTS = (
    "author_native",
    "fp32_second_moment",
    "fp32_centered",
    "fixed_fp32_logp_bf16_reductions",
)
VARIANCE_FLOOR = 1e-9
DENOMINATOR_EPSILON = 1e-9


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _digest_ids(values: Iterable[int]) -> str:
    return hashlib.sha256(_json_bytes(list(values))).hexdigest()


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids(values: Any) -> tuple[int, ...]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple)) or any(
        not isinstance(value, Integral) or isinstance(value, bool) or value < 0
        for value in values
    ):
        raise ValueError("Token IDs must be a one-dimensional sequence of nonnegative integers.")
    return tuple(int(value) for value in values)


@dataclass(frozen=True)
class PreparedInput:
    identifier: str
    input_mode: str
    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    text: str | None
    convention: dict[str, Any]
    original_token_ids: tuple[int, ...] | None = None


@dataclass
class LocalCheckpoint:
    model: Any
    tokenizer: Any
    device: Any
    context_limit: int
    identity: dict[str, Any]


def prepare_olmo_plaintext(
    tokenizer: Any, text: str, identifier: str, *, max_tokens: int = 4096,
) -> PreparedInput:
    """Mirror the pinned author's plain input path, without chat formatting."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Plaintext must be a nonempty string.")
    raw_text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    # Upstream data_utils.extract_text strips before tokenization and Zlib.
    text = text.strip()
    if max_tokens < 2 or max_tokens > 4096:
        raise ValueError("OLMo plaintext max_tokens must be between 2 and 4096.")
    if getattr(tokenizer, "truncation_side", "right") != "right":
        raise ValueError("The author-compatible plaintext path requires right truncation.")
    encoded = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_tokens)
    input_ids = _ids(encoded["input_ids"])
    complete = _ids(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
    if len(input_ids) < 2:
        raise ValueError("Plaintext must yield at least two input tokens.")
    targets = input_ids[1:]
    eos = getattr(tokenizer, "eos_token_id", None)
    removed_eos = bool(targets and eos is not None and targets[-1] == eos)
    if removed_eos:
        targets = targets[:-1]
    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
    if any(value in special_ids for value in targets):
        raise ValueError("Special token found in plaintext targets after final EOS removal.")
    if not targets:
        raise ValueError("Plaintext has no targets after the causal shift and EOS handling.")
    return PreparedInput(str(identifier), "olmo_author_plaintext", input_ids, targets, text, {
        "tokenization": "tokenizer(add_special_tokens=True,truncation=True)",
        "source_text_preprocessing": "str.strip() before tokenization and Zlib, matching upstream extract_text",
        "raw_source_text_sha256": raw_text_sha256,
        "chat_template_applied": False,
        "max_tokens": max_tokens,
        "author_max_tokens": 4096,
        "context_variant": max_tokens != 4096,
        "source_token_count": len(complete),
        "source_token_ids_sha256": _digest_ids(complete),
        "truncated": len(complete) > len(input_ids),
        "final_eos_target_removed": removed_eos,
        "other_special_targets": "rejected",
    })


def prepare_checkmiabench_tokens(
    token_ids: Any, identifier: str, *, text: str | None = None,
    max_tokens: int | None = None,
) -> PreparedInput:
    """Use released token IDs directly; never decode and re-encode them.

    Forward the source except its final token; score every following token,
    including that final external target. This omits an unused final query,
    preserving all original causal targets without exceeding a 2048-input
    context for released 2049-ID sequences. Original IDs are retained in full.
    An explicit cap counts FORWARDED inputs and declares a prefix experiment.
    """
    original = _ids(token_ids)
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("An explicit forwarded-input cap must be at least one token.")
    selected = original if max_tokens is None else original[:max_tokens + 1]
    if len(selected) < 2:
        raise ValueError("A source sequence must contain at least two tokens.")
    if text is not None and not isinstance(text, str):
        raise ValueError("Optional source text must be a string.")
    return PreparedInput(str(identifier), "checkmiabench_original_tokens", selected[:-1],
                         selected[1:], text, {
        "tokenization": "none: original released token IDs",
        "decode_reencode": False,
        "max_tokens": max_tokens,
        "max_tokens_unit": "forwarded input tokens; retains up to max_tokens+1 source IDs",
        "context_variant": max_tokens is not None,
        "source_token_count": len(original),
        "source_token_ids_sha256": _digest_ids(original),
        "truncated": len(selected) < len(original),
        "selected_source_token_count": len(selected),
        "causal_target_layout": "input=selected_source[:-1];targets=selected_source[1:]",
        "unused_final_query_omitted": True,
        "final_target_external_to_forward_input": True,
        "final_target_original_position": len(selected) - 1,
        "original_token_ids_retained": True,
        "final_eos_target_removed": False,
        "other_special_targets": "retained as original training-sequence tokens",
    }, original_token_ids=original)


def load_local_checkpoint(
    path: str | Path, *, device: str = "mps", model_dtype: str = "bfloat16",
    attention_implementation: str = "sdpa", seed: int = 42,
) -> LocalCheckpoint:
    """Load an existing local checkpoint; downloads and custom code are disabled."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_dir() or not (checkpoint / "config.json").is_file():
        raise ValueError("Model path must be an existing local checkpoint with config.json.")
    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if model_dtype not in dtypes or device not in {"cpu", "mps", "cuda"}:
        raise ValueError("Choose an explicit supported device and model dtype.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; select CPU explicitly for a separate recorded run.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable.")
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True,
                                             trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint), local_files_only=True, trust_remote_code=False,
        dtype=dtypes[model_dtype], attn_implementation=attention_implementation,
    )
    model.to(torch.device(device))
    model.eval()
    model.config.use_cache = False
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if not isinstance(context_limit, int) or context_limit < 2:
        raise ValueError("Model must expose a finite max_position_embeddings context limit.")
    from contamcheckr.hashing import local_model_signature, tokenizer_fingerprint
    identity = {
        "local_checkpoint": str(checkpoint),
        "model_dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
        "model_type": getattr(model.config, "model_type", None),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "context_limit": context_limit,
        "requested_attention_implementation": attention_implementation,
        "resolved_attention_implementation": getattr(model.config, "_attn_implementation", None),
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "local_metadata_signature": local_model_signature(str(checkpoint)),
        "weight_content_sha256": None,
        "weight_identity_note": "Attach the separately verified download/content-hash receipt; metadata alone is not weight identity.",
        "use_cache": False,
        "seed": seed,
    }
    return LocalCheckpoint(model, tokenizer, next(model.parameters()).device, context_limit, identity)


def _sync(device: Any) -> None:
    import torch
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_snapshot(device: Any) -> dict[str, Any]:
    import torch
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"process_peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024}
    if device.type == "mps":
        result.update({name: getattr(torch.mps, name)() for name in (
            "current_allocated_memory", "driver_allocated_memory", "recommended_max_memory",
        )})
    return result


def _vector_payload(tensor: Any) -> list[Any]:
    """Keep JSON standard: nonfinite observations are explicit nulls plus support masks."""
    values = tensor.detach().cpu().tolist()
    return [float(value) if math.isfinite(float(value)) else None for value in values]


def score_logits_variants(
    logits: Any, target_ids: Any, *, k_percent: float = 20.0,
    chunk_tokens: int = 64, text: str | None = None,
    include_cpu_float64: bool = False, include_pandora: bool = False,
    pandora_k_percent: float = 10.0,
) -> dict[str, Any]:
    """Compute all variants from the same aligned [targets,vocabulary] logits.

    Vocabulary statistics are bounded by ``chunk_tokens``. The original model
    logits themselves remain resident. No per-token support is silently dropped.
    Native versus FP32 changes both log-softmax and arithmetic precision. The
    fixed-logp diagnostic instead holds FP32 logp/probability/products fixed.
    Its FP32 moment results are explicitly cast to BF16 after reduction.
    """
    import torch
    import zlib
    if logits.ndim != 2 or not logits.is_floating_point() or logits.shape[0] < 1:
        raise ValueError("Expected nonempty floating [targets,vocabulary] logits.")
    targets = _ids(target_ids)
    if len(targets) != logits.shape[0] or max(targets) >= logits.shape[1]:
        raise ValueError("Target IDs must align exactly with logits rows and vocabulary.")
    if not math.isfinite(k_percent) or not 0 < k_percent <= 100 or chunk_tokens < 1:
        raise ValueError("k_percent must be in (0,100] and chunk_tokens positive.")
    if include_pandora and (not math.isfinite(pandora_k_percent) or not 0 < pandora_k_percent <= 100):
        raise ValueError("pandora_k_percent must be in (0,100].")
    variants = list(VARIANTS) + (["cpu_float64_centered"] if include_cpu_float64 else [])
    keys = ("logp", "mu", "variance", "sigma", "z", "clamp_mask", "support")
    saved: dict[str, dict[str, list[Any]]] = {name: {key: [] for key in keys} for name in variants}
    component_dtypes: dict[str, dict[str, str]] = {}
    pandora_saved: dict[str, dict[str, list[Any]]] = {
        name: {key: [] for key in ("logp", "mean", "std", "z", "support")}
        for name in ("native", "fp32")
    } if include_pandora else {}
    pandora_dtypes: dict[str, dict[str, str]] = {}

    def retain_pandora(name: str, logp: Any, target_logp: Any) -> None:
        # Pinned Pandora LOSS.py uses unweighted torch.std_mean, correction=1;
        # this is separate from the probability-weighted Min-K++ variants.
        std, mean = torch.std_mean(logp, dim=-1, correction=1)
        z = (target_logp - mean) / (std + 1e-8)
        support = (torch.isfinite(target_logp) & torch.isfinite(mean) &
                   torch.isfinite(std) & torch.isfinite(z))
        for key, value in dict(logp=target_logp, mean=mean, std=std, z=z, support=support).items():
            pandora_saved[name][key].append(value.detach())
        pandora_dtypes[name] = {
            "log_softmax": str(logp.dtype), "mean": str(mean.dtype),
            "std": str(std.dtype), "normalized_score": str(z.dtype),
            "computation_device": str(logp.device),
        }

    def retain(name: str, logp: Any, mu: Any, variance: Any, target_logp: Any,
               *, normalize_fp32: bool = False, product_dtype: Any = None,
               reduction_dtype: Any = None) -> None:
        sigma = variance.clamp(min=VARIANCE_FLOOR).sqrt()
        if normalize_fp32:
            z = (target_logp.float() - mu.float()) / (sigma.float() + DENOMINATOR_EPSILON)
        else:
            z = (target_logp - mu) / (sigma + DENOMINATOR_EPSILON)
        support = (torch.isfinite(target_logp) & torch.isfinite(mu) &
                   torch.isfinite(variance) & torch.isfinite(sigma) & torch.isfinite(z))
        values = dict(logp=target_logp, mu=mu, variance=variance, sigma=sigma, z=z,
                      clamp_mask=variance < VARIANCE_FLOOR, support=support)
        for key, value in values.items():
            saved[name][key].append(value.detach())
        component_dtypes[name] = {
            "log_softmax": str(logp.dtype), "target_logp": str(target_logp.dtype),
            "moment_product": str(product_dtype or logp.dtype),
            "moment_reduction": str(reduction_dtype or mu.dtype),
            "moment_result": str(mu.dtype),
            "variance": str(variance.dtype), "sigma": str(sigma.dtype), "normalized_score": str(z.dtype),
            "computation_device": str(logp.device),
        }

    with torch.inference_mode():
        for start in range(0, len(targets), chunk_tokens):
            stop = min(start + chunk_tokens, len(targets))
            block = logits[start:stop]
            selected = torch.tensor(targets[start:stop], dtype=torch.long, device=block.device)
            native_lp = torch.log_softmax(block, dim=-1)
            native_p = native_lp.exp()
            native_mu = (native_p * native_lp).sum(dim=-1)
            native_second = (native_p * native_lp.square()).sum(dim=-1)
            native_target = native_lp.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
            retain("author_native", native_lp, native_mu, native_second - native_mu.square(), native_target)
            if include_pandora:
                retain_pandora("native", native_lp, native_target)

            lp = native_lp if block.dtype == torch.float32 else torch.log_softmax(block.float(), dim=-1)
            probability = lp.exp()
            first_products = probability * lp
            second_products = probability * lp.square()
            mu = first_products.sum(dim=-1)
            second = second_products.sum(dim=-1)
            target = lp.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
            if include_pandora:
                retain_pandora("fp32", lp, target)
            retain("fp32_second_moment", lp, mu, second - mu.square(), target)
            centered = (probability * (lp - mu.unsqueeze(-1)).square()).sum(dim=-1)
            retain("fp32_centered", lp, mu, centered, target)

            # Hold FP32 logp, probabilities and moment products fixed. Only
            # moment results/variance/sqrt use BF16; target z remains FP32.
            # Explicit casts occur AFTER the existing FP32 sums. A BF16 dtype
            # argument to sum would first quantize its operands as well.
            # This is a mechanism diagnostic, not the author's native pipeline.
            reduced_mu = mu.to(torch.bfloat16)
            reduced_second = second.to(torch.bfloat16)
            retain("fixed_fp32_logp_bf16_reductions", lp, reduced_mu,
                   reduced_second - reduced_mu.square(), target, normalize_fp32=True,
                   product_dtype=torch.float32, reduction_dtype=torch.float32)
            if include_cpu_float64:
                # FP64 is used solely on CPU, including when logits originated
                # on MPS. No FP64 tensor is allocated on MPS.
                # Transfer first: a combined device/dtype conversion can ask
                # MPS to perform the unsupported FP64 cast before transfer.
                gold_block = block.detach().cpu().to(dtype=torch.float64)
                gold_lp = torch.log_softmax(gold_block, dim=-1)
                gold_p = gold_lp.exp()
                gold_mu = (gold_p * gold_lp).sum(dim=-1)
                gold_var = (gold_p * (gold_lp - gold_mu.unsqueeze(-1)).square()).sum(dim=-1)
                gold_target = gold_lp.gather(-1, selected.cpu().unsqueeze(-1)).squeeze(-1)
                retain("cpu_float64_centered", gold_lp, gold_mu, gold_var, gold_target)

    compressed_bytes = len(zlib.compress(text.encode("utf-8"))) if text is not None else None
    selected_count = max(1, int((k_percent / 100.0) * len(targets)))
    result: dict[str, Any] = {}
    for name in variants:
        vectors = {key: torch.cat(parts) for key, parts in saved[name].items()}
        supported = bool(vectors["support"].all().item())
        scores = {key: None for key in ("loss", "mink", "minkpp", "zlib", "sum_logp")}
        mink_indices: list[int] = []
        minkpp_indices: list[int] = []
        aggregate_valid = False
        if supported:
            worst_lp = torch.topk(vectors["logp"], selected_count, largest=False)
            worst_z = torch.topk(vectors["z"], selected_count, largest=False)
            sum_logp = float(vectors["logp"].sum().item())
            scores = {
                "loss": float(vectors["logp"].mean().item()),
                "mink": float(worst_lp.values.mean().item()),
                "minkpp": float(worst_z.values.mean().item()),
                "sum_logp": sum_logp,
                "zlib": sum_logp / compressed_bytes if compressed_bytes is not None else None,
            }
            mink_indices = worst_lp.indices.detach().cpu().tolist()
            minkpp_indices = worst_z.indices.detach().cpu().tolist()
            aggregate_valid = all(value is None or math.isfinite(value) for value in scores.values())
            if not aggregate_valid:
                scores = {key: None for key in scores}
        result[name] = {
            "status": "complete" if supported and aggregate_valid else "invalid_aggregate" if supported else "invalid_support",
            "aggregate_valid": aggregate_valid,
            "scores": scores,
            "component_dtypes": component_dtypes[name],
            "variance_formula": "centered" if name.endswith("centered") else "second_moment_minus_mean_squared",
            "supported_target_count": int(vectors["support"].sum().item()),
            "clamped_target_count": int(vectors["clamp_mask"].sum().item()),
            "selected_token_count": selected_count,
            "mink_selected_indices": mink_indices,
            "minkpp_selected_indices": minkpp_indices,
            "per_token": {
                **{key: _vector_payload(vectors[key]) for key in ("logp", "mu", "variance", "sigma", "z")},
                "clamp_mask": vectors["clamp_mask"].cpu().tolist(),
                "support": vectors["support"].cpu().tolist(),
            },
        }
    pandora_results = {}
    pandora_selected_count = int((pandora_k_percent / 100.0) * len(targets)) if include_pandora else None
    for name, components in pandora_saved.items():
        vectors = {key: torch.cat(parts) for key, parts in components.items()}
        supported = bool(vectors["support"].all().item())
        score = feature = None
        indices = []
        status = "invalid_support" if not supported else "invalid_selection"
        if supported and pandora_selected_count:
            ordered, order = torch.sort(vectors["z"])
            # Pinned MinK.reduce uses sum/count, rather than Tensor.mean().
            reduced = ordered[:pandora_selected_count].sum() / pandora_selected_count
            high_member = float(reduced.item())
            if math.isfinite(high_member):
                score, feature = high_member, -high_member
                indices = order[:pandora_selected_count].detach().cpu().tolist()
                status = "complete"
            else:
                status = "invalid_aggregate"
        pandora_results[name] = {
            "status": status,
            "scores": {"pandora_minkxx": score, "pandora_minkxx_feature": feature},
            "score_orientation": "pandora_minkxx higher_is_member; pandora_minkxx_feature lower_is_member",
            "component_dtypes": pandora_dtypes[name],
            "supported_target_count": int(vectors["support"].sum().item()),
            "selected_token_count": pandora_selected_count,
            "selected_indices": indices,
            "per_token": {**{key: _vector_payload(vectors[key]) for key in ("logp", "mean", "std", "z")},
                          "support": vectors["support"].cpu().tolist()},
        }
    return {
        "variants": result,
        "selection": {"k_percent": k_percent, "rounding": "max(1,floor(ratio*target_count))", "selected_count": selected_count},
        "variance_floor": VARIANCE_FLOOR,
        "denominator_epsilon": DENOMINATOR_EPSILON,
        "statistics_chunk_tokens": chunk_tokens,
        "forward_logit_dtype": str(logits.dtype),
        "zlib_convention": "sum_scored_logp / zlib.compress(full_source_text_utf8, default_level=6)_bytes",
        "compressed_source_text_bytes": compressed_bytes,
        "pandora_comparators": pandora_results,
        "pandora_convention": {
            "enabled": include_pandora,
            "source": PANDORA_SOURCE,
            "normalization": "unweighted torch.std_mean(logp,dim=-1,correction=1); z=(target_logp-mean)/(std+1e-8)",
            "variance_clamp": None,
            "k_percent": pandora_k_percent if include_pandora else None,
            "selected_count": pandora_selected_count,
            "tail_rule": "floor(ratio*target_count), no minimum-one adjustment; zero selection is invalid",
            "aggregation": "torch.sort(z).values[:selected_count].sum()/selected_count; original feature is its negative",
            "scope": "Normalization/reduction comparators from the same engine forward logits, not whole-pipeline historical replication. The engine uses all-ones attention masks; Pandora defaults to input_ids>0 when callers supply no mask.",
        },
        "precision_interpretation": {
            "author_native_vs_fp32": "Changes log-softmax, probability, moment and score arithmetic precision; not reduction-only isolation.",
            "fp32_second_moment_vs_centered": "Same FP32 logp/probabilities/target support; changes variance formula.",
            "fixed_fp32_logp_bf16_reductions": "Same FP32 logp/probabilities/products and FP32 moment reductions, followed by explicit BF16 moment-result casts, BF16 variance/sqrt, unchanged FP32 target logp and FP32 z. This isolates moment-result quantization and subsequent arithmetic, not internal kernel accumulation. Diagnostic, not author-native scoring; v1 used sum(dtype=BF16) and also cast operands before reduction, so its fourth variant is superseded.",
        },
    }


def score_prepared(
    checkpoint: LocalCheckpoint, prepared: PreparedInput, *, chunk_tokens: int = 64,
    k_percent: float = 20.0, include_cpu_float64: bool = False,
    include_pandora: bool = False, pandora_k_percent: float = 10.0,
) -> dict[str, Any]:
    """One batch-one model forward, followed by shared-logit score variants."""
    import torch
    if len(prepared.input_ids) > checkpoint.context_limit:
        raise ValueError("Input exceeds model context; declare a prefix experiment explicitly.")
    if prepared.original_token_ids is not None:
        selected_source = prepared.original_token_ids[:len(prepared.input_ids) + 1]
        if (prepared.input_mode != "checkmiabench_original_tokens" or
                prepared.input_ids != selected_source[:-1] or
                prepared.target_ids != selected_source[1:]):
            raise ValueError("External targets must match the exact original causal source prefix.")
    elif prepared.target_ids != prepared.input_ids[1:1 + len(prepared.target_ids)]:
        raise ValueError("Prepared targets must be the contiguous causal input prefix after position zero.")
    vocab = int(checkpoint.model.config.vocab_size)
    if max(prepared.input_ids) >= vocab or max(prepared.target_ids) >= vocab:
        raise ValueError("Input or target ID exceeds the loaded model vocabulary.")
    _sync(checkpoint.device)
    started = time.perf_counter()
    tokens = torch.tensor([prepared.input_ids], dtype=torch.long, device=checkpoint.device)
    with torch.inference_mode():
        output = checkpoint.model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False)
        logits = output.logits
        if logits.ndim != 3 or logits.shape[:2] != tokens.shape:
            raise ValueError("Model did not return a complete batch-one sequence of logits.")
        _sync(checkpoint.device)
        forwarded = time.perf_counter()
        measurements = score_logits_variants(
            logits[0, :len(prepared.target_ids), :], prepared.target_ids,
            chunk_tokens=chunk_tokens, k_percent=k_percent, text=prepared.text,
            include_cpu_float64=include_cpu_float64, include_pandora=include_pandora,
            pandora_k_percent=pandora_k_percent,
        )
    _sync(checkpoint.device)
    finished = time.perf_counter()
    memory = _memory_snapshot(checkpoint.device)
    del output, logits, tokens
    return {
        "engine_version": ENGINE_VERSION,
        "author_source": AUTHOR_SOURCE,
        "id": prepared.identifier,
        "status": "complete" if all(value["status"] == "complete" for value in
                                     [*measurements["variants"].values(), *measurements["pandora_comparators"].values()]) else "invalid_support",
        "input_mode": prepared.input_mode,
        "input_convention": prepared.convention,
        "input_token_ids": list(prepared.input_ids),
        "original_token_ids": list(prepared.original_token_ids) if prepared.original_token_ids is not None else None,
        "input_token_ids_sha256": _digest_ids(prepared.input_ids),
        "target_token_ids": list(prepared.target_ids),
        "target_token_ids_sha256": _digest_ids(prepared.target_ids),
        "target_sequence_positions": list(range(1, len(prepared.target_ids) + 1)),
        "logit_positions": list(range(len(prepared.target_ids))),
        "source_text_sha256": hashlib.sha256(prepared.text.encode("utf-8")).hexdigest() if prepared.text is not None else None,
        "target_count": len(prepared.target_ids),
        "batch_size": 1,
        "attention_mask_convention": "all ones on every forwarded input position",
        "checkpoint_identity": checkpoint.identity,
        "timing": {"forward_seconds": forwarded - started, "statistics_and_serialization_seconds": finished - forwarded,
                   "total_seconds": finished - started},
        "memory": memory,
        **measurements,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing paths are rejected.")
    parser.add_argument("--input-kind", choices=("olmo_plaintext", "checkmiabench_tokens"), required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--model-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attention-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--chunk-tokens", type=int, default=64)
    parser.add_argument("--k-percent", type=float, default=20.)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--tokens-field", default="tokens")
    parser.add_argument("--cpu-float64-diagnostic", action="store_true")
    parser.add_argument("--include-pandora", action="store_true", help="Add explicitly named uniform-vocabulary same-forward comparators.")
    parser.add_argument("--pandora-k-percent", type=float, default=10.)
    parser.add_argument("--gzip-vectors", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output directory already exists.")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive.")
    # Merely importing this module or requesting --help never loads a model.
    project = Path(__file__).resolve().parent.parent
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    payload = args.input.read_bytes()
    records = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    selected = records if args.limit is None else records[:args.limit]
    if not selected:
        parser.error("Input contains no records.")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "engine_version": ENGINE_VERSION,
        "engine_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_file_sha256": hashlib.sha256(payload).hexdigest(),
        "input_records": len(records), "selected_records": len(selected),
        "settings": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")},
        "status": "running",
    }
    manifest_path = args.output / "execution.json"
    manifest_path.write_bytes(_json_bytes(manifest) + b"\n")
    count = 0
    score_path: Path | None = None
    started = time.perf_counter()
    try:
        checkpoint = load_local_checkpoint(args.model_path, device=args.device, model_dtype=args.model_dtype,
                                           attention_implementation=args.attention_implementation)
        filename = "scores.jsonl.gz" if args.gzip_vectors else "scores.jsonl"
        score_path = args.output / filename
        opener = gzip.open if args.gzip_vectors else open
        with opener(score_path, "xb") as handle:
            for index, record in enumerate(selected):
                identifier = str(record.get(args.id_field, f"row-{index}"))
                if args.input_kind == "olmo_plaintext":
                    prepared = prepare_olmo_plaintext(checkpoint.tokenizer, record[args.text_field], identifier,
                                                     max_tokens=args.max_tokens if args.max_tokens is not None else 4096)
                else:
                    prepared = prepare_checkmiabench_tokens(record[args.tokens_field], identifier,
                        text=record.get(args.text_field), max_tokens=args.max_tokens)
                result = score_prepared(checkpoint, prepared, chunk_tokens=args.chunk_tokens,
                    k_percent=args.k_percent, include_cpu_float64=args.cpu_float64_diagnostic,
                    include_pandora=args.include_pandora, pandora_k_percent=args.pandora_k_percent)
                result["record_index"] = index
                handle.write(_json_bytes(result) + b"\n")
                handle.flush()
                count += 1
                print(f"Scored {count}/{len(selected)}: {identifier}", flush=True)
                if result["status"] != "complete":
                    raise ValueError(f"Invalid score support for {identifier}; vectors retained.")
        manifest.update(status="complete", scored_records=count, wall_seconds=time.perf_counter() - started,
                        checkpoint_identity=checkpoint.identity, score_file=filename)
        print(f"Saved {count} real-input score records to {args.output}", flush=True)
        return 0
    except BaseException as error:
        manifest.update(status="failed", scored_records=count, wall_seconds=time.perf_counter() - started,
                        error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if score_path is not None and score_path.is_file():
            manifest.update(score_file=score_path.name, score_file_sha256=_digest_file(score_path),
                            score_file_bytes=score_path.stat().st_size)
        manifest_path.write_bytes(_json_bytes(manifest) + b"\n")


if __name__ == "__main__":
    raise SystemExit(main())
