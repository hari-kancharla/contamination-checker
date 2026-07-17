from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import random
from typing import Any, Callable, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .hashing import (
    local_model_signature,
    model_fingerprint,
    resolved_commit_hash,
    tokenizer_fingerprint,
)
from .methods.token_stats import TokenStats, token_stats_from_logits


ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class ModelCapabilities:
    supports_generation: bool
    supports_logprobs: bool
    supports_full_distribution: bool

    def as_dict(self) -> dict[str, bool]:
        return {
            "supports_generation": self.supports_generation,
            "supports_logprobs": self.supports_logprobs,
            "supports_full_distribution": self.supports_full_distribution,
        }


@dataclass
class LocalCausalLM:
    model_id: str
    tokenizer: Any
    model: Any
    device: torch.device
    dtype: torch.dtype
    capabilities: ModelCapabilities
    fingerprint: str
    tokenizer_fingerprint: str
    revision: str | None = None
    context_limit: int | None = None
    parameter_count: int | None = None
    resolved_revision: str | None = None
    tokenizer_revision: str | None = None
    provenance: dict[str, Any] | None = None

    def count_tokens(self, text: str) -> int:
        """Count tokens without special tokens (for truncation diagnostics)."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    # -- prompt formatting -------------------------------------------------

    def format_prompt(self, prompt: str, *, chat_template: bool = False) -> str:
        """Optionally wrap a prompt in the tokenizer's chat template."""
        if not chat_template:
            return prompt
        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        if not callable(apply_template):
            return prompt
        try:
            return str(
                apply_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except (TypeError, ValueError, AttributeError):
            return prompt

    # -- generation --------------------------------------------------------

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        return self.generate_batch(
            [prompt], max_new_tokens, batch_size=1
        )[0]

    def generate_batch(
        self,
        prompts: Sequence[str],
        max_new_tokens: int,
        *,
        batch_size: int = 4,
    ) -> list[str]:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least one.")
        if not prompts:
            return []
        original_padding_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"
        results: list[str] = []
        try:
            for start in range(0, len(prompts), max(1, batch_size)):
                batch = list(prompts[start : start + max(1, batch_size)])
                encoded = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self._generation_context(max_new_tokens),
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                with torch.inference_mode():
                    output = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                generated = output[:, input_ids.shape[1] :]
                decoded = self.tokenizer.batch_decode(
                    generated, skip_special_tokens=True
                )
                results.extend(str(value) for value in decoded)
        finally:
            self.tokenizer.padding_side = original_padding_side
        return results

    def _generation_context(self, max_new_tokens: int) -> int:
        if self.context_limit is None:
            return max_new_tokens + 1024
        return max(8, self.context_limit - max_new_tokens)

    # -- token statistics --------------------------------------------------

    def token_stats(
        self,
        text: str,
        *,
        max_tokens: int = 512,
    ) -> TokenStats:
        return self.token_stats_batch(
            [text], max_tokens=max_tokens, batch_size=1
        )[0]

    def token_stats_batch(
        self,
        texts: Sequence[str],
        *,
        max_tokens: int = 512,
        batch_size: int = 4,
        progress: ProgressCallback | None = None,
        progress_label: str = "Scoring tokens",
    ) -> list[TokenStats]:
        if max_tokens < 2:
            raise ValueError("max_tokens must be at least two.")
        if not texts:
            return []
        effective_batch = self._effective_batch_size(batch_size, max_tokens)
        results: list[TokenStats] = []
        total = len(texts)
        for start in range(0, total, effective_batch):
            batch = list(texts[start : start + effective_batch])
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_tokens,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            attention_mask = attention_mask.to(self.device)
            with torch.inference_mode():
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
            for row_index, _ in enumerate(batch):
                length = int(attention_mask[row_index].sum().item())
                if length < 2:
                    raise ValueError(
                        "Text must produce at least two tokens for scoring."
                    )
                row_logits = logits[row_index, : length - 1, :]
                row_targets = input_ids[row_index, 1:length]
                truncated = length >= max_tokens
                results.append(
                    token_stats_from_logits(
                        row_logits,
                        row_targets,
                        truncated=truncated,
                    )
                )
            del logits
            if progress is not None:
                progress(progress_label, min(start + len(batch), total), total)
        return results

    # -- sequence likelihood ----------------------------------------------

    def sequence_loglikelihood(
        self,
        text: str,
        *,
        max_tokens: int = 512,
    ) -> float:
        return self.sequence_loglikelihood_batch(
            [text], max_tokens=max_tokens, batch_size=1
        )[0]

    def sequence_loglikelihood_batch(
        self,
        texts: Sequence[str],
        *,
        max_tokens: int = 512,
        batch_size: int = 8,
        progress: ProgressCallback | None = None,
        progress_label: str = "Scoring sequences",
    ) -> list[float]:
        if max_tokens < 2:
            raise ValueError("max_tokens must be at least two.")
        if not texts:
            return []
        effective_batch = self._effective_batch_size(batch_size, max_tokens)
        results: list[float] = []
        total = len(texts)
        for start in range(0, total, effective_batch):
            batch = list(texts[start : start + effective_batch])
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_tokens,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            attention_mask = attention_mask.to(self.device)
            with torch.inference_mode():
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
                shifted_logits = logits[:, :-1, :].float()
                targets = input_ids[:, 1:]
                log_probs = torch.log_softmax(shifted_logits, dim=-1)
                selected = log_probs.gather(
                    2, targets.unsqueeze(-1)
                ).squeeze(-1)
                # Sum in float64. A float32 sum near -700 has a resolution of
                # about 1e-4, and the checked-in order test treated smaller
                # differences as permutation wins. MPS has no float64
                # tensors, so accumulate on CPU, which is cheap at this size.
                selected_cpu = selected.detach().cpu().double()
                mask = attention_mask[:, 1:].detach().cpu().double()
                totals = torch.sum(selected_cpu * mask, dim=1)
            results.extend(float(value) for value in totals.tolist())
            del logits, shifted_logits, log_probs
            if progress is not None:
                progress(progress_label, min(start + len(batch), total), total)
        return results

    def _effective_batch_size(self, requested: int, max_tokens: int) -> int:
        if requested < 1:
            raise ValueError("Batch size must be at least one.")
        vocabulary_size = int(
            getattr(getattr(self.model, "config", None), "vocab_size", 0) or 0
        )
        if vocabulary_size <= 0:
            return requested
        bytes_per_example = max(1, max_tokens * vocabulary_size * 4)
        budget = 512_000_000 if self.device.type == "mps" else 1_000_000_000
        safe = max(1, budget // bytes_per_example)
        return min(requested, safe)

    # -- compatibility -----------------------------------------------------

    def token_logprob_table(
        self, text: str, *, max_tokens: int = 512
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return shifted logits and target ids for one text."""
        if not text.strip():
            raise ValueError("Text must contain at least one non-whitespace character.")
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
        )
        input_ids = encoded["input_ids"].to(self.device)
        if input_ids.shape[1] < 2:
            raise ValueError("Text must produce at least two tokens.")
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.inference_mode():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
        return logits[0, :-1].float().cpu(), input_ids[0, 1:].cpu()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        if resolved.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available.")
        return resolved
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(device: torch.device, dtype: str) -> torch.dtype:
    if dtype != "auto":
        choices = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in choices:
            raise ValueError(f"Unsupported dtype: {dtype}.")
        return choices[dtype]
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def probe_capabilities(model: Any) -> ModelCapabilities:
    supports_generation = callable(getattr(model, "generate", None))
    supports_logprobs = callable(getattr(model, "forward", None))
    vocabulary_size = getattr(getattr(model, "config", None), "vocab_size", None)
    supports_full_distribution = (
        supports_logprobs
        and isinstance(vocabulary_size, int)
        and vocabulary_size > 0
    )
    return ModelCapabilities(
        supports_generation=supports_generation,
        supports_logprobs=supports_logprobs,
        supports_full_distribution=supports_full_distribution,
    )


def _transformers_version() -> str:
    try:
        return importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


def _infer_context_limit(model: Any, tokenizer: Any) -> int | None:
    candidates = [
        getattr(getattr(model, "config", None), "max_position_embeddings", None),
        getattr(getattr(model, "config", None), "n_positions", None),
        getattr(tokenizer, "model_max_length", None),
    ]
    values = [
        int(value)
        for value in candidates
        if isinstance(value, int) and 1 < value < 10_000_000
    ]
    return min(values) if values else None


def load_model(
    model_id: str,
    device: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    seed: int = 0,
    local_files_only: bool = False,
    revision: str | None = None,
    progress: ProgressCallback | None = None,
) -> LocalCausalLM:
    if not model_id.strip():
        raise ValueError("Model identifier cannot be empty.")
    set_seed(seed)
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(resolved_device, dtype)
    if progress is not None:
        progress("Loading tokenizer", 0, 1)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if progress is not None:
        progress("Loading model", 0, 1)
    load_kwargs: dict[str, Any] = {
        "revision": revision,
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=resolved_dtype,
            **load_kwargs,
        )
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=resolved_dtype,
            **load_kwargs,
        )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    model.to(resolved_device)
    model.eval()
    capabilities = probe_capabilities(model)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    resolved_revision = resolved_commit_hash(model)
    tokenizer_revision = resolved_commit_hash(tokenizer)
    local_signature = local_model_signature(model_id)
    provenance = {
        "requested_revision": revision,
        "resolved_model_commit": resolved_revision,
        "resolved_tokenizer_commit": tokenizer_revision,
        "weight_identity": (
            "local_metadata_signature"
            if local_signature is not None
            else "hub_commit_reference" if resolved_revision else "unresolved"
        ),
        "weight_sha256": None,
        "local_signature": local_signature,
        "note": (
            "Local weights are identified by file sizes and modification times; "
            "weight contents were not cryptographically hashed."
            if local_signature is not None
            else "Resolved Hub commits are references supplied by the loader; "
            "loaded weight contents were not cryptographically hashed."
            if resolved_revision
            else "The loader did not expose a resolved model commit. The requested "
            "model name/revision does not identify immutable weight contents."
        ),
    }
    fingerprint = model_fingerprint(
        model_id=model_id,
        revision=revision,
        dtype=str(resolved_dtype),
        tokenizer=tokenizer,
        parameter_count=parameter_count,
        transformers_version=_transformers_version(),
        resolved_revision=resolved_revision,
        tokenizer_revision=tokenizer_revision,
    )
    if progress is not None:
        progress("Model ready", 1, 1)
    return LocalCausalLM(
        model_id=model_id,
        tokenizer=tokenizer,
        model=model,
        device=resolved_device,
        dtype=resolved_dtype,
        capabilities=capabilities,
        fingerprint=fingerprint,
        tokenizer_fingerprint=tokenizer_fingerprint(tokenizer),
        revision=revision,
        context_limit=_infer_context_limit(model, tokenizer),
        parameter_count=parameter_count,
        resolved_revision=resolved_revision,
        tokenizer_revision=tokenizer_revision,
        provenance=provenance,
    )
