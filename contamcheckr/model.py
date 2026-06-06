from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    capabilities: ModelCapabilities

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        context_limit = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(context_limit, int) and context_limit > 1:
            max_new_tokens = min(max_new_tokens, context_limit - 1)
            max_input_tokens = context_limit - max_new_tokens
            if input_ids.shape[1] > max_input_tokens:
                input_ids = input_ids[:, -max_input_tokens:]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, -max_input_tokens:]
        input_ids = input_ids.to(self.device)
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
        generated = output[0, input_ids.shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def token_logprob_table(
        self, text: str, *, max_tokens: int = 512
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def sequence_loglikelihood(self, text: str, *, max_tokens: int = 512) -> float:
        logits, targets = self.token_logprob_table(text, max_tokens=max_tokens)
        log_probs = torch.log_softmax(logits, dim=-1)
        selected = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        value = float(selected.sum().item())
        if not np.isfinite(value):
            raise ValueError("Sequence log-likelihood was not finite.")
        return value


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


def load_model(
    model_id: str,
    device: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    seed: int = 0,
    local_files_only: bool = False,
) -> LocalCausalLM:
    if not model_id.strip():
        raise ValueError("Model identifier cannot be empty.")
    set_seed(seed)
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(resolved_device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        dtype=resolved_dtype,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    model.to(resolved_device)
    model.eval()
    capabilities = probe_capabilities(model)
    return LocalCausalLM(
        model_id=model_id,
        tokenizer=tokenizer,
        model=model,
        device=resolved_device,
        capabilities=capabilities,
    )
