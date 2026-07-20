"""Remote guided probes preserve text and scoring across batching and caches."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer

from contamcheckr.api_adapter import load_remote_model, UnicodeCodepointTokenizer
from contamcheckr.cache import ScoreCache, cache_key
from contamcheckr.methods.guided import (
    GENERATION_CACHE_VERSION,
    _generate_all,
    _truncate_to_units,
    run_guided_overlap,
    split_text,
)
from contamcheckr.model import LocalCausalLM, ModelCapabilities


@pytest.mark.parametrize("text", ["éx", "café東京", "a👩🏽‍💻b", "🧪漢e\u0301🙂"])
@pytest.mark.parametrize("ratio", [0.25, 0.5, 0.75])
def test_proxy_split_reconstructs_exact_unicode(text: str, ratio: float) -> None:
    result = split_text(UnicodeCodepointTokenizer(), text, ratio)
    assert result.prefix + result.suffix == text
    assert result.suffix_token_count == len(result.suffix)
    assert "�" not in result.prefix + result.suffix


def _remote(calls: list[dict]):
    def transport(method, url, headers, body):
        calls.append(body.copy())
        # Deliberately exceed the budget: transport stubs and cached older
        # outputs need the same normalization as an ordinary provider reply.
        return 200, {"choices": [{"text": "é🧪東京abcdef" * 10}]}

    return load_remote_model(
        "remote", base_url="https://example.invalid/v1", transport=transport,
    )


def _outputs(model, *, batch_size, cache=None):
    return _generate_all(
        model, ["prompt-a", "prompt-b", "prompt-c"], [3, 7, 4],
        batch_size=batch_size, cache=cache,
        cache_model_fingerprint="remote-test",
        cache_tokenizer_fingerprint="unicode-proxy-test",
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_completion_and_request_budgets_are_batch_invariant(batch_size: int) -> None:
    calls: list[dict] = []
    outputs = _outputs(_remote(calls), batch_size=batch_size)
    assert outputs == ["é🧪東", "é🧪東京abc", "é🧪東京"]
    assert [call["max_tokens"] for call in calls] == [3, 7, 4]


def test_singleton_is_truncated_like_a_batch() -> None:
    model = _remote([])
    common = {
        "batch_size": 4, "cache": None,
        "cache_model_fingerprint": "", "cache_tokenizer_fingerprint": "",
    }
    singleton = _generate_all(model, ["a"], [3], **common)
    batch = _generate_all(model, ["a", "b"], [3, 3], **common)
    assert singleton == batch[:1] == ["é🧪東"]


def test_cache_hits_and_misses_share_completion_truncation(tmp_path) -> None:
    calls: list[dict] = []
    model = _remote(calls)
    with ScoreCache(tmp_path / "remote.sqlite3") as cache:
        key = cache_key(GENERATION_CACHE_VERSION, "remote-test", "unicode-proxy-test", "prompt-a", 3)
        cache.put_generation(key, "é🧪東京abcdef" * 10)
        mixed = _outputs(model, batch_size=4, cache=cache)
        assert len(calls) == 2
        calls.clear()
        fully_cached = _outputs(model, batch_size=1, cache=cache)
        assert calls == []
    assert fully_cached == mixed == ["é🧪東", "é🧪東京abc", "é🧪東京"]


def test_old_generation_cache_semantics_are_invalidated(tmp_path) -> None:
    calls: list[dict] = []
    with ScoreCache(tmp_path / "old.sqlite3") as cache:
        old_key = cache_key("remote-test", "unicode-proxy-test", "prompt-a", 3)
        cache.put_generation(old_key, "invalid old result")
        outputs = _outputs(_remote(calls), batch_size=4, cache=cache)
    assert len(calls) == 3
    assert outputs == ["é🧪東", "é🧪東京abc", "é🧪東京"]


def test_guided_remote_scores_and_references_are_batch_invariant() -> None:
    examples = [
        {"id": "a", "text": "éx"},
        {"id": "b", "text": "Unicode café 東京🧪 test"},
        {"id": "c", "text": "a👩🏽‍💻b"},
    ]
    common = {"dataset_name": "unicode", "split_name": "test", "seed": 17}
    single = run_guided_overlap(
        _remote([]), examples, generation_batch_size=1, **common,
    )
    batched = run_guided_overlap(
        _remote([]), examples, generation_batch_size=4, **common,
    )
    assert single == batched
    for item in single["per_example"]:
        assert "�" not in item["reference"]
        assert "�" not in item["guided_output"]


class _ContextTokenizer:
    """A tokenizer that exercises LocalCausalLM's real context reservation."""

    padding_side = "right"
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        return "".join(chr(int(value)) for value in ids if int(value) != 0)

    def batch_decode(self, rows, **kwargs):
        return [self.decode(row, **kwargs) for row in rows]

    def __call__(self, texts, *, max_length, **kwargs):
        rows = [self.encode(text)[:max_length] for text in texts]
        width = max(map(len, rows))
        return {
            "input_ids": torch.tensor([[0] * (width - len(row)) + row for row in rows]),
            "attention_mask": torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in rows]),
        }


def _context_model(calls: list[tuple[int, int]]) -> LocalCausalLM:
    class Generator:
        def generate(self, input_ids, *, max_new_tokens, **kwargs):
            calls.append((len(input_ids), max_new_tokens))
            # Depend on the actual truncated prompt. An oversized batch
            # budget changes the first generated token in the broken path.
            suffix = torch.tensor([
                [65 + int(row.sum()) % 26] * max_new_tokens for row in input_ids
            ])
            return torch.cat([input_ids, suffix], dim=1)

    return LocalCausalLM(
        model_id="offline-context-model", tokenizer=_ContextTokenizer(), model=Generator(),
        device=torch.device("cpu"), dtype=torch.float32,
        capabilities=ModelCapabilities(True, False, False),
        fingerprint="offline-context-model", tokenizer_fingerprint="context-tokenizer",
        context_limit=32,
    )


def test_local_context_is_invariant_under_mixed_budgets_and_partial_cache(tmp_path) -> None:
    prompts = ["first prompt " * 8, "second prompt " * 8, "third prompt " * 8]
    budgets = [3, 7, 3]

    def generate(model, batch_size, cache=None):
        return _generate_all(
            model, prompts, budgets, batch_size=batch_size, cache=cache,
            cache_model_fingerprint="context-model", cache_tokenizer_fingerprint="context-tokenizer",
        )

    singleton = generate(_context_model([]), 1)
    calls: list[tuple[int, int]] = []
    batched = generate(_context_model(calls), 4)
    assert singleton == batched
    assert calls == [(2, 3), (1, 7)]  # Equal budgets retain local batching.
    with ScoreCache(tmp_path / "local.sqlite3") as cache:
        key = cache_key(GENERATION_CACHE_VERSION, "context-model", "context-tokenizer", prompts[0], budgets[0])
        cache.put_generation(key, singleton[0])
        assert generate(_context_model([]), 4, cache) == singleton
        cached_calls: list[tuple[int, int]] = []
        assert generate(_context_model(cached_calls), 1, cache) == singleton
        assert cached_calls == []


@pytest.fixture(scope="module")
def byte_bpe_tokenizer():
    source = Path(__file__).resolve().parent.parent / "experiments" / "models" / "sshleifer--tiny-gpt2"
    candidates = [str(source), "sshleifer/tiny-gpt2"] if source.exists() else ["sshleifer/tiny-gpt2"]
    for candidate in candidates:
        try:
            return AutoTokenizer.from_pretrained(candidate, local_files_only=True)
        except (OSError, ValueError, RuntimeError):
            continue
    pytest.skip("No cached byte-BPE tokenizer; no download attempted.")


@pytest.mark.parametrize("text", ["a🙂bb", "café 東京 🧪 done", "汉字🙂e\u0301 끝", " Hello  world , café"])
def test_cached_byte_bpe_split_reconstructs_original(byte_bpe_tokenizer, text: str) -> None:
    result = split_text(byte_bpe_tokenizer, text)
    assert result.prefix + result.suffix == text
    assert "�" not in result.prefix + result.suffix


def test_cached_byte_bpe_truncation_never_introduces_partial_unicode(byte_bpe_tokenizer) -> None:
    text = "a🙂bb"
    assert _truncate_to_units(byte_bpe_tokenizer, text, 2) == "a"
    assert _truncate_to_units(byte_bpe_tokenizer, text, 3) == "a🙂"
    assert _truncate_to_units(byte_bpe_tokenizer, "🙂", 1) == ""


def test_unavoidable_byte_bpe_split_is_explicitly_declined(byte_bpe_tokenizer) -> None:
    assert len(byte_bpe_tokenizer.encode("🙂", add_special_tokens=False)) > 1
    with pytest.raises(ValueError, match="lossless tokenizer boundary"):
        split_text(byte_bpe_tokenizer, "🙂")
    model = SimpleNamespace(tokenizer=byte_bpe_tokenizer, generate=lambda *args: "text")
    result = run_guided_overlap(
        model, [{"id": "unsafe", "text": "🙂"}, {"id": "safe", "text": "a🙂bb"}],
        dataset_name="unicode", split_name="test",
    )
    assert result["skipped_examples"] == [{
        "id": "unsafe", "reason": "Text has no lossless tokenizer boundary for guided scoring.",
    }]
    assert result["per_example"][0]["id"] == "safe"


def test_nonreversible_tokenizer_is_explicitly_declined() -> None:
    class LowercaseTokenizer(UnicodeCodepointTokenizer):
        def decode(self, ids, **kwargs):
            return super().decode(ids).lower()

    with pytest.raises(ValueError, match="lossless tokenizer reconstruction"):
        split_text(LowercaseTokenizer(), "UPPER CASE text")


def test_unencodable_completion_is_declined_instead_of_ignoring_budget() -> None:
    def encode(*args, **kwargs):
        raise ValueError("cannot encode")

    with pytest.raises(ValueError, match="cannot be encoded for lossless truncation"):
        _truncate_to_units(SimpleNamespace(encode=encode), "completion", 2)
