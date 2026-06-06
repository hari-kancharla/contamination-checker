from pathlib import Path

import pytest
import torch

from contamcheck.benchmarks import load_benchmark
from contamcheck.config import RunConfig
from contamcheck.html_report import build_html
from contamcheck.model import ModelCapabilities, load_model
from contamcheck.report import build_report, format_summary


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) % 32 for character in text]

    def decode(
        self, token_ids: list[int], skip_special_tokens: bool = True
    ) -> str:
        return "".join(chr(token_id + 96) for token_id in token_ids)


class FakeModel:
    tokenizer = CharacterTokenizer()
    capabilities = ModelCapabilities(True, True, True)

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        return "plain deterministic completion"

    def token_logprob_table(
        self, text: str, *, max_tokens: int = 512
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = min(max(2, len(text) // 8), max_tokens)
        targets = torch.tensor(
            [(ord(character) % 5) for character in text[:count]],
            dtype=torch.long,
        )
        if len(targets) < count:
            targets = torch.cat(
                [targets, torch.zeros(count - len(targets), dtype=torch.long)]
            )
        rows = torch.arange(count, dtype=torch.float32).unsqueeze(1)
        columns = torch.arange(5, dtype=torch.float32).unsqueeze(0)
        logits = -torch.abs(columns - ((rows + len(text)) % 5))
        return logits, targets

    def sequence_loglikelihood(self, text: str, *, max_tokens: int = 512) -> float:
        parts = text.split("\n\n---\n\n")
        return -float(
            sum(
                left > right
                for index, left in enumerate(parts)
                for right in parts[index + 1 :]
            )
        )


def _config(control: bool) -> RunConfig:
    return RunConfig(
        model_id="fake",
        benchmark_source="sample",
        control_source="sample" if control else None,
        limit=10,
        shard_count=2,
        permutation_count=5,
    )


def test_full_flow_is_reproducible_with_and_without_control() -> None:
    benchmark = load_benchmark("sample", limit=10)
    control = load_benchmark("sample", limit=10, sample_control=True)
    model = FakeModel()

    without = build_report(model, benchmark, None, _config(False))
    repeated = build_report(model, benchmark, None, _config(False))
    with_control = build_report(model, benchmark, control, _config(True))

    required = {
        "contamination_index",
        "confidence_interval",
        "index_basis",
        "per_example",
        "capabilities",
        "caveats",
    }
    assert required <= without.keys()
    assert 0.0 <= without["contamination_index"] <= 1.0
    assert without["index_basis"] == "guided_plus_order"
    assert with_control["index_basis"] == "combined"
    assert without["contamination_index"] == repeated["contamination_index"]
    assert without["confidence_interval"] == repeated["confidence_interval"]
    html = build_html(with_control)
    assert "<!doctype html>" in html
    assert "Benchmark contamination evidence" in html
    summary = format_summary(with_control)
    assert f"Band: {with_control['band']}" in summary
    for caveat in with_control["caveats"]:
        assert caveat in summary


def test_local_file_with_wrong_text_field_has_readable_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "x", "body": "text"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Available fields"):
        load_benchmark(str(path), text_field="content")


def test_local_file_requires_an_id(tmp_path: Path) -> None:
    path = tmp_path / "missing-id.jsonl"
    path.write_text('{"text": "plain text"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="required 'id'"):
        load_benchmark(str(path))


def test_cached_huggingface_model_full_flow() -> None:
    try:
        model = load_model(
            "sshleifer/tiny-gpt2",
            device="cpu",
            local_files_only=True,
            seed=11,
        )
    except (OSError, RuntimeError, ValueError) as error:
        pytest.skip(f"sshleifer/tiny-gpt2 is not cached: {error}")
    benchmark = load_benchmark("sample", limit=10)
    control = load_benchmark("sample", limit=10, sample_control=True)
    without_config = RunConfig(
        model_id="sshleifer/tiny-gpt2",
        benchmark_source="sample",
        limit=10,
        shard_count=2,
        permutation_count=2,
        seed=11,
    )
    with_config = RunConfig(
        model_id="sshleifer/tiny-gpt2",
        benchmark_source="sample",
        control_source="sample",
        limit=10,
        shard_count=2,
        permutation_count=2,
        seed=11,
    )
    first = build_report(model, benchmark, None, without_config)
    second = build_report(model, benchmark, None, without_config)
    controlled = build_report(model, benchmark, control, with_config)
    assert first["index_basis"] == "guided_plus_order"
    assert controlled["index_basis"] == "combined"
    assert first["contamination_index"] == second["contamination_index"]
    assert first["confidence_interval"] == second["confidence_interval"]
