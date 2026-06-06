from pathlib import Path

import pytest
import torch

from contamcheck.model import ModelCapabilities
from contamcheck.web import (
    DEFAULT_FORM,
    WebState,
    execute_audit,
    form_to_config,
    render_dashboard,
    run_server,
)


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


def test_form_to_config_uses_local_defaults() -> None:
    config = form_to_config(DEFAULT_FORM)
    assert config.model_id == "sshleifer/tiny-gpt2"
    assert config.benchmark_source == "sample"
    assert config.limit == 10
    assert config.device == "auto"


def test_form_to_config_rejects_bad_number() -> None:
    fields = DEFAULT_FORM | {"limit": "many"}
    with pytest.raises(ValueError, match="Limit must be an integer"):
        form_to_config(fields)


def test_dashboard_renders_form_and_escapes_values() -> None:
    page = render_dashboard(form=DEFAULT_FORM | {"model": "<unsafe>"})
    assert 'action="/run"' in page
    assert "Run contamination check" in page
    assert "&lt;unsafe&gt;" in page
    assert "<unsafe>" not in page


def test_execute_audit_writes_artifacts(tmp_path: Path) -> None:
    state = WebState(tmp_path)
    fields = DEFAULT_FORM | {"model": "fake", "limit": "10"}
    report = execute_audit(
        state,
        fields,
        model_loader=lambda *args, **kwargs: FakeModel(),
    )
    assert report["example_count"] == 10
    assert state.json_path.exists()
    assert state.csv_path.exists()
    assert state.html_path.exists()
    page = render_dashboard(form=fields, report=report)
    assert "Latest audit" in page
    assert "/artifacts/latest.json" in page
    assert report["band"].title() in page
    for caveat in report["caveats"]:
        assert caveat in page


def test_server_rejects_non_loopback_host(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="localhost-only"):
        run_server("0.0.0.0", 8000, tmp_path)
