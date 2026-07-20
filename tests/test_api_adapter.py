"""Tests for the black-box API adapter (fully offline stub transport)."""

from __future__ import annotations

import io
import urllib.error

import pytest

from contamcheckr.api_adapter import (
    ByteTokenizer,
    UnicodeCodepointTokenizer,
    load_remote_model,
)
from contamcheckr.benchmarks import load_benchmark
from contamcheckr.config import RunConfig
from contamcheckr.report import build_report


def _stub_transport(calls: list[dict], completion: str = "plain deterministic completion", status: int = 200):
    def transport(method, url, headers, body):
        calls.append({"url": url, "body": body, "headers": headers})
        if status != 200:
            return status, {"error": {"message": "boom"}}
        return 200, {"choices": [{"text": completion}]}

    return transport


def test_adapter_generate_parses_choices() -> None:
    calls: list[dict] = []
    model = load_remote_model(
        "remote", base_url="https://api.example/v1", transport=_stub_transport(calls)
    )
    output = model.generate("Hello", 5)
    assert output == "plain deterministic completion"
    assert calls[0]["url"] == "https://api.example/v1/completions"
    assert calls[0]["body"]["temperature"] == 0.0
    assert calls[0]["body"]["max_tokens"] == 5


def test_adapter_batch_and_api_key_header(monkeypatch) -> None:
    monkeypatch.setenv("MY_KEY_ENV", "secret-123")
    calls: list[dict] = []
    model = load_remote_model(
        "remote",
        base_url="https://api.example/v1",
        api_key_env="MY_KEY_ENV",
        transport=_stub_transport(calls),
        model_name="gpt-x",
    )
    assert model.generate_batch(["a", "b"], 3) == [
        "plain deterministic completion",
        "plain deterministic completion",
    ]
    assert calls[0]["headers"]["Authorization"] == "Bearer secret-123"
    assert calls[0]["body"]["model"] == "gpt-x"


def test_adapter_http_error_raises() -> None:
    model = load_remote_model(
        "remote",
        base_url="https://api.example/v1",
        transport=_stub_transport([], status=429),
    )
    with pytest.raises(RuntimeError, match="HTTP 429"):
        model.generate("Hello", 5)


def test_byte_tokenizer_roundtrip() -> None:
    tokenizer = ByteTokenizer()
    assert tokenizer.decode(tokenizer.encode("hello")) == "hello"
    assert tokenizer.decode(tokenizer.encode("")) == ""


@pytest.mark.parametrize("text", ["éx", "café東京", "🧪👩🏽‍💻e\u0301", ""])
def test_proxy_tokenizer_preserves_unicode_and_declares_units(text: str) -> None:
    tokenizer = UnicodeCodepointTokenizer()
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert len(tokenizer.encode(text)) == len(text)
    assert tokenizer.is_proxy is True
    assert tokenizer.unit == "unicode_codepoint"


@pytest.mark.parametrize(
    "payload",
    [None, "secret-prompt", ["secret-prompt"], {}, {"error": "secret-prompt"},
     {"error": {"message": "secret-prompt"}}],
)
def test_http_error_payload_shapes_are_safe(payload) -> None:
    model = load_remote_model(
        "remote", base_url="https://api.example/v1",
        transport=lambda *args: (502, payload),
    )
    with pytest.raises(RuntimeError, match="HTTP 502") as captured:
        model.generate("private-prompt", 3)
    assert "secret-prompt" not in str(captured.value)
    assert "private-prompt" not in str(captured.value)


@pytest.mark.parametrize(
    "payload",
    [None, "secret-prompt", [], {}, {"choices": []}, {"choices": {}},
     {"choices": [None]}, {"choices": [{"text": None}]},
     {"choices": [{"text": {"secret": "secret-prompt"}}]}],
)
def test_malformed_success_payload_is_rejected_without_echo(payload) -> None:
    model = load_remote_model(
        "remote", base_url="https://api.example/v1",
        transport=lambda *args: (200, payload),
    )
    with pytest.raises(RuntimeError, match="Unexpected completion API reply") as captured:
        model.generate("private-prompt", 3)
    assert "secret-prompt" not in str(captured.value)
    assert "private-prompt" not in str(captured.value)


@pytest.mark.parametrize("status", [None, "502", 200.0, {"secret": "private"}])
def test_invalid_status_is_rejected_without_echo(status) -> None:
    model = load_remote_model(
        "remote", base_url="https://api.example/v1",
        transport=lambda *args: (status, {"choices": [{"text": "private"}]}),
    )
    with pytest.raises(RuntimeError, match="invalid HTTP status") as captured:
        model.generate("private-prompt", 3)
    assert "private" not in str(captured.value)


def test_empty_completion_is_valid() -> None:
    model = load_remote_model(
        "remote", base_url="https://api.example/v1",
        transport=_stub_transport([], completion=""),
    )
    assert model.generate("prompt", 3) == ""


def test_transport_failure_does_not_echo_url_or_credentials() -> None:
    def transport(*args):
        raise urllib.error.URLError("https://secret-token@host/private-prompt")

    model = load_remote_model(
        "remote", base_url="https://api.example/v1", transport=transport,
    )
    with pytest.raises(RuntimeError, match="request failed") as captured:
        model.generate("private-prompt", 3)
    assert "secret-token" not in str(captured.value)
    assert "private-prompt" not in str(captured.value)


def test_invalid_json_response_is_safe(monkeypatch) -> None:
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b"private-prompt secret-token not-json"

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    model = load_remote_model("remote", base_url="https://api.example/v1")
    with pytest.raises(RuntimeError, match="invalid JSON") as captured:
        model.generate("private-prompt", 3)
    assert "private-prompt" not in str(captured.value)
    assert "secret-token" not in str(captured.value)


def test_default_transport_http_error_does_not_echo_body(monkeypatch) -> None:
    def urlopen(*args, **kwargs):
        raise urllib.error.HTTPError(
            "https://secret-token@host", 429, "private-prompt", {},
            io.BytesIO(b"secret-token private-prompt"),
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    model = load_remote_model("remote", base_url="https://api.example/v1")
    with pytest.raises(RuntimeError, match="HTTP 429") as captured:
        model.generate("private-prompt", 3)
    assert "private-prompt" not in str(captured.value)
    assert "secret-token" not in str(captured.value)


def test_remote_metadata_discloses_proxy_and_endpoint_limits() -> None:
    model = load_remote_model("remote", base_url="https://api.example/v1")
    metadata = model.generation_metadata
    assert metadata["split_unit"] == "unicode_codepoint"
    assert metadata["provider_tokenizer_known"] is False
    assert metadata["provider_token_counts_known"] is False
    assert metadata["request_batching"] == "sequential_per_prompt"
    assert "not provider tokens" in " ".join(metadata["caveats"])


def test_black_box_report_skips_likelihood_and_order() -> None:
    calls: list[dict] = []
    # Completion echoes a rubric-free constant; guided gap is zero.
    model = load_remote_model(
        "remote", base_url="https://api.example/v1", transport=_stub_transport(calls)
    )
    benchmark = load_benchmark("sample", limit=6)
    report = build_report(
        model,
        benchmark,
        None,
        RunConfig(model_id="remote", benchmark_source="sample", limit=6),
    )
    assert report["capabilities"] == {
        "supports_generation": True,
        "supports_logprobs": False,
        "supports_full_distribution": False,
    }
    reasons = {item["method"]: item["reason"] for item in report["skipped_methods"]}
    assert "minkpp" in reasons
    assert "canonical_order" in reasons
    assert "guided" not in reasons
    assert isinstance(report["refutation_panel"], list)
