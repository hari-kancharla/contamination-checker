"""Crash-resistance stress suite for :func:`contamcheckr.report.build_report`.

Everything here runs against a local deterministic ``FakeModel`` (the same
pattern used by ``tests/test_end_to_end.py``) so the whole file stays fast
and fully offline -- no model downloads, no tokenizer files.

Purpose: prove that ``build_report`` never crashes on small, degenerate,
duplicate, oversized, or non-ASCII inputs, and that mid-scoring model
failures degrade to recorded skips instead of aborting the audit.
Known crash bugs are documented as ``xfail`` tests, not hidden.
"""

import pytest
import torch

from contamcheckr.benchmarks import BenchmarkData
from contamcheckr.config import RunConfig
from contamcheckr.model import ModelCapabilities
from contamcheckr.report import build_report

KNOWN_BANDS = {
    "insufficient evidence",
    "low evidence",
    "moderate evidence",
    "high evidence",
}


# ---------------------------------------------------------------------------
# Fake models (copied/adapted from tests/test_end_to_end.py)
# ---------------------------------------------------------------------------


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text]

    def decode(
        self, token_ids: list[int], skip_special_tokens: bool = True
    ) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


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


class StrictFakeModel(FakeModel):
    """FakeModel that mirrors the real wrapper's short-text failures.

    ``LocalCausalLM.token_logprob_table`` raises ``ValueError`` for
    whitespace-only text and for text that tokenizes to fewer than two
    tokens; the base ``FakeModel`` above silently pads so it never
    reproduces that production behaviour.  This subclass does.
    """

    def token_logprob_table(
        self, text: str, *, max_tokens: int = 512
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not text.strip():
            raise ValueError(
                "Text must contain at least one non-whitespace character."
            )
        if len(self.tokenizer.encode(text, add_special_tokens=False)) < 2:
            raise ValueError("Text must produce at least two tokens.")
        return super().token_logprob_table(text, max_tokens=max_tokens)


class PartiallyFailingModel(FakeModel):
    """Raises while scoring one specific text, otherwise healthy."""

    def token_logprob_table(
        self, text: str, *, max_tokens: int = 512
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "boom" in text:
            raise ValueError("simulated scoring failure")
        return super().token_logprob_table(text, max_tokens=max_tokens)


class FlakyBatchGenerateModel(FakeModel):
    """``generate_batch`` raises on the second chunk mid-scoring."""

    def __init__(self) -> None:
        self._batch_calls = 0

    def generate_batch(
        self, prompts: list[str], max_new_tokens: int, *, batch_size: int = 4
    ) -> list[str]:
        self._batch_calls += 1
        if self._batch_calls >= 2:
            raise RuntimeError("simulated batch generation failure")
        return [self.generate(prompt, max_new_tokens) for prompt in prompts]

    def token_logprob_table(
        self, text: str, *, max_tokens: int = 512
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "boom" in text:
            raise ValueError("simulated scoring failure")
        return super().token_logprob_table(text, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _benchmark(texts: list[str], *, prefix: str = "ex") -> BenchmarkData:
    return BenchmarkData(
        examples=[{"id": f"{prefix}-{index}", "text": text} for index, text in enumerate(texts)],
        source="stress",
        dataset_name="stress",
        split="test",
    )


def _config() -> RunConfig:
    # Small bootstrap run keeps this file fast; cache is inert for the fake
    # models (no ``fingerprint`` attribute), so the real cache is untouched.
    return RunConfig(
        model_id="fake",
        benchmark_source="stress",
        limit=10,
        shard_count=2,
        permutation_count=2,
        bootstrap_samples=200,
        seed=0,
    )


def _texts(count: int, *, offset: int = 0) -> list[str]:
    base = [
        "The lamp glows on the table.",
        "Birds fly south for the winter.",
        "Tea cools slowly in the morning light.",
        "Doors open wide to the garden.",
        "Rain taps against the glass pane.",
        "The river runs cold and clear.",
        "Stars shine bright above the hill.",
        "The train arrives at noon sharp.",
    ]
    return [base[(offset + index) % len(base)] + f" variant {offset + index}" for index in range(count)]


def _assert_sane(report: dict, expected_count: int) -> None:
    assert isinstance(report, dict)
    assert report["example_count"] == expected_count
    assert len(report["per_example"]) == expected_count
    assert report["band"] in KNOWN_BANDS
    assert "contamination_index" in report
    assert "confidence_interval" in report
    # No silent NaN leakage into the headline index.
    index = report["contamination_index"]
    if index is not None:
        assert 0.0 <= index <= 1.0


# ---------------------------------------------------------------------------
# Crash-resistance: small, degenerate, and adversarial inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 2, 3])
def test_tiny_benchmark_and_control_sizes(count: int) -> None:
    """1/2/3-example benchmarks and controls must not crash the report."""
    benchmark = _benchmark(_texts(count))
    control = _benchmark(_texts(count, offset=100), prefix="ctrl")
    report = build_report(FakeModel(), benchmark, control, _config())
    _assert_sane(report, count)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_tiny_benchmark_without_control(count: int) -> None:
    benchmark = _benchmark(_texts(count))
    report = build_report(FakeModel(), benchmark, None, _config())
    _assert_sane(report, count)


def test_one_token_text_is_skipped_not_fatal() -> None:
    """A single-token text must be skipped by guided split and recorded as a
    minkpp failure, while the report still completes for the healthy texts.
    """
    benchmark = _benchmark(
        ["The lamp glows on the table.", "a", "Birds fly south for the winter."]
    )
    control = _benchmark(_texts(3, offset=100), prefix="ctrl")
    report = build_report(StrictFakeModel(), benchmark, control, _config())
    _assert_sane(report, 3)
    guided_skipped = {item["id"] for item in report["guided_skipped_examples"]}
    assert "ex-1" in guided_skipped  # the "a" example
    minkpp_skipped = report["minkpp_skipped"]
    assert any(item.startswith("ex-1") for item in minkpp_skipped)
    # The healthy examples still produced scores.
    scored = [row for row in report["per_example"] if row["minkpp_score"] is not None]
    assert len(scored) == 2


def test_one_token_and_whitespace_texts() -> None:
    """Mixed degenerate texts (1 token, whitespace-only) are survived."""
    benchmark = _benchmark(
        [
            "The lamp glows on the table.",
            "x",
            "   \n\t  ",
            "Birds fly south for the winter.",
        ]
    )
    control = _benchmark(_texts(4, offset=100), prefix="ctrl")
    report = build_report(StrictFakeModel(), benchmark, control, _config())
    _assert_sane(report, 4)
    guided_skipped = {item["id"] for item in report["guided_skipped_examples"]}
    assert {"ex-1", "ex-2"} <= guided_skipped
    minkpp_skipped = report["minkpp_skipped"]
    assert any(item.startswith("ex-1") for item in minkpp_skipped)
    assert any(item.startswith("ex-2") for item in minkpp_skipped)
    scored = [row for row in report["per_example"] if row["minkpp_score"] is not None]
    assert len(scored) == 2


def test_all_degenerate_benchmark() -> None:
    """Fixed: build_report records guided as a skipped method instead of crashing."""
    benchmark = _benchmark(["a", "b", "   \n  "])
    control = _benchmark(_texts(3, offset=100), prefix="ctrl")
    report = build_report(StrictFakeModel(), benchmark, control, _config())
    _assert_sane(report, 3)


def test_very_long_texts() -> None:
    """5000-token-ish texts (repeat pattern) must not crash or hang."""
    pattern = "alpha beta gamma delta epsilon zeta eta theta "
    long_text = (pattern * 120)[:5000]  # ~5000 characters == ~5000 char-tokens
    assert len(long_text) >= 5000
    benchmark = _benchmark([long_text, "The lamp glows on the table.", long_text[::-1]])
    control = _benchmark(_texts(3, offset=100), prefix="ctrl")
    report = build_report(FakeModel(), benchmark, control, _config())
    _assert_sane(report, 3)


def test_many_duplicate_texts() -> None:
    """Identical texts repeated many times must not crash."""
    identical = "The lamp glows on the table."
    benchmark = _benchmark([identical] * 12)
    control = _benchmark([identical] * 8, prefix="ctrl")
    report = build_report(FakeModel(), benchmark, control, _config())
    _assert_sane(report, 12)


def test_non_ascii_and_emoji_texts() -> None:
    texts = [
        "Héllo wörld — café naïve résumé façade piñata.",
        "你好世界，这是中文文本样本。",
        "Emoji test: 🚀🎉🔥🌍✨ plus some plain words here.",
        "Mixed: 日本語 テスト 🎭 and English words together.",
        "Ñoño déjà vu — accents and symbols galore 🎶.",
    ]
    benchmark = _benchmark(texts)
    control = _benchmark(_texts(5, offset=100), prefix="ctrl")
    report = build_report(FakeModel(), benchmark, control, _config())
    _assert_sane(report, 5)


def test_duplicate_ids_are_rejected_before_records_collapse() -> None:
    """Ambiguous IDs must not silently overwrite different scored texts."""
    benchmark = BenchmarkData(
        examples=[
            {"id": "dup", "text": "The lamp glows on the table."},
            {"id": "dup", "text": "Birds fly south for the winter."},
            {"id": "dup", "text": "Tea cools slowly in the morning light."},
            {"id": "solo", "text": "Doors open wide to the garden."},
            {"id": "solo2", "text": "Rain taps against the glass pane."},
        ],
        source="stress",
        dataset_name="stress",
        split="test",
    )
    control = _benchmark(_texts(5, offset=100), prefix="ctrl")
    with pytest.raises(ValueError, match="duplicate example ids"):
        build_report(FakeModel(), benchmark, control, _config())


# ---------------------------------------------------------------------------
# Mid-scoring model failures
# ---------------------------------------------------------------------------


def test_single_scoring_failure_degrades_gracefully() -> None:
    """token_logprob_table failing for one text is recorded, not fatal."""
    benchmark = _benchmark(
        [
            "The lamp glows on the table.",
            "this boom text fails scoring",
            "Birds fly south for the winter.",
            "Tea cools slowly in the morning light.",
        ]
    )
    control = _benchmark(_texts(4, offset=100), prefix="ctrl")
    report = build_report(PartiallyFailingModel(), benchmark, control, _config())
    _assert_sane(report, 4)
    assert any(item.startswith("ex-1") for item in report["minkpp_skipped"])
    scored = [row for row in report["per_example"] if row["minkpp_score"] is not None]
    assert len(scored) == 3


def test_generate_batch_failure_mid_scoring() -> None:
    """Fixed: a mid-scoring generate_batch failure degrades to a skipped method."""
    benchmark = _benchmark(
        [
            "The lamp glows on the table.",
            "this boom text fails scoring",
            "Birds fly south for the winter.",
            "Tea cools slowly in the morning light.",
            "Doors open wide to the garden.",
            "Rain taps against the glass pane.",
        ]
    )
    control = _benchmark(_texts(6, offset=100), prefix="ctrl")
    report = build_report(FlakyBatchGenerateModel(), benchmark, control, _config())
    _assert_sane(report, 6)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_same_model_same_report() -> None:
    """Two runs over the same deterministic model give identical headline
    numbers (kept small; the same property is covered at larger scale in
    test_end_to_end.py)."""
    benchmark = _benchmark(_texts(8))
    control = _benchmark(_texts(8, offset=100), prefix="ctrl")
    model = FakeModel()
    config = _config()
    first = build_report(model, benchmark, control, config)
    second = build_report(model, benchmark, control, config)
    assert first["contamination_index"] == second["contamination_index"]
    assert first["confidence_interval"] == second["confidence_interval"]
