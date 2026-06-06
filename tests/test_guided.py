import pytest

from contamcheck.methods.guided import (
    build_general_prompt,
    build_guided_prompt,
    score_completion,
    split_text,
)
from contamcheck.model import load_model


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text]

    def decode(
        self, token_ids: list[int], skip_special_tokens: bool = True
    ) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def test_token_split_uses_a_token_boundary() -> None:
    result = split_text(CharacterTokenizer(), "abcdef", 0.5)
    assert result.prefix == "abc"
    assert result.suffix == "def"
    assert result.suffix_token_count == 3


def test_token_split_handles_unicode() -> None:
    result = split_text(CharacterTokenizer(), "café東京", 0.5)
    assert result.prefix + result.suffix == "café東京"
    assert result.suffix_token_count > 0


def test_token_split_rejects_empty_and_single_token_text() -> None:
    tokenizer = CharacterTokenizer()
    with pytest.raises(ValueError, match="non-whitespace"):
        split_text(tokenizer, "  ")
    with pytest.raises(ValueError, match="at least two"):
        split_text(tokenizer, "x")


def test_prompt_builders_exact_strings() -> None:
    assert build_guided_prompt("alpha", "demo", "test") == (
        "You are given the first part of an example from the demo dataset, split "
        "test. Reproduce the exact second part as it appears in the dataset, with "
        "no extra commentary.\n\nFirst part:\nalpha\n\nSecond part:\n"
    )
    assert build_general_prompt("alpha") == (
        "You are given the first part of an example from a dataset. Reproduce the "
        "exact second part as it appears in the dataset, with no extra commentary."
        "\n\nFirst part:\nalpha\n\nSecond part:\n"
    )


def test_exact_and_near_match_scoring() -> None:
    exact = score_completion("Hello  World", "hello world")
    near = score_completion("abcdefghij", "abcdefghiX")
    far = score_completion("abcdefghij", "xxxxxxxxxx")
    assert exact["exact"] is True
    assert exact["near"] is True
    assert near["exact"] is False
    assert near["near"] is True
    assert far["near"] is False


def test_generation_smoke_when_tiny_model_is_cached() -> None:
    try:
        model = load_model("sshleifer/tiny-gpt2", local_files_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        pytest.skip(f"sshleifer/tiny-gpt2 is not cached: {error}")
    output = model.generate("The capital of France is", 2)
    assert isinstance(output, str)
    assert model.tokenizer.pad_token_id is not None
    logits, targets = model.token_logprob_table("word " * 200, max_tokens=32)
    assert len(logits) == len(targets) == 31
