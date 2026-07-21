"""Tests for context-windowed token statistics.

Uses the pristine HF snapshot ``sshleifer/tiny-gpt2`` (tiny and
deterministic on CPU).  All tests self-check against the *same* model
object, so they do not depend on the fine-tuned copy under
``experiments/models/``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from contamcheckr.methods.token_stats import score_token_stats
from contamcheckr.methods.windowed import (
    windowed_token_stats,
    windowed_token_stats_batch,
)
from contamcheckr.model import load_model

SHORT_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Hello world, this is a test of token statistics."
)
# ~880 tokens with the tiny GPT-2 tokenizer (>600, so >5 windows of 128).
LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Packaging boxes today is harder than sewing knots in silk. "
) * 40


@pytest.fixture(scope="module")
def model():
    try:
        return load_model(
            "sshleifer/tiny-gpt2", device="cpu", local_files_only=True
        )
    except Exception as exc:  # pragma: no cover - missing snapshot
        pytest.skip(f"tiny model unavailable: {exc}")


def test_short_text_matches_single_pass(model) -> None:
    """(a): short text windowed == single forward pass, atol 1e-5."""
    single = model.token_stats(SHORT_TEXT, max_tokens=512)
    windowed = windowed_token_stats(model, SHORT_TEXT, window_tokens=128)

    assert windowed.truncated is False
    assert windowed.token_count == single.token_count
    assert windowed.token_count > 0
    np.testing.assert_allclose(
        windowed.target_logprobs, single.target_logprobs, atol=1e-5
    )
    np.testing.assert_allclose(windowed.means, single.means, atol=1e-5)


def test_long_text_counts_and_finite(model) -> None:
    """(b): long text is fully scored with no silent truncation.

    Overlapping windows (stride < window) score every token except the
    global first one, so ``token_count == len(tokens) - 1``.  The
    default contiguous stride (stride == window) drops each window
    start (its first token has no left context inside its window), so
    ``token_count == N - ceil(N / window)``; both report
    ``truncated=False`` with all-finite statistics.
    """
    tokens = model.tokenizer.encode(LONG_TEXT, add_special_tokens=False)
    n = len(tokens)
    assert n >= 600
    window = 128

    overlapped = windowed_token_stats(
        model, LONG_TEXT, window_tokens=window, stride_tokens=64
    )
    assert overlapped.token_count == len(tokens) - 1
    assert overlapped.truncated is False
    assert all(math.isfinite(value) for value in overlapped.target_logprobs)

    contiguous = windowed_token_stats(model, LONG_TEXT, window_tokens=window)
    # ceil(n / window) window starts are dropped (see windowed.py docs).
    expected_contiguous = n - (n + window - 1) // window
    assert contiguous.token_count == expected_contiguous
    assert contiguous.truncated is False
    assert all(math.isfinite(value) for value in contiguous.target_logprobs)


def test_long_text_truncated_differs_and_first_window_matches_full(
    model,
) -> None:
    """(c): strongest determinable invariant (verified empirically).

    Verified on sshleifer/tiny-gpt2: windowed scores do *not* equal
    full-context scores within 1e-4 (differences ~0.06 in minkpp, since
    later windows lose left history), so the "equals full" branch does
    not hold.  The holds are:
      1. the first window shares its left context with the full
         forward pass, hence windowed[:127] == full[:127] exactly; and
      2. truncated (128-token) scoring gives a different minkpp from
         windowed (difference ~0.05, far above 0), because it averages
         only the first window.
    """
    tokens = model.tokenizer.encode(LONG_TEXT, add_special_tokens=False)
    assert len(tokens) >= 600

    full = model.token_stats(LONG_TEXT, max_tokens=1024)
    truncated = model.token_stats(LONG_TEXT, max_tokens=128)
    windowed = windowed_token_stats(model, LONG_TEXT, window_tokens=128)

    assert full.truncated is False
    assert truncated.truncated is True
    assert truncated.token_count == 127
    assert full.token_count == len(tokens) - 1
    assert windowed.truncated is False

    # First window's predictions use identical left context.
    np.testing.assert_allclose(
        windowed.target_logprobs[:127],
        full.target_logprobs[:127],
        atol=1e-5,
    )
    np.testing.assert_allclose(windowed.means[:127], full.means[:127], atol=1e-5)

    windowed_minkpp = score_token_stats(windowed, LONG_TEXT).minkpp
    truncated_minkpp = score_token_stats(truncated, LONG_TEXT).minkpp
    assert abs(windowed_minkpp - truncated_minkpp) > 1e-3


def test_empty_text_raises(model) -> None:
    with pytest.raises(ValueError):
        windowed_token_stats(model, "", window_tokens=128)
    with pytest.raises(ValueError):
        windowed_token_stats(model, "   ", window_tokens=128)


def test_invalid_window_and_stride_raise(model) -> None:
    with pytest.raises(ValueError):
        windowed_token_stats(model, SHORT_TEXT, window_tokens=1)
    with pytest.raises(ValueError):
        windowed_token_stats(
            model, SHORT_TEXT, window_tokens=128, stride_tokens=200
        )
    with pytest.raises(ValueError):
        windowed_token_stats(
            model, SHORT_TEXT, window_tokens=128, stride_tokens=0
        )


def test_batch_wrapper_matches_single(model) -> None:
    results = windowed_token_stats_batch(
        model, [SHORT_TEXT, SHORT_TEXT], window_tokens=128, batch_size=2
    )
    assert len(results) == 2
    single = windowed_token_stats(model, SHORT_TEXT, window_tokens=128)
    for result in results:
        assert result.token_count == single.token_count
        np.testing.assert_allclose(
            result.target_logprobs, single.target_logprobs, atol=1e-5
        )
