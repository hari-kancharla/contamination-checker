"""Pure unit tests for the paraphrase-robustness probe.

Numpy + pytest only: no model downloads, no training, no torch
forward passes. Slow runner code stays behind ``run_paraphrase_probe``
(imports are local), so importing this module is light.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from paraphrase import cosine, format_results_table, mean_embedding  # noqa: E402


# ---------------------------------------------------------------------------
# cosine
# ---------------------------------------------------------------------------


def test_cosine_identical_is_one() -> None:
    assert cosine(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)


def test_cosine_orthonormal_is_zero() -> None:
    assert cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert cosine(np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)


def test_cosine_orthogonal_is_zero() -> None:
    assert cosine(np.array([1.0, 1.0]), np.array([1.0, -1.0])) == pytest.approx(0.0)


def test_cosine_opposite_is_minus_one() -> None:
    assert cosine(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(-1.0)


def test_cosine_accepts_lists() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_zero_vector_is_zero() -> None:
    assert cosine(np.zeros(3), np.zeros(3)) == 0.0
    assert cosine(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0
    assert cosine(np.array([1.0, 2.0, 3.0]), np.zeros(3)) == 0.0


def test_cosine_empty_is_zero() -> None:
    assert cosine(np.array([]), np.array([])) == 0.0
    assert cosine([], []) == 0.0
    assert cosine(np.array([]), np.array([1.0])) == 0.0


def test_cosine_dimension_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        cosine(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# mean_embedding edge validation (no heavy model: empty text raises first)
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    def __init__(self, token_ids: list[int]) -> None:
        self._token_ids = token_ids

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(self._token_ids)


class _FakeModel:
    def __init__(self, token_ids: list[int]) -> None:
        self.tokenizer = _FakeTokenizer(token_ids)
        self.device = "cpu"
        self.model = None


def test_mean_embedding_empty_text_raises() -> None:
    with pytest.raises(ValueError):
        mean_embedding(None, "")  # type: ignore[arg-type]


def test_mean_embedding_whitespace_raises() -> None:
    with pytest.raises(ValueError):
        mean_embedding(None, "   \n\t  ")  # type: ignore[arg-type]


def test_mean_embedding_empty_tokenization_raises() -> None:
    with pytest.raises(ValueError):
        mean_embedding(_FakeModel([]), "hello")


def test_mean_embedding_invalid_max_tokens_raises() -> None:
    with pytest.raises(ValueError):
        mean_embedding(_FakeModel([1, 2]), "hello", max_tokens=0)


# ---------------------------------------------------------------------------
# format_results_table
# ---------------------------------------------------------------------------


def test_format_table_contains_groups_and_values() -> None:
    table = format_results_table(
        {
            "member_mean_cosine": 0.95,
            "clean_mean_cosine": 0.80,
            "gap": 0.15,
            "auc": 0.90,
            "index": 0.20,
        }
    )
    assert "member" in table
    assert "clean" in table
    assert "0.950" in table
    assert "0.800" in table
    assert "0.900" in table


def test_format_table_none_is_na() -> None:
    table = format_results_table(
        {
            "member_mean_cosine": None,
            "clean_mean_cosine": None,
            "gap": None,
            "auc": None,
            "index": None,
        }
    )
    assert table.count("n/a") >= 1
