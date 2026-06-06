import pytest

from contamcheckr.metrics import (
    auc,
    levenshtein,
    paired_bootstrap_ci,
    rouge_l_f1,
)


def test_rouge_l_exact_disjoint_and_partial() -> None:
    assert rouge_l_f1("alpha beta", "alpha beta") == 1.0
    assert rouge_l_f1("alpha beta", "gamma delta") == 0.0
    assert rouge_l_f1("alpha beta gamma", "alpha gamma") == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("kitten", "sitting", 3),
        ("flaw", "lawn", 2),
        ("", "abc", 3),
        ("same", "same", 0),
    ],
)
def test_levenshtein_known_pairs(left: str, right: str, expected: int) -> None:
    assert levenshtein(left, right) == expected


def test_auc_perfect_reversed_and_tied() -> None:
    labels = [0, 0, 1, 1]
    assert auc(labels, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auc(labels, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auc(labels, [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_paired_bootstrap_is_deterministic() -> None:
    first = paired_bootstrap_ci([0.2, 0.4, 0.6], [0.1, 0.2, 0.3], seed=7)
    second = paired_bootstrap_ci([0.2, 0.4, 0.6], [0.1, 0.2, 0.3], seed=7)
    assert first == second
    assert first is not None
    assert first[0] <= first[1]

