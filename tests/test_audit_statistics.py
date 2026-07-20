"""Statistical regressions for the small-sample audit and extreme tails."""

import itertools
import math

import pytest

from contamcheckr.stats import mann_whitney_u_test


def test_exact_mann_whitney_default_minimum_is_super_uniform() -> None:
    # Under the exchangeable continuous null, all 70 rank-label assignments
    # are equally likely.  The old normal approximation rejected 4/70 at .05.
    p_values = []
    for chosen in itertools.combinations(range(8), 4):
        other = [rank for rank in range(8) if rank not in chosen]
        result = mann_whitney_u_test(chosen, other)
        assert result["method"] == "exact_label_permutation"
        assert result["label_assignments"] == 70
        p_values.append(float(result["p_value"]))
    assert sum(p <= 0.05 for p in p_values) == 2
    for alpha in (0.01, 0.05, 0.10, 0.25, 0.50):
        assert sum(p <= alpha for p in p_values) / 70 <= alpha + 1e-12


def test_exact_mann_whitney_boundary_and_directions() -> None:
    result = mann_whitney_u_test([2, 5, 6, 7], [0, 1, 3, 4])
    assert result["u"] == 14.0
    assert result["p_value"] == pytest.approx(4 / 70)
    assert mann_whitney_u_test([5, 6, 7, 8], [1, 2, 3, 4])["p_value"] == pytest.approx(1 / 70)
    assert mann_whitney_u_test(
        [1, 2, 3, 4], [5, 6, 7, 8], alternative="less"
    )["p_value"] == pytest.approx(1 / 70)
    assert mann_whitney_u_test(
        [5, 6, 7, 8], [1, 2, 3, 4], alternative="two-sided"
    )["p_value"] == pytest.approx(2 / 70)


def test_exact_mann_whitney_includes_tied_label_assignments() -> None:
    # One of the six choices assigns both high observations to the test
    # group; labels remain distinct even when their score values tie.
    result = mann_whitney_u_test([1, 1], [0, 0])
    assert result["p_value"] == pytest.approx(1 / 6)
    assert result["label_assignments"] == 6
    assert mann_whitney_u_test(
        [1, 1], [0, 0], alternative="two-sided"
    )["p_value"] == pytest.approx(1 / 3)
    for alternative in ("greater", "less", "two-sided"):
        assert mann_whitney_u_test([1] * 4, [1] * 4, alternative=alternative)["p_value"] == 1.0


def test_exact_mann_whitney_imbalanced_groups() -> None:
    result = mann_whitney_u_test(list(range(1, 101)), [0])
    assert result["label_assignments"] == 101
    assert result["p_value"] == pytest.approx(1 / 101)


def test_mann_whitney_large_finite_tail_does_not_cancel_to_zero() -> None:
    positive = list(range(200, 400))
    negative = list(range(200))
    result = mann_whitney_u_test(positive, negative)
    z = 20_000 / math.sqrt(200 * 200 * 401 / 12)
    expected = 0.5 * math.erfc(z / math.sqrt(2))
    assert result["method"] == "tie_corrected_normal_approximation"
    assert result["label_assignments"] is None
    assert 0 < result["p_value"] < 1e-60
    assert result["p_value"] == pytest.approx(expected, rel=1e-12, abs=0)
    assert mann_whitney_u_test(
        negative, positive, alternative="less"
    )["p_value"] == pytest.approx(expected, rel=1e-12, abs=0)
    assert mann_whitney_u_test(
        positive, negative, alternative="two-sided"
    )["p_value"] == pytest.approx(2 * expected, rel=1e-12, abs=0)


def test_mann_whitney_validates_direction_even_when_all_tied() -> None:
    with pytest.raises(ValueError, match="Alternative"):
        mann_whitney_u_test([1, 1], [1, 1], alternative="invalid")
