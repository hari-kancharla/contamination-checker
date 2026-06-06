import pytest

from contamcheck.methods.canonical_order import canonical_order_test


SEPARATOR = "\n\n---\n\n"


def inversion_scorer(text: str) -> float:
    values = [int(value) for value in text.split(SEPARATOR)]
    inversions = sum(
        left > right
        for index, left in enumerate(values)
        for right in values[index + 1 :]
    )
    return -float(inversions)


def test_canonical_order_has_positive_advantage() -> None:
    examples = [{"id": str(index), "text": str(index)} for index in range(8)]
    result = canonical_order_test(
        examples,
        inversion_scorer,
        shard_count=4,
        permutation_count=25,
        seed=3,
    )
    assert result.canonical_advantage > 0.0
    assert result.p_value <= 2 / 26
    assert result.index_order == pytest.approx(1.0 - result.p_value)
    assert len(result.shards) == 4


def test_canonical_order_skips_dataset_that_is_too_small() -> None:
    examples = [{"id": str(index), "text": str(index)} for index in range(3)]
    with pytest.raises(ValueError, match="at least four"):
        canonical_order_test(examples, inversion_scorer)


def test_large_shard_setting_does_not_create_empty_shards() -> None:
    examples = [{"id": str(index), "text": str(index)} for index in range(4)]
    result = canonical_order_test(
        examples,
        inversion_scorer,
        shard_count=20,
        permutation_count=3,
    )
    assert len(result.shards) == 2
    assert all(shard["example_ids"] for shard in result.shards)


def test_non_finite_sequence_score_raises() -> None:
    examples = [{"id": str(index), "text": str(index)} for index in range(4)]
    with pytest.raises(ValueError, match="not finite"):
        canonical_order_test(examples, lambda text: float("nan"))
