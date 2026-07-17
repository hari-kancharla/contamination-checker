import pytest

from contamcheckr.methods.canonical_order import canonical_order_test


SEPARATOR = "\n\n---\n\n"


def inversion_scorer(text: str) -> float:
    values = [int(value) for value in text.split(SEPARATOR)]
    inversions = sum(
        left > right
        for index, left in enumerate(values)
        for right in values[index + 1 :]
    )
    return -float(inversions)


def test_canonical_order_detects_a_sorted_sequence_but_does_not_aggregate() -> None:
    examples = [{"id": str(index), "text": str(index)} for index in range(8)]
    result = canonical_order_test(
        examples,
        inversion_scorer,
        shard_count=4,
        permutation_count=25,
        seed=3,
    )
    assert result.canonical_advantage > 0.0
    assert result.p_value < 0.05
    assert result.index_order is None
    assert result.aggregate_reason is not None
    assert len(result.shards) == 4


def test_powered_order_effect_enters_the_index() -> None:
    examples = [{"id": str(index), "text": str(index)} for index in range(40)]
    result = canonical_order_test(
        examples,
        inversion_scorer,
        shard_count=10,
        permutation_count=25,
        seed=3,
    )
    assert result.shard_count == 10
    assert result.p_value < 0.05
    assert result.canonical_advantage > 0.001
    assert result.index_order == pytest.approx(1.0 - result.p_value)
    assert result.aggregate_reason is None


def test_published_shard_logs_do_not_enter_the_index() -> None:
    from contamcheckr.methods.canonical_order import order_aggregate_index

    leak = [
        3.173828122271516e-05,
        7.324218699977791e-06,
        -9.765625009094947e-06,
        1.9531250018189894e-05,
        1.464843751364242e-05,
    ]
    sst2 = [
        4.8238372802734375,
        9.428009033203125,
        -1.626617431640625,
        -5.911163330078125,
        -6.4127197265625,
    ]
    leak_p, leak_index, leak_reason = order_aggregate_index(leak)
    sst_p, sst_index, sst_reason = order_aggregate_index(sst2)
    assert leak_p == pytest.approx(0.06936, abs=1e-4)
    assert leak_index is None
    assert leak_reason is not None
    assert sst_p == pytest.approx(0.4927, abs=1e-3)
    assert sst_index is None
    assert sst_reason is not None


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
    with pytest.raises(ValueError, match="finite"):
        canonical_order_test(examples, lambda text: float("nan"))
