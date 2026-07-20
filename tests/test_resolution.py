"""Exact arithmetic/CLI checks, not simulated membership-performance evidence."""

from decimal import Decimal, localcontext
from fractions import Fraction
import json

import pytest

from contamcheckr.resolution import reference_resolution


def independent_bh_count(values, q):
    """Literal exact step-up thresholds; does not use the planner's formula."""
    ordered = sorted(values)
    last = 0
    for rank, value in enumerate(ordered, 1):
        if value <= q * Fraction(rank, len(ordered)):
            last = rank
    return last


@pytest.mark.parametrize(
    "n,m,group,rank,needed",
    [(8, 8, None, 9, 79), (40, 40, 10, 10, 399), (1000, 1000, 10, 10, 9999)],
)
def test_real_reference_count_examples(n, m, group, rank, needed):
    result = reference_resolution(n, m)
    assert result["minimum_jointly_extreme_group_size"] == group
    assert result["required_bh_rank_at_minimum_p"] == rank
    assert result["best_case_discovery_possible"] == (group is not None)
    assert result["controls_needed_for_one_extreme_candidate"] == needed
    assert Fraction(result["minimum_attainable_p_value_exact"]) == Fraction(1, m + 1)


def test_minimum_group_matches_exhaustive_bh_thresholds():
    # Enumerate every extreme-group size in each small exact design. This
    # checks off-by-one/tie boundaries independently of the closed-form ceil.
    for n in range(1, 13):
        for m in range(0, 16):
            for q in (Fraction(1, 100), Fraction(1, 10), Fraction(1, 5), Fraction(1, 4), Fraction(1, 2), Fraction(9, 10)):
                smallest = Fraction(1, m + 1)
                possible_sizes = []
                for size in range(n + 1):
                    flags = independent_bh_count([smallest] * size + [Fraction(1)] * (n - size), q)
                    assert flags in (0, size)
                    if flags:
                        possible_sizes.append(size)
                result = reference_resolution(n, m, q=q)
                assert result["minimum_jointly_extreme_group_size"] == (min(possible_sizes) if possible_sizes else None)


def test_first_rank_control_requirement_is_minimal():
    for n in range(1, 31):
        for q in (Fraction(1, 100), Fraction(1, 10), Fraction(1, 4), Fraction(1, 2), Fraction(9, 10)):
            result = reference_resolution(n, 0, q=q)
            needed = result["controls_needed_for_one_extreme_candidate"]
            assert independent_bh_count([Fraction(1, needed + 1)] + [Fraction(1)] * (n - 1), q) == 1
            if needed:
                assert independent_bh_count([Fraction(1, needed)] + [Fraction(1)] * (n - 1), q) == 0


def test_decimal_boundary_is_not_rounded_to_binary_float():
    exact = reference_resolution(10, 99, q="0.10")
    below = reference_resolution(10, 99, q="0.09999999999999999999999999999999")
    above = reference_resolution(10, 99, q="0.10000000000000000000000000000001")
    assert exact["minimum_jointly_extreme_group_size"] == 1
    assert below["minimum_jointly_extreme_group_size"] == 2
    assert above["minimum_jointly_extreme_group_size"] == 1
    assert reference_resolution(1, 9, q="0.10")["controls_needed_for_one_extreme_candidate"] == 9
    assert reference_resolution(1, 9, q="0.09999999999999999999999999")["controls_needed_for_one_extreme_candidate"] == 10
    assert reference_resolution(10, 99, q=0.1)["q_exact"] == "1/10"


def test_huge_integers_and_low_decimal_context_remain_exact():
    n = 10**200 + 1
    m = 10**400
    with localcontext() as context:
        context.prec = 2
        result = reference_resolution(n, m, q=Decimal("1e-250"))
    q = Fraction(1, 10**250)
    assert result["controls_needed_for_one_extreme_candidate"] == n * 10**250 - 1
    threshold = Fraction(n, 1) / (q * (m + 1))
    rank = result["required_bh_rank_at_minimum_p"]
    assert rank - 1 < threshold <= rank
    smallest = Fraction(result["minimum_attainable_p_value_exact"])
    assert smallest == Fraction(1, m + 1) > 0
    assert float(smallest) == 0.0  # Demonstrates why the output keeps a rational.
    assert json.loads(json.dumps(result))["controls"] == m
    assert result["minimum_attainable_p_value_exact"] != "0"


@pytest.mark.parametrize("q", ["1e-5000", Decimal("1e-5000"), Decimal("1e-1000000000"),
                               Fraction(1, 10**5000), Fraction(10**5000 + 1, 10**5001 + 3)],
                         ids=["tiny_string", "tiny_decimal", "huge_exponent", "huge_denominator", "huge_components"])
def test_unsupported_q_representation_has_clear_bounded_error(q):
    with pytest.raises(ValueError, match="supported exact representation"):
        reference_resolution(8, 8, q=q)


@pytest.mark.parametrize("n,m", [(10**5000, 8), (8, 10**5000), (10**1000, 8)],
                         ids=["huge_candidates", "huge_controls", "input_bound"])
def test_unsupported_integer_sizes_fail_before_string_conversion(n, m):
    with pytest.raises(ValueError, match="1000-decimal-digit input bound"):
        reference_resolution(n, m)


def test_maximum_supported_components_serialize_without_global_limit_changes():
    largest = 10**1000 - 1
    result = reference_resolution(largest, largest, q=Fraction(1, largest))
    assert result["controls_needed_for_one_extreme_candidate"] == largest**2 - 1
    assert json.loads(json.dumps(result))["candidates"] == largest


@pytest.mark.parametrize("n,m,q,group", [(7, 9, "0.7", 1), (6, 9, "0.10", 6), (12, 9, "0.10", 12)])
def test_ideal_threshold_result_does_not_claim_float_runtime_parity(n, m, q, group):
    result = reference_resolution(n, m, q=q)
    smallest = Fraction(1, m + 1)
    values = [smallest] * group + [Fraction(1)] * (n - group)
    assert independent_bh_count(values, Fraction(q)) == group
    assert result["minimum_jointly_extreme_group_size"] == group
    # This literal float adjusted-q calculation reproduces the current edge
    # without making a future production BH fix fail this planner regression.
    float_adjusted = min(float(smallest) * n / rank for rank in range(1, group + 1))
    assert float_adjusted > float(q)
    assert result["scope"] == "ideal_exact_individual_conformal_BH_arithmetic"
    assert "Not guaranteed" in result["runtime_parity"]
    assert "equality boundaries" in result["runtime_parity"]


def test_count_eligibility_and_resolution_are_separate():
    enough_resolution = reference_resolution(1, 1, q="0.5")
    assert enough_resolution["best_case_discovery_possible"]
    assert not enough_resolution["minimum_calibration"]["count_requirement_met"]
    enough_count = reference_resolution(8, 4)
    assert enough_count["minimum_calibration"]["count_requirement_met"]
    assert not enough_count["best_case_discovery_possible"]
    assert reference_resolution(2, 2, min_calibration_examples=2)["minimum_calibration"]["count_requirement_met"]
    one_candidate = reference_resolution(1, 100)
    assert one_candidate["best_case_discovery_possible"]
    assert one_candidate["minimum_calibration"]["control_count_met"]
    assert not one_candidate["minimum_calibration"]["candidate_count_met"]
    assert not one_candidate["minimum_calibration"]["count_requirement_met"]


@pytest.mark.parametrize("n,m", [(0, 8), (-1, 8), (8, -1), (True, 8), (8, False), (1.0, 8), (8, 2.0)])
def test_invalid_counts(n, m):
    with pytest.raises(ValueError, match="integer"):
        reference_resolution(n, m)


@pytest.mark.parametrize("q", ["0", "-0.1", "1", "1.01", "NaN", "sNaN", "Infinity", "-Infinity", "", "nonsense", "1/10", float("nan"), float("inf"), True, None])
def test_invalid_q(q):
    with pytest.raises(ValueError, match="finite decimal strictly between"):
        reference_resolution(8, 8, q=q)


@pytest.mark.parametrize("minimum", [0, 1, -1, True, 4.0])
def test_invalid_minimum_calibration_count(minimum):
    with pytest.raises(ValueError, match="integer at least 2"):
        reference_resolution(8, 8, min_calibration_examples=minimum)


def test_cli_resolution_loads_no_model_or_benchmark(monkeypatch, capsys):
    from contamcheckr import cli

    def forbidden(*args, **kwargs):
        raise AssertionError("Planning must not invoke a model/dataset/run handler")

    monkeypatch.setattr(cli, "load_model", forbidden)
    monkeypatch.setattr(cli, "load_benchmark", forbidden)
    monkeypatch.setattr(cli, "_run", forbidden)
    monkeypatch.setattr(cli, "_capabilities", forbidden)
    assert cli.main(["resolution", "--candidates", "1000", "--controls", "1000", "--q", "0.10"]) == 0
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert not output.err
    assert result["minimum_jointly_extreme_group_size"] == 10
    assert result["controls_needed_for_one_extreme_candidate"] == 9999
    assert result["scope"] == "ideal_exact_individual_conformal_BH_arithmetic"


@pytest.mark.parametrize("q,message", [("NaN", "finite decimal strictly between"),
                                      ("1e-5000", "supported exact representation")])
def test_cli_invalid_input_does_not_fall_through_to_audit(monkeypatch, capsys, q, message):
    from contamcheckr import cli

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid planning input reached a model/dataset call")

    monkeypatch.setattr(cli, "load_model", forbidden)
    monkeypatch.setattr(cli, "load_benchmark", forbidden)
    monkeypatch.setattr(cli, "_run", forbidden)
    assert cli.main(["resolution", "--candidates", "8", "--controls", "8", "--q", q]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert message in output.err
