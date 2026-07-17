"""Ideal exact best-case resolution of the individual conformal/BH screen.

This is arithmetic, not a power calculation or validation of conformal/FDR
assumptions or parity with floating-point runtime thresholds. It neither loads
data/models nor changes any audit eligibility.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any


MAX_INPUT_DECIMAL_DIGITS = 1000
_INTEGER_INPUT_LIMIT = 10**MAX_INPUT_DECIMAL_DIGITS
_LEVEL_REPRESENTATION_ERROR = (
    "q exceeds the supported exact representation: at most 1000 decimal digits "
    "per integer component and absolute decimal exponent at most 1000."
)


def _count(value: int, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}.")
    if value >= _INTEGER_INPUT_LIMIT:
        raise ValueError(f"{name} exceeds the supported {MAX_INPUT_DECIMAL_DIGITS}-decimal-digit input bound.")
    return value


def _level(value: str | Decimal | Fraction | int | float) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("q must be a finite decimal strictly between 0 and 1.")
    if isinstance(value, Fraction):
        level = value
    else:
        if not isinstance(value, (str, Decimal, int, float)):
            raise ValueError("q must be a finite decimal strictly between 0 and 1.")
        try:
            decimal = value if isinstance(value, Decimal) else Decimal(str(value).strip())
        except (InvalidOperation, ValueError):
            raise ValueError("q must be a finite decimal strictly between 0 and 1.") from None
        if not decimal.is_finite():
            raise ValueError("q must be a finite decimal strictly between 0 and 1.")
        if not 0 < decimal < 1:
            raise ValueError("q must be a finite decimal strictly between 0 and 1.")
        representation = decimal.as_tuple()
        if len(representation.digits) > MAX_INPUT_DECIMAL_DIGITS or abs(representation.exponent) > MAX_INPUT_DECIMAL_DIGITS:
            raise ValueError(_LEVEL_REPRESENTATION_ERROR)
        # A decimal spelling is exact; float callers use str(value), never its
        # binary expansion. The CLI passes the original decimal string.
        level = Fraction(decimal)
    if not 0 < level < 1:
        raise ValueError("q must be a finite decimal strictly between 0 and 1.")
    if level.numerator >= _INTEGER_INPUT_LIMIT or level.denominator >= _INTEGER_INPUT_LIMIT:
        raise ValueError(_LEVEL_REPRESENTATION_ERROR)
    return level


def _ceil(value: Fraction) -> int:
    return (value.numerator + value.denominator - 1) // value.denominator


def reference_resolution(
    candidates: int,
    controls: int,
    *,
    q: str | Decimal | Fraction | int | float = "0.10",
    min_calibration_examples: int = 4,
) -> dict[str, Any]:
    """Plan ideal exact-BH ranks for nominal usable counts.

    The best attainable conformal p-value is 1/(M+1). If exactly k candidates
    attain that value and all others have p=1, BH can flag those k iff
    k >= ceil(N/[q*(M+1)]). The required first-rank reference count is
    ceil(N/q)-1. Counts presume prior quality validation; matching, clean
    controls, exchangeability and FDR conditions are not checked here.
    Counts/Fraction components are bounded to 1000 decimal digits; Decimal
    exponent magnitude is bounded before constructing a potentially huge int.
    The existing float BH implementation may differ at equality boundaries.
    """
    candidates = _count(candidates, "candidates", minimum=1)
    controls = _count(controls, "controls", minimum=0)
    minimum = _count(min_calibration_examples, "min_calibration_examples", minimum=2)
    level = _level(q)
    smallest_p = Fraction(1, controls + 1)
    required_rank = _ceil(Fraction(candidates, 1) / (level * (controls + 1)))
    possible = required_rank <= candidates
    controls_for_one = _ceil(Fraction(candidates, 1) / level) - 1
    if possible:
        summary = (
            f"Under ideal exact BH, the individual screen can reach its thresholds "
            f"with {required_rank} jointly extreme candidate(s). Actual flags are unknown."
        )
    else:
        summary = (
            "Ideal exact BH permits no individual-screen discovery at these counts and q, "
            "even if every candidate exceeds every control."
        )
    return {
        "scope": "ideal_exact_individual_conformal_BH_arithmetic",
        "candidates": candidates,
        "controls": controls,
        "q_exact": str(level),
        "minimum_attainable_p_value_exact": str(smallest_p),
        "required_bh_rank_at_minimum_p": required_rank,
        "minimum_jointly_extreme_group_size": required_rank if possible else None,
        "best_case_discovery_possible": possible,
        "controls_needed_for_one_extreme_candidate": controls_for_one,
        "additional_controls_for_one_extreme_candidate": max(0, controls_for_one - controls),
        "minimum_calibration": {
            "required_examples_per_group": minimum,
            "candidate_count_met": candidates >= minimum,
            "control_count_met": controls >= minimum,
            "count_requirement_met": candidates >= minimum and controls >= minimum,
            "scope": "Count-only requirement, separate from BH arithmetic and other quality checks.",
        },
        "summary": summary,
        "runtime_parity": "Not guaranteed: the existing floating-point BH implementation can differ at equality boundaries.",
        "interpretation": (
            "Jointly extreme means candidates attaining the smallest p-value, not leaked items. "
            "Counts must be usable after quality validation; matching and control nonmembership "
            "remain unverified. This is not actual power or an FDR-validity guarantee."
        ),
        "other_audit_evidence": "Dataset-level comparisons and guided/order evidence have separate requirements.",
    }
