"""Small, dependency-light statistical toolkit.

The functions here are deliberately written from first principles so that the
package does not need SciPy.  Every routine has a docstring stating the null
hypothesis or the quantity it estimates.  The module is the numerical
foundation for the evidence synthesis in :mod:`contamcheckr.calibration`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import itertools
import math
from statistics import NormalDist

import numpy as np


_EPS = 3.0e-14
_FPMIN = 1.0e-300
_NORMAL = NormalDist()


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp ``value`` to ``[lower, upper]``."""
    if lower > upper:
        raise ValueError("Lower bound cannot exceed upper bound.")
    return float(min(max(value, lower), upper))


def _as_float_array(values: Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


# ---------------------------------------------------------------------------
# Special functions
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, 401):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function ``I_x(a, b)``."""
    if a <= 0.0 or b <= 0.0:
        raise ValueError("Beta parameters must be positive.")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _gammainc_series(a: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(1, 1000):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gammaincc_cf(a: float, x: float) -> float:
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b if abs(b) > _FPMIN else 1.0 / _FPMIN
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def regularized_upper_gamma(a: float, x: float) -> float:
    """Regularized upper incomplete gamma function ``Q(a, x)``."""
    if a <= 0.0:
        raise ValueError("Gamma shape must be positive.")
    if x <= 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gammainc_series(a, x)
    return _gammaincc_cf(a, x)


def normal_ppf(probability: float) -> float:
    """Inverse standard normal CDF."""
    if not 0.0 < probability < 1.0:
        raise ValueError("Probability must be strictly between 0 and 1.")
    return float(_NORMAL.inv_cdf(probability))


def normal_cdf(value: float) -> float:
    """Standard normal CDF."""
    return float(_NORMAL.cdf(value))


def student_t_sf(value: float, degrees_of_freedom: float) -> float:
    """Survival function ``P(T > value)`` for a Student t distribution."""
    if degrees_of_freedom <= 0.0:
        raise ValueError("Degrees of freedom must be positive.")
    if not math.isfinite(value):
        return 0.0 if value > 0 else 1.0
    if value == 0.0:
        return 0.5
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * regularized_incomplete_beta(
        degrees_of_freedom / 2.0, 0.5, x
    )
    return tail if value > 0 else 1.0 - tail


def student_t_cdf(value: float, degrees_of_freedom: float) -> float:
    """CDF of a Student t distribution."""
    return 1.0 - student_t_sf(value, degrees_of_freedom)


def student_t_ppf(probability: float, degrees_of_freedom: float) -> float:
    """Quantile of a Student t distribution via monotone bisection."""
    if not 0.0 < probability < 1.0:
        raise ValueError("Probability must be strictly between 0 and 1.")
    if degrees_of_freedom <= 0.0:
        raise ValueError("Degrees of freedom must be positive.")
    if degrees_of_freedom > 1_000_000:
        return normal_ppf(probability)
    low, high = -1.0e4, 1.0e4
    for _ in range(200):
        middle = 0.5 * (low + high)
        if student_t_cdf(middle, degrees_of_freedom) < probability:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


# ---------------------------------------------------------------------------
# Effect sizes and confidence intervals
# ---------------------------------------------------------------------------


def mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
) -> tuple[float, float] | None:
    """Student-t confidence interval for a mean."""
    array = _as_float_array(values, name="values")
    if len(array) < 2:
        return None
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must be between 0 and 1.")
    standard_error = float(np.std(array, ddof=1)) / math.sqrt(len(array))
    if standard_error == 0.0:
        return float(np.mean(array)), float(np.mean(array))
    critical = student_t_ppf(0.5 + confidence / 2.0, len(array) - 1)
    center = float(np.mean(array))
    return center - critical * standard_error, center + critical * standard_error


def one_sided_t_test(
    values: Sequence[float],
    *,
    null_mean: float = 0.0,
) -> dict[str, float]:
    """One-sided t-test for ``mean(values) > null_mean``."""
    array = _as_float_array(values, name="values")
    if len(array) < 2:
        raise ValueError("A t-test requires at least two observations.")
    center = float(np.mean(array))
    standard_deviation = float(np.std(array, ddof=1))
    if standard_deviation == 0.0:
        if center > null_mean:
            return {"statistic": math.inf, "p_value": 0.0, "df": len(array) - 1.0}
        return {"statistic": 0.0, "p_value": 1.0, "df": len(array) - 1.0}
    statistic = (center - null_mean) / (standard_deviation / math.sqrt(len(array)))
    p_value = student_t_sf(statistic, len(array) - 1)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "df": float(len(array) - 1),
    }


def cohen_dz(differences: Sequence[float]) -> float:
    """Paired standardized mean difference (Cohen's dz)."""
    array = _as_float_array(differences, name="differences")
    if len(array) < 2:
        return 0.0
    standard_deviation = float(np.std(array, ddof=1))
    if standard_deviation == 0.0:
        return 0.0
    return float(np.mean(array)) / standard_deviation


def cohen_d(left: Sequence[float], right: Sequence[float]) -> float:
    """Pooled two-sample standardized mean difference."""
    left_array = _as_float_array(left, name="left")
    right_array = _as_float_array(right, name="right")
    if len(left_array) < 2 or len(right_array) < 2:
        return 0.0
    left_variance = float(np.var(left_array, ddof=1))
    right_variance = float(np.var(right_array, ddof=1))
    pooled = math.sqrt(
        ((len(left_array) - 1) * left_variance
         + (len(right_array) - 1) * right_variance)
        / (len(left_array) + len(right_array) - 2)
    )
    if pooled == 0.0:
        return 0.0
    return (float(np.mean(left_array)) - float(np.mean(right_array))) / pooled


def rankdata(values: Sequence[float]) -> np.ndarray:
    """Average ranks, with ties receiving the mean of their positions."""
    array = _as_float_array(values, name="values")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    sorted_values = array[order]
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def mann_whitney_auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """AUC = P(positive > negative) + 0.5 P(positive == negative)."""
    positive_array = _as_float_array(positive, name="positive")
    negative_array = _as_float_array(negative, name="negative")
    if len(positive_array) == 0 or len(negative_array) == 0:
        raise ValueError("AUC requires both positive and negative scores.")
    combined = np.concatenate([positive_array, negative_array])
    ranks = rankdata(combined)
    positive_rank_sum = float(np.sum(ranks[: len(positive_array)]))
    count_positive = len(positive_array)
    count_negative = len(negative_array)
    return (
        positive_rank_sum - count_positive * (count_positive + 1) / 2.0
    ) / (count_positive * count_negative)


def auc_de_long_ci(
    positive: Sequence[float],
    negative: Sequence[float],
    *,
    confidence: float = 0.95,
) -> dict[str, float]:
    """AUC with a DeLong variance estimate and normal-approximation interval."""
    positive_array = _as_float_array(positive, name="positive")
    negative_array = _as_float_array(negative, name="negative")
    if len(positive_array) < 2 or len(negative_array) < 2:
        raise ValueError("DeLong intervals require at least two scores per group.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must be between 0 and 1.")

    positive_count = len(positive_array)
    negative_count = len(negative_array)
    # Chunk to keep the pairwise comparison matrix bounded.
    positive_placements = np.empty(positive_count, dtype=float)
    negative_placements = np.empty(negative_count, dtype=float)
    negative_placements_sum = np.zeros(negative_count, dtype=float)
    chunk = max(1, 4_000_000 // max(1, negative_count))
    for start in range(0, positive_count, chunk):
        block = positive_array[start : start + chunk]
        greater = block[:, None] > negative_array[None, :]
        equal = block[:, None] == negative_array[None, :]
        positive_placements[start : start + len(block)] = np.mean(
            greater + 0.5 * equal, axis=1
        )
        negative_placements_sum += np.sum(greater + 0.5 * equal, axis=0)
    negative_placements[:] = negative_placements_sum / positive_count
    auc_value = float(np.mean(positive_placements))
    variance = float(
        np.var(positive_placements, ddof=1) / positive_count
        + np.var(negative_placements, ddof=1) / negative_count
    )
    standard_error = math.sqrt(max(variance, 0.0))
    critical = normal_ppf(0.5 + confidence / 2.0)
    return {
        "auc": auc_value,
        "standard_error": standard_error,
        "ci_low": clamp(auc_value - critical * standard_error),
        "ci_high": clamp(auc_value + critical * standard_error),
        "positive_count": float(positive_count),
        "negative_count": float(negative_count),
    }


# ---------------------------------------------------------------------------
# Paired and two-sample tests
# ---------------------------------------------------------------------------


def sign_test_p(
    differences: Sequence[float],
    *,
    alternative: str = "greater",
) -> float:
    """Exact sign test for the median of paired differences."""
    array = _as_float_array(differences, name="differences")
    nonzero = array[array != 0.0]
    if len(nonzero) == 0:
        return 1.0
    positives = int(np.sum(nonzero > 0.0))
    negatives = len(nonzero) - positives
    if alternative == "greater":
        tail = positives
        return float(
            sum(
                math.comb(len(nonzero), index)
                for index in range(tail, len(nonzero) + 1)
            )
            / (2 ** len(nonzero))
        )
    if alternative == "less":
        tail = negatives
        return float(
            sum(
                math.comb(len(nonzero), index)
                for index in range(tail, len(nonzero) + 1)
            )
            / (2 ** len(nonzero))
        )
    if alternative == "two-sided":
        observed = max(positives, negatives)
        return float(
            min(
                1.0,
                2.0
                * sum(
                    math.comb(len(nonzero), index)
                    for index in range(observed, len(nonzero) + 1)
                )
                / (2 ** len(nonzero)),
            )
        )
    raise ValueError("Alternative must be greater, less, or two-sided.")


def paired_permutation_test(
    differences: Sequence[float],
    *,
    permutations: int = 10_000,
    seed: int = 0,
    alternative: str = "greater",
) -> dict[str, float]:
    """Randomization test that flips the sign of each paired difference."""
    array = _as_float_array(differences, name="differences")
    if len(array) == 0:
        raise ValueError("Permutation test requires at least one difference.")
    if permutations < 1:
        raise ValueError("Permutation count must be at least one.")
    observed = float(np.mean(array))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        signs = rng.integers(0, 2, len(array), dtype=np.int8) * 2 - 1
        statistic = float(np.mean(array * signs))
        if alternative == "greater":
            exceed += statistic >= observed
        elif alternative == "less":
            exceed += statistic <= observed
        elif alternative == "two-sided":
            exceed += abs(statistic) >= abs(observed)
        else:
            raise ValueError("Alternative must be greater, less, or two-sided.")
    return {
        "statistic": observed,
        "p_value": float((1 + exceed) / (permutations + 1)),
    }


def permutation_test_two_sample(
    left: Sequence[float],
    right: Sequence[float],
    *,
    statistic: Callable[[np.ndarray, np.ndarray], float] | None = None,
    permutations: int = 10_000,
    seed: int = 0,
    alternative: str = "greater",
) -> dict[str, float]:
    """Label-permutation test for a two-sample statistic."""
    left_array = _as_float_array(left, name="left")
    right_array = _as_float_array(right, name="right")
    if len(left_array) == 0 or len(right_array) == 0:
        raise ValueError("Two-sample test requires both groups.")
    if permutations < 1:
        raise ValueError("Permutation count must be at least one.")
    if statistic is None:
        statistic = lambda a, b: float(np.mean(a) - np.mean(b))  # noqa: E731
    observed = float(statistic(left_array, right_array))
    pooled = np.concatenate([left_array, right_array])
    left_count = len(left_array)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        order = rng.permutation(len(pooled))
        permuted_left = pooled[order[:left_count]]
        permuted_right = pooled[order[left_count:]]
        value = float(statistic(permuted_left, permuted_right))
        if alternative == "greater":
            exceed += value >= observed
        elif alternative == "less":
            exceed += value <= observed
        elif alternative == "two-sided":
            exceed += abs(value) >= abs(observed)
        else:
            raise ValueError("Alternative must be greater, less, or two-sided.")
    return {
        "statistic": observed,
        "p_value": float((1 + exceed) / (permutations + 1)),
    }


def mann_whitney_u_test(
    positive: Sequence[float],
    negative: Sequence[float],
    *,
    alternative: str = "greater",
) -> dict[str, float | int | str | None]:
    """Test equal score distributions under exchangeable group labels.

    Enumerate all label assignments when their number is at most 100,000.
    Unlike the usual no-ties U table, this conditional permutation test also
    handles tied scores.  Larger problems use a tie-corrected normal
    approximation without a continuity correction.  ``method`` identifies
    which calculation was used; neither calculation establishes that the
    groups are exchangeable or that a shift is caused by contamination.
    """
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("Alternative must be greater, less, or two-sided.")
    positive_array = _as_float_array(positive, name="positive")
    negative_array = _as_float_array(negative, name="negative")
    if len(positive_array) == 0 or len(negative_array) == 0:
        raise ValueError("Mann-Whitney requires both groups.")
    combined = np.concatenate([positive_array, negative_array])
    ranks = rankdata(combined)
    positive_count = len(positive_array)
    negative_count = len(negative_array)
    rank_sum = float(np.sum(ranks[:positive_count]))
    u_value = rank_sum - positive_count * (positive_count + 1) / 2.0
    mean_u = positive_count * negative_count / 2.0
    label_assignments = math.comb(len(combined), positive_count)
    if label_assignments <= 100_000:
        # Average ranks are integers or half-integers.  Doubling them makes
        # inclusive tail comparisons exact, including at tied observations.
        doubled_ranks = [int(round(2.0 * rank)) for rank in ranks]
        observed_sum = sum(doubled_ranks[:positive_count])
        total_rank_sum = sum(doubled_ranks)
        smaller_count = min(positive_count, negative_count)
        upper = lower = 0
        for assignment in itertools.combinations(range(len(combined)), smaller_count):
            selected_sum = sum(doubled_ranks[index] for index in assignment)
            candidate_sum = (
                selected_sum
                if positive_count <= negative_count
                else total_rank_sum - selected_sum
            )
            upper += candidate_sum >= observed_sum
            lower += candidate_sum <= observed_sum
        if alternative == "greater":
            p_value = upper / label_assignments
        elif alternative == "less":
            p_value = lower / label_assignments
        else:
            # Doubled smaller inclusive tail, conservatively capped at one.
            p_value = min(1.0, 2.0 * min(upper, lower) / label_assignments)
        return {
            "u": float(u_value),
            "p_value": float(p_value),
            "auc": float(u_value / (positive_count * negative_count)),
            "method": "exact_label_permutation",
            "label_assignments": label_assignments,
        }
    _, counts = np.unique(combined, return_counts=True)
    tie_correction = float(
        np.sum(counts.astype(float) ** 3 - counts.astype(float))
        / (len(combined) * (len(combined) - 1))
    ) if len(combined) > 1 else 0.0
    variance = (
        positive_count
        * negative_count
        / 12.0
        * ((len(combined) + 1) - tie_correction)
    )
    if variance <= 0.0:
        return {
            "u": float(u_value),
            "p_value": 1.0,
            "auc": 0.5,
            "method": "degenerate_all_ties",
            "label_assignments": None,
        }
    z_value = (u_value - mean_u) / math.sqrt(variance)
    if alternative == "greater":
        p_value = 0.5 * math.erfc(z_value / math.sqrt(2.0))
    elif alternative == "less":
        p_value = 0.5 * math.erfc(-z_value / math.sqrt(2.0))
    else:
        p_value = math.erfc(abs(z_value) / math.sqrt(2.0))
    return {
        "u": float(u_value),
        "p_value": float(clamp(p_value)),
        "auc": float(u_value / (positive_count * negative_count)),
        "method": "tie_corrected_normal_approximation",
        "label_assignments": None,
    }


# ---------------------------------------------------------------------------
# P-value combination and multiple testing
# ---------------------------------------------------------------------------


def _validated_pvalues(pvalues: Sequence[float]) -> np.ndarray:
    array = _as_float_array(pvalues, name="pvalues")
    if len(array) == 0:
        raise ValueError("At least one p-value is required.")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("P-values must lie in [0, 1].")
    return array


def cauchy_combination(
    pvalues: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
) -> float:
    """Cauchy combination test, robust to arbitrary dependence.

    Liu & Xie (2020).  The test statistic is a weighted sum of
    ``tan((0.5 - p) * pi)``; its null distribution is standard Cauchy.
    """
    array = _validated_pvalues(pvalues)
    if weights is None:
        weight_array = np.ones(len(array), dtype=float)
    else:
        weight_array = _as_float_array(weights, name="weights")
        if len(weight_array) != len(array):
            raise ValueError("Weights must match the number of p-values.")
        if np.any(weight_array < 0.0) or float(np.sum(weight_array)) <= 0.0:
            raise ValueError("Weights must be non-negative with a positive sum.")
    weight_array = weight_array / float(np.sum(weight_array))
    if np.any(array <= 0.0):
        return 0.0
    terms = weight_array * np.tan((0.5 - array) * math.pi)
    statistic = float(np.sum(terms))
    combined = 0.5 - math.atan(statistic) / math.pi
    return float(clamp(combined))


def fisher_combination(pvalues: Sequence[float]) -> float:
    """Fisher's method.  Assumes independence or positive dependence."""
    array = _validated_pvalues(pvalues)
    if np.any(array <= 0.0):
        return 0.0
    statistic = float(-2.0 * np.sum(np.log(array)))
    return float(clamp(regularized_upper_gamma(len(array), statistic / 2.0)))


def simes_combination(pvalues: Sequence[float]) -> float:
    """Simes' test, valid under positive regression dependence."""
    array = _validated_pvalues(pvalues)
    ordered = np.sort(array)
    count = len(ordered)
    values = ordered * count / np.arange(1, count + 1)
    return float(clamp(float(np.min(values))))


def bonferroni_combination(pvalues: Sequence[float]) -> float:
    """Bonferroni correction; valid under any dependence."""
    array = _validated_pvalues(pvalues)
    return float(clamp(float(np.min(array)) * len(array)))
