from dataclasses import dataclass

import numpy as np

from .methods.canonical_order import order_aggregate_index
from .metrics import auc, clamp
from .stats import (
    auc_de_long_ci,
    benjamini_hochberg_flags,
    cohen_d,
    conformal_pvalues,
    mann_whitney_u_test,
)


@dataclass(frozen=True)
class CombinedIndex:
    contamination_index: float | None
    index_basis: str
    band: str
    active_weights: dict[str, float]


def calibrate_minkpp(
    benchmark_scores: list[float],
    control_scores: list[float] | None,
) -> tuple[float | None, float | None]:
    if not control_scores:
        return None, None
    if not benchmark_scores:
        raise ValueError("Min-K%++ calibration requires benchmark scores.")
    labels = [1] * len(benchmark_scores) + [0] * len(control_scores)
    measured_auc = auc(labels, benchmark_scores + control_scores)
    return clamp(2.0 * (measured_auc - 0.5)), measured_auc


def minkpp_diagnostics(
    benchmark_scores: list[float],
    control_scores: list[float],
    *,
    confidence: float = 0.95,
) -> dict[str, float | list[float] | str] | None:
    """Uncertainty diagnostics for a calibrated Min-K%++ AUC.

    Returns the AUC with a DeLong interval, a one-sided Mann-Whitney
    p-value (benchmark above control), and the standardized mean
    difference.  ``None`` when either group has fewer than two scores,
    because the interval estimate needs variation within each group.
    """
    if len(benchmark_scores) < 2 or len(control_scores) < 2:
        return None
    auc_result = auc_de_long_ci(
        benchmark_scores, control_scores, confidence=confidence
    )
    mann_whitney = mann_whitney_u_test(
        benchmark_scores, control_scores, alternative="greater"
    )
    return {
        "auc": float(auc_result["auc"]),
        "auc_ci": [float(auc_result["ci_low"]), float(auc_result["ci_high"])],
        "auc_standard_error": float(auc_result["standard_error"]),
        "mann_whitney_p_value": float(mann_whitney["p_value"]),
        "mann_whitney_method": str(mann_whitney.get("method", "unspecified")),
        "cohen_d": float(cohen_d(benchmark_scores, control_scores)),
    }


def auc_posterior_mean(
    benchmark_scores: list[float],
    control_scores: list[float],
    *,
    prior_alpha: float = 0.5,
) -> float | None:
    """Legacy add-half AUC smoothing heuristic, not a Bayesian posterior.

    Pairwise wins share observations and are not independent Bernoulli
    trials. The add-half formula is retained for artifact compatibility;
    it provides neither posterior uncertainty nor a small-sample validity
    guarantee. The combined index uses the unsmoothed AUC.
    """
    if not benchmark_scores or not control_scores:
        return None
    if prior_alpha <= 0.0:
        raise ValueError("Prior alpha must be positive.")
    wins = 0.0
    for benchmark_value in benchmark_scores:
        for control_value in control_scores:
            if benchmark_value > control_value:
                wins += 1.0
            elif benchmark_value == control_value:
                wins += 0.5
    pairs = float(len(benchmark_scores) * len(control_scores))
    return float((wins + prior_alpha) / (pairs + 2.0 * prior_alpha))


def control_null_check(
    control_scores: list[float],
    *,
    min_examples: int = 4,
) -> dict[str, float | list[float] | int] | None:
    """Null calibration: does the control agree with itself?

    The control scores are split into interleaved halves (even/odd
    positions, which spreads any ordering artefact across both halves)
    and scored against each other. A detected difference can warn about
    ordering or heterogeneity. A nonsignificant result cannot establish
    homogeneity or benchmark/control exchangeability. ``None`` when a
    half is below ``min_examples``.
    """
    first = [float(value) for value in control_scores[0::2]]
    second = [float(value) for value in control_scores[1::2]]
    if len(first) < min_examples or len(second) < min_examples:
        return None
    auc_result = auc_de_long_ci(first, second)
    mann_whitney = mann_whitney_u_test(first, second, alternative="two-sided")
    return {
        "auc": float(auc_result["auc"]),
        "auc_ci": [float(auc_result["ci_low"]), float(auc_result["ci_high"])],
        "mann_whitney_p_value": float(mann_whitney["p_value"]),
        "cohen_d": float(cohen_d(first, second)),
        "first_half": len(first),
        "second_half": len(second),
        "note": (
            "Interleaved split-half check on the control itself. "
            "Failure to reject does not establish homogeneity or exchangeability."
        ),
    }


def domain_shift_check(
    benchmark_lengths: list[float],
    control_lengths: list[float],
) -> dict[str, float | int] | None:
    """Domain check on token lengths, independent of the likelihood score.

    Under a domain-mismatched control, text lengths (a content-free,
    tokenizer-level statistic) differ sharply between benchmark and
    control, and likelihood AUC partly reflects that mismatch.  A
    two-sided Mann-Whitney test with medians and Cohen's d quantifies
    the shift.  ``None`` when either side has fewer than two texts.
    """
    if len(benchmark_lengths) < 2 or len(control_lengths) < 2:
        return None
    import numpy as np

    result = mann_whitney_u_test(
        benchmark_lengths, control_lengths, alternative="two-sided"
    )
    return {
        "benchmark_median_tokens": float(np.median(benchmark_lengths)),
        "control_median_tokens": float(np.median(control_lengths)),
        "mann_whitney_p_value": float(result["p_value"]),
        "cohen_d": float(cohen_d(benchmark_lengths, control_lengths)),
        "note": (
            "Token-length domain check. A large shift warns that the "
            "benchmark-vs-control AUC may partly reflect domain mismatch."
        ),
    }


def _digit_run_values(texts: list[str]) -> list[float]:
    values: list[float] = []
    def append_decimal(run: str) -> None:
        try:
            value = float(run)
        except (ValueError, OverflowError):
            return
        if np.isfinite(value):
            values.append(value)

    for text in texts:
        current = ""
        for char in text:
            if char.isdecimal():
                current += char
            elif current:
                append_decimal(current)
                current = ""
        if current:
            append_decimal(current)
    return values


def domain_fingerprint(
    benchmark_texts: list[str],
    control_texts: list[str],
) -> dict[str, object] | None:
    """Content-free domain fingerprint: do benchmark and control look
    like the same symbol population?

    Two measurements, both independent of the likelihood score:
    character-class fractions per text (digit/upper/lower/punct),
    compared with Mann-Whitney; and the numeric-identifier values
    embedded in the texts (digit runs).  Symbol/code frequency bias
    (e.g. leak IDs 100-139 vs clean IDs 600-639) leaves character
    classes nearly identical but shifts the numeric population
    completely, so the numeric check is what catches it.  Both are
    reported, and a strong numeric shift triggers the caller's caveat.
    ``None`` when a side has fewer than two texts or no digits.
    """
    if len(benchmark_texts) < 2 or len(control_texts) < 2:
        return None
    classes = ("digit", "upper", "lower", "punct")

    def fractions(texts: list[str], pred) -> list[float]:
        out = []
        for text in texts:
            total = max(len(text), 1)
            out.append(sum(1 for ch in text if pred(ch)) / total)
        return out

    class_tests: dict[str, float] = {}
    for name, pred in {
        "digit": str.isdigit,
        "upper": str.isupper,
        "lower": str.islower,
        "punct": lambda ch: ch in ".,:;!?-",
    }.items():
        class_tests[name] = float(
            mann_whitney_u_test(
                fractions(benchmark_texts, pred),
                fractions(control_texts, pred),
                alternative="two-sided",
            )["p_value"]
        )
    # The independent unit is a text, not each digit run within a text.
    # Pooling all runs would give number-heavy examples extra weight and
    # would treat correlated numbers as independent observations.
    bench_runs = _digit_run_values(benchmark_texts)
    control_runs = _digit_run_values(control_texts)
    def finite_median(values: list[float]) -> float:
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        left, right = ordered[middle - 1], ordered[middle]
        return left + (right - left) / 2.0

    bench_digits = [
        finite_median(values)
        for text in benchmark_texts
        if (values := _digit_run_values([text]))
    ]
    control_digits = [
        finite_median(values)
        for text in control_texts
        if (values := _digit_run_values([text]))
    ]
    digit_p: float | None = None
    digit_d: float | None = None
    digit_effect: float | None = None
    if len(bench_digits) >= 2 and len(control_digits) >= 2:
        digit_test = mann_whitney_u_test(
            bench_digits, control_digits, alternative="two-sided"
        )
        digit_p = float(digit_test["p_value"])
        numeric_scale = max(1.0, max(bench_digits), max(control_digits))
        digit_d = float(cohen_d(
            [value / numeric_scale for value in bench_digits],
            [value / numeric_scale for value in control_digits],
        ))
        digit_effect = 2.0 * float(digit_test["auc"]) - 1.0
    return {
        "class_p_values": class_tests,
        "digit_count_benchmark": len(bench_runs),
        "digit_count_control": len(control_runs),
        "numeric_text_count_benchmark": len(bench_digits),
        "numeric_text_count_control": len(control_digits),
        "numeric_test_unit": "per-text median digit-run value",
        "numeric_parser_limit": "Finite decimal digit runs only; nondecimal numeric symbols and unrepresentable values are ignored.",
        "digit_mann_whitney_p_value": digit_p,
        "digit_cohen_d": digit_d,
        "digit_rank_effect": digit_effect,
        "note": (
            "Content-free symbol population check. A strong numeric shift "
            "with otherwise matched classes signals identifier/code bias "
            "between benchmark and control. Absence of a detected shift does "
            "not establish matching."
        ),
    }


def conformal_membership_screen(
    benchmark_scores: list[float],
    control_scores: list[float],
    example_ids: list[str],
    *,
    fdr_level: float = 0.10,
) -> dict[str, object] | None:
    """Conditional per-example score-outlier screen against a control.

    Each benchmark score gets a conformal p-value against the control
    distribution (higher scores count as more suspicious), and the
    Benjamini-Hochberg procedure flags examples at the requested FDR
    level. The conformal null requires clean calibration records and
    exchangeable null scores; BH additionally requires its dependence
    assumptions. Neither condition is verified by a control label. Returns
    ``None`` when the groups are too small or the id alignment is off.
    """
    if len(benchmark_scores) < 2 or len(control_scores) < 2:
        return None
    if len(example_ids) != len(benchmark_scores):
        return None
    if not 0.0 < fdr_level < 1.0:
        raise ValueError("FDR level must be between 0 and 1.")
    p_values = conformal_pvalues(
        benchmark_scores, control_scores, higher_is_more_suspicious=True
    )
    flags, q_values = benjamini_hochberg_flags(
        [float(value) for value in p_values], alpha=fdr_level
    )
    flagged = [
        example_ids[index]
        for index, is_flagged in enumerate(flags)
        if bool(is_flagged)
    ]
    return {
        "method": (
            "conformal p-values against the matched control, "
            "Benjamini-Hochberg FDR"
        ),
        "fdr_level": float(fdr_level),
        "example_ids": list(example_ids),
        "p_values": [float(value) for value in p_values],
        "q_values": [float(value) for value in q_values],
        "flagged_ids": flagged,
        "n_flagged": len(flagged),
        "n_scored": len(example_ids),
        "calibration_count": len(control_scores),
        "minimum_attainable_p_value": 1.0 / (len(control_scores) + 1.0),
        "assumptions": [
            "Control examples are nonmembers of the exposure process being tested.",
            "Null benchmark scores and control scores are exchangeable.",
            "The joint null p-values satisfy the BH dependence conditions.",
        ],
        "note": (
            "Higher Min-K%++ than the control is treated as more suspicious. "
            "Flags are conditional score outliers for follow-up, not proof of membership. "
            "The requested FDR is not an empirically validated contamination error rate."
        ),
    }


def combine_indices(
    index_guided: float | None,
    index_minkpp: float | None,
    index_order: float | None,
) -> CombinedIndex:
    values = {
        "guided": index_guided,
        "minkpp": index_minkpp,
        "order": index_order,
    }
    base_weights = {"guided": 0.5, "minkpp": 0.3, "order": 0.2}
    active = {
        name: base_weights[name]
        for name, value in values.items()
        if value is not None
    }
    if not active:
        return CombinedIndex(None, "none", "insufficient evidence", {})
    total_weight = sum(active.values())
    normalized = {
        name: weight / total_weight for name, weight in active.items()
    }
    score = 0.0
    for name in normalized:
        value = values[name]
        if value is None:
            continue
        score += normalized[name] * value
    names = set(normalized)
    if names == {"guided"}:
        basis = "guided_only"
    elif names == {"guided", "order"}:
        basis = "guided_plus_order"
    elif names == {"guided", "minkpp"}:
        basis = "guided_plus_minkpp"
    elif names == {"guided", "minkpp", "order"}:
        basis = "combined"
    else:
        basis = "_plus_".join(sorted(names))
    if score < 0.2:
        band = "low evidence"
    elif score < 0.5:
        band = "moderate evidence"
    else:
        band = "high evidence"
    return CombinedIndex(clamp(score), basis, band, normalized)


def bootstrap_combined_ci(
    *,
    guided_gaps: list[float] | None,
    minkpp_benchmark: list[float] | None,
    minkpp_control: list[float] | None,
    order_shard_advantages: list[float] | None,
    samples: int = 2000,
    seed: int = 0,
) -> tuple[float, float] | None:
    if sum(value is not None for value in (guided_gaps, minkpp_benchmark, order_shard_advantages)) > 1:
        # Methods share observations. Independent resampling would lose
        # cross-method covariance; the aligned joint design is unimplemented.
        return None
    if guided_gaps is not None and len(guided_gaps) < 2:
        return None
    if minkpp_benchmark is not None and (
        len(minkpp_benchmark) < 2
        or minkpp_control is None
        or len(minkpp_control) < 2
    ):
        return None
    if order_shard_advantages is not None and len(order_shard_advantages) < 2:
        return None
    if not any(
        value is not None
        for value in (guided_gaps, minkpp_benchmark, order_shard_advantages)
    ):
        return None

    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(samples):
        guided_index = None
        minkpp_index = None
        order_index = None
        if guided_gaps is not None:
            selected = rng.choice(guided_gaps, len(guided_gaps), replace=True)
            guided_index = clamp(float(np.mean(selected)))
        if minkpp_benchmark is not None and minkpp_control is not None:
            selected_benchmark = rng.choice(
                minkpp_benchmark, len(minkpp_benchmark), replace=True
            ).tolist()
            selected_control = rng.choice(
                minkpp_control, len(minkpp_control), replace=True
            ).tolist()
            minkpp_index, _ = calibrate_minkpp(
                selected_benchmark, selected_control
            )
        if order_shard_advantages is not None:
            selected_advantages = rng.choice(
                order_shard_advantages,
                len(order_shard_advantages),
                replace=True,
            )
            _, order_index, _ = order_aggregate_index(selected_advantages)
        estimate = combine_indices(
            guided_index, minkpp_index, order_index
        ).contamination_index
        estimates.append(0.0 if estimate is None else estimate)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)

