"""Conservative reporting policy, separate from detector measurements.

These checks can find reasons to withhold an interpretation. They cannot
establish exchangeability, training-set membership, or a contamination FPR.
The policy is deliberately versioned so its coverage can be evaluated on
held-out experiments rather than adjusted silently after seeing results.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from .stats import bonferroni_combination


POLICY_VERSION = "comparison-v1"
DIAGNOSTIC_FAMILY = ("guided", "minkpp", "order")


def combine_diagnostic_pvalues(
    p_values: dict[str, float | None],
) -> dict[str, Any]:
    """Use a fixed family even when evidence gates censor components.

    Replacing a component by 1 can only make this correction more
    conservative. Selecting a smaller denominator after observing the
    component tests does not have that property. Any guarantee still
    requires valid component p-values for their respective nulls.
    """
    unknown = set(p_values) - set(DIAGNOSTIC_FAMILY)
    if unknown:
        raise ValueError(f"Unknown diagnostic methods: {sorted(unknown)}")
    effective = {
        name: 1.0 if p_values.get(name) is None else float(p_values[name])
        for name in DIAGNOSTIC_FAMILY
    }
    corrected = bonferroni_combination(list(effective.values()))
    observed = [name for name in DIAGNOSTIC_FAMILY if p_values.get(name) is not None]
    return {
        "p_value": corrected if observed else None,
        "family": list(DIAGNOSTIC_FAMILY),
        "family_size": len(DIAGNOSTIC_FAMILY),
        "effective_p_values": effective,
        "observed_methods": observed,
        "missing_component_value": 1.0,
        "interpretation": (
            "Bonferroni adjustment over a fixed three-method diagnostic family. "
            "Conditional on each method's null assumptions; this is not a "
            "calibrated test of training contamination."
        ),
    }


def validate_example_ids(examples: Sequence[dict[str, str]], label: str) -> None:
    """Prevent dictionary-based scoring from silently overwriting records."""
    identifiers = [str(item["id"]) for item in examples]
    if any(not identifier.strip() for identifier in identifiers):
        raise ValueError(f"{label} contains an empty example id.")
    duplicates = [key for key, count in Counter(identifiers).items() if count > 1]
    if duplicates:
        raise ValueError(f"{label} contains duplicate example ids: {duplicates[:5]}")


def likelihood_comparison_validity(
    benchmark_texts: Sequence[str],
    control_texts: Sequence[str] | None,
    *,
    benchmark_scored: int,
    control_scored: int,
    min_examples: int,
    domain_shift: dict[str, Any] | None,
    fingerprint: dict[str, Any] | None,
    control_null: dict[str, Any] | None,
    benchmark_scoring_ids: Sequence[str] | None = None,
    control_scoring_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Decide whether to report conditional likelihood inference.

    This is a screening policy with explicitly heuristic thresholds, not
    a learned detector or proof that an unblocked control is suitable.
    Repeated/overlapping text violates the intended distinct-record
    design; do not silently count it as independent calibration evidence.
    """
    reasons: list[dict[str, str]] = []

    def reason(code: str, message: str) -> None:
        reasons.append({"code": code, "message": message})

    normalize = lambda text: " ".join(text.casefold().split())
    benchmark_counts = Counter(normalize(text) for text in benchmark_texts)
    control_counts = Counter(normalize(text) for text in (control_texts or []))
    overlap = len(set(benchmark_counts) & set(control_counts))
    duplicate_benchmark = sum(count - 1 for count in benchmark_counts.values())
    duplicate_control = sum(count - 1 for count in control_counts.values())
    if control_texts is None:
        reason("no_control", "A separate matched control is required.")
    elif benchmark_scored < min_examples or control_scored < min_examples:
        reason("insufficient_scores", f"At least {min_examples} scored examples per side are required.")
    if overlap:
        reason("overlapping_text", f"Benchmark and control share {overlap} normalized text(s).")
    if duplicate_benchmark:
        reason("duplicate_benchmark_text", f"Benchmark repeats {duplicate_benchmark} normalized text record(s).")
    if duplicate_control:
        reason("duplicate_control_text", f"Control repeats {duplicate_control} normalized text record(s).")
    input_identity_checked = benchmark_scoring_ids is not None and control_scoring_ids is not None
    scoring_overlap = 0
    if input_identity_checked:
        benchmark_inputs = Counter(benchmark_scoring_ids)
        control_inputs = Counter(control_scoring_ids)
        scoring_overlap = len(set(benchmark_inputs) & set(control_inputs))
        if scoring_overlap:
            reason("overlapping_scored_inputs", f"Benchmark and control share {scoring_overlap} effective token sequence(s) after scoring transforms/truncation.")
        if any(count > 1 for count in benchmark_inputs.values()) or any(count > 1 for count in control_inputs.values()):
            reason("duplicate_scored_inputs", "Distinct rows repeat an effective scoring token sequence; treating them as independent would inflate evidence.")

    digit_p = (fingerprint or {}).get("digit_mann_whitney_p_value")
    digit_effect = (fingerprint or {}).get("digit_rank_effect")
    if digit_p is not None and digit_effect is not None:
        if float(digit_p) < 0.05 and abs(float(digit_effect)) > 0.5:
            reason(
                "numeric_shift",
                "numeric identifier populations differ strongly between benchmark and control "
                f"(per-text MW p={float(digit_p):.3g}, rank effect={float(digit_effect):.2f}).",
            )
    if domain_shift is not None:
        ratio = float(domain_shift["benchmark_median_tokens"]) / max(
            float(domain_shift["control_median_tokens"]), 1.0
        )
        if float(domain_shift["mann_whitney_p_value"]) < 0.05 and (ratio > 1.5 or ratio < 1.0 / 1.5):
            reason("length_shift", f"Token-length distributions differ strongly (median ratio {ratio:.2f}).")
    if control_null is not None:
        interval = control_null.get("auc_ci")
        excludes_chance = interval and (float(interval[0]) > 0.5 or float(interval[1]) < 0.5)
        if excludes_chance and float(control_null["mann_whitney_p_value"]) < 0.05:
            reason("heterogeneous_control", "The control split-half comparison is inconsistent with a homogeneous control.")

    blocking = [item for item in reasons if item["code"] not in {"no_control", "insufficient_scores"}]
    status = "blocked" if blocking else ("unavailable" if reasons else "conditional")
    return {
        "policy_version": POLICY_VERSION,
        "status": status,
        "eligible": not reasons,
        "reasons": reasons,
        "shared_normalized_texts": overlap,
        "duplicate_benchmark_records": duplicate_benchmark,
        "duplicate_control_records": duplicate_control,
        "scored_input_identity_check": "checked" if input_identity_checked else "unavailable",
        "shared_effective_token_sequences": scoring_overlap,
        "interpretation": (
            "A checked blocker was found; inferential likelihood outputs are withheld."
            if blocking else (
                "The comparison lacks sufficient calibration data."
                if reasons else
                "No checked blocker detected. Exchangeability and control nonmembership remain unverified."
            )
        ),
        "policy_limit": "Heuristic screening thresholds; predictive validity and retained coverage are unvalidated.",
    }
