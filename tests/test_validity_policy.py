"""Regression checks for accusations, abstention, and report parity."""

from dataclasses import replace

import numpy as np
import pytest

from contamcheckr.benchmarks import BenchmarkData
from contamcheckr.calibration import bootstrap_combined_ci, domain_fingerprint
from contamcheckr.config import RunConfig
from contamcheckr.html_report import build_html
from contamcheckr.methods.minkpp import MinKScores
from contamcheckr.model import ModelCapabilities
from contamcheckr.report import _build_refutation_panel, _make_order_scorer, build_report, format_summary
from contamcheckr.validity import combine_diagnostic_pvalues, likelihood_comparison_validity
from contamcheckr.web import DEFAULT_FORM, render_dashboard


def test_censoring_a_component_never_reduces_the_family_size() -> None:
    observed = {"guided": 0.03, "minkpp": None, "order": None}
    result = combine_diagnostic_pvalues(observed)
    assert result["p_value"] == pytest.approx(0.09)
    assert result["family_size"] == 3
    assert result["effective_p_values"] == {"guided": 0.03, "minkpp": 1.0, "order": 1.0}
    assert combine_diagnostic_pvalues({})["p_value"] is None
    with pytest.raises(ValueError):
        combine_diagnostic_pvalues({"guided": -0.1})


def test_fixed_family_remains_conservative_after_signal_selection() -> None:
    # Exhaust a discrete uniform null, including the old p<.05 selection.
    # This is a family-accounting check, not model contamination validation.
    grid = (np.arange(100) + 0.5) / 100
    rejects = 0
    for first in grid:
        for second in grid:
            result = combine_diagnostic_pvalues({
                "minkpp": first, "order": second if second < 0.05 else None,
            })
            rejects += result["p_value"] <= 0.05
    assert rejects / 10_000 <= 0.05


def test_numeric_gate_catches_constant_separated_identifiers() -> None:
    benchmark = [f"Card 100 unique member word {chr(97 + index)}" for index in range(20)]
    control = [f"Card 600 unique control word {chr(97 + index)}" for index in range(20)]
    fingerprint = domain_fingerprint(benchmark, control)
    assert fingerprint["digit_rank_effect"] == -1.0
    decision = likelihood_comparison_validity(
        benchmark, control, benchmark_scored=20, control_scored=20,
        min_examples=4, domain_shift=None, fingerprint=fingerprint, control_null=None,
    )
    assert decision["status"] == "blocked"
    assert "numeric_shift" in [item["code"] for item in decision["reasons"]]


def test_unicode_math_and_oversized_numbers_do_not_abort_diagnostics() -> None:
    import json

    result = domain_fingerprint(["x² + y² = z²", "A " + "9" * 400], ["x³ = y³", "Value 17"])
    json.dumps(result, allow_nan=False)
    assert result["digit_mann_whitney_p_value"] is None
    huge = "1" + "0" * 307
    finite = domain_fingerprint([f"{huge} {huge}"] * 4, ["17 19"] * 4)
    json.dumps(finite, allow_nan=False)


def test_distinct_full_texts_cannot_hide_overlapping_scored_prefixes() -> None:
    result = likelihood_comparison_validity(
        ["shared prefix first tail", "unique b", "unique c", "unique d"],
        ["shared prefix second tail", "unique e", "unique f", "unique g"],
        benchmark_scored=4, control_scored=4, min_examples=4,
        domain_shift=None, fingerprint=None, control_null=None,
        benchmark_scoring_ids=["same-prefix", "b", "c", "d"],
        control_scoring_ids=["same-prefix", "e", "f", "g"],
    )
    assert result["status"] == "blocked"
    assert result["shared_effective_token_sequences"] == 1
    assert any(item["code"] == "overlapping_scored_inputs" for item in result["reasons"])


def test_combined_interval_is_withheld_without_joint_resampling() -> None:
    assert bootstrap_combined_ci(
        guided_gaps=[0.1, 0.3], minkpp_benchmark=[2.0, 3.0],
        minkpp_control=[0.0, 1.0], order_shard_advantages=None,
    ) is None


def test_cached_order_fallback_preserves_the_scoring_token_cap() -> None:
    class Scorer:
        def sequence_loglikelihood(self, text, *, max_tokens=512):
            return -float(max_tokens)

    class MemoryCache:
        values = {}
        def get_sequence_ll(self, key):
            return self.values.get(key)
        def put_sequence_ll(self, key, value):
            self.values[key] = value

    config = RunConfig(model_id="fake", benchmark_source="b", max_tokens=8)
    uncached = _make_order_scorer(Scorer(), config=config, cache=None, cache_model_fingerprint="m", cache_tokenizer_fingerprint="t")
    cached = _make_order_scorer(Scorer(), config=config, cache=MemoryCache(), cache_model_fingerprint="m", cache_tokenizer_fingerprint="t")
    assert cached(["one", "two"]) == uncached(["one", "two"]) == [-8.0, -8.0]
    assert cached(["one"]) == [-8.0]


@pytest.mark.parametrize("problem", ["shared", "duplicates", "length", "too_small"])
def test_control_problems_withhold_the_comparison(problem: str) -> None:
    benchmark = ["alpha", "beta", "gamma", "delta"]
    control = ["one", "two", "three", "four"]
    domain = None
    count = 4
    if problem == "shared":
        control[0] = "  ALPHA  "
    elif problem == "duplicates":
        control[1] = control[0]
    elif problem == "length":
        domain = {"benchmark_median_tokens": 100, "control_median_tokens": 10, "mann_whitney_p_value": 0.001}
    else:
        count = 2
    result = likelihood_comparison_validity(
        benchmark, control, benchmark_scored=count, control_scored=count,
        min_examples=4, domain_shift=domain, fingerprint=None, control_null=None,
    )
    assert result["eligible"] is False
    assert result["status"] in {"blocked", "unavailable"}


class LikelihoodOnly:
    capabilities = ModelCapabilities(False, False, True)


def _benchmark(prefix: str, number: int) -> BenchmarkData:
    return BenchmarkData([
        {"id": f"{prefix}-{index}", "text": f"Archive {number} word {chr(97 + index)}."}
        for index in range(20)
    ], prefix, prefix, "test")


def test_rejected_control_cannot_emit_flags_and_all_outputs_explain_why(monkeypatch) -> None:
    def scores(_model, examples, **kwargs):
        is_benchmark = kwargs.get("id_prefix", "") == ""
        value = 10.0 if is_benchmark else 0.0
        return {
            item["id"]: MinKScores(value, value, 12, 0, value)
            for item in examples
        }, []

    monkeypatch.setattr("contamcheckr.report._score_examples", scores)
    monkeypatch.setattr("contamcheckr.report.perturb_delta_mean", lambda _model, texts, **kwargs: [0.0] * len(texts))
    benchmark, control = _benchmark("benchmark", 100), _benchmark("control", 600)
    config = RunConfig(model_id="offline-fake", benchmark_source="benchmark", bootstrap_samples=10)
    report = build_report(LikelihoodOnly(), benchmark, control, config)
    assert report["comparison_validity"]["status"] == "blocked"
    assert report["index_minkpp"] is None
    assert report["combined_p_value"] is None
    assert report["membership_screen"] is None
    assert report["membership_screen_status"]["status"] == "withheld"
    assert report["minkpp_auc"] == 1.0  # Descriptive evidence is retained.
    assert report["likelihood_baseline"]["loss_auc"] == 1.0
    assert report["raw_method_p_values"]["minkpp"] < 0.05
    assert len(report["per_example"]) == 20
    summary, html, dashboard = format_summary(report), build_html(report), render_dashboard(form=DEFAULT_FORM, report=report)
    for output in (summary, html, dashboard):
        assert "blocked" in output
        assert "numeric identifier" in output
    assert "Withheld:" in html and "Withheld:" in dashboard
    assert report["manifest"]["benchmark_ordered_hash"]
    assert report["manifest"]["source_snapshot"]["sha256"]


def test_duplicate_ids_fail_before_scoring() -> None:
    benchmark = _benchmark("benchmark", 100)
    duplicate = replace(benchmark, examples=[benchmark.examples[0], benchmark.examples[0]])
    with pytest.raises(ValueError, match="duplicate example ids"):
        build_report(LikelihoodOnly(), duplicate, None, RunConfig(model_id="fake", benchmark_source="b"))


def test_unmeasured_panel_never_claims_contamination_is_best_explanation() -> None:
    panel = _build_refutation_panel(
        domain_shift=None, fingerprint=None, control_null=None, guided_result=None,
        repro_threshold=0.6, minkpp_auc=None, loss_auc=None, combined_p_value=0.001,
    )
    assert all(item["verdict"] == "unmeasured" for item in panel)
    assert not any("best-supported" in item["explanation"] for item in panel)


def test_interpretation_context_escapes_diagnostic_text(monkeypatch) -> None:
    from contamcheckr.html_report import evidence_context_html

    html = evidence_context_html({"comparison_validity": {
        "status": "blocked", "interpretation": "<script>evil</script>",
        "reasons": [{"message": "<img src=x>"}],
    }})
    assert "<script>" not in html and "<img" not in html
    assert "&lt;script&gt;" in html
