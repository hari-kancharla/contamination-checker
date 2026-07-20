"""Tests for the batched engine and statistical synthesis surfaces."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from contamcheckr.benchmarks import load_benchmark
from contamcheckr.calibration import (
    auc_posterior_mean,
    conformal_membership_screen,
    control_null_check,
    minkpp_diagnostics,
)
from contamcheckr.cli import _parser
from contamcheckr.config import RunConfig
from contamcheckr.methods.guided import run_guided_overlap
from contamcheckr.methods.minkpp import score_text, score_text_batch
from contamcheckr.model import ModelCapabilities, load_model
from contamcheckr.methods.ngram import overlap_score, score_corpus
from contamcheckr.report import build_report, format_summary
from contamcheckr.stats import effective_sample_size, tpr_at_fpr

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "experiments"))
from evaluate import evaluate_scores  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TINY_MODEL = ROOT / "experiments" / "models" / "sshleifer--tiny-gpt2"


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text]

    def decode(
        self, token_ids: list[int], skip_special_tokens: bool = True
    ) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class FakeModel:
    """Compatibility fake: no batch methods, no fingerprints, no cache."""

    tokenizer = CharacterTokenizer()
    capabilities = ModelCapabilities(True, True, True)

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        return "plain deterministic completion"

    def token_logprob_table(
        self, text: str, *, max_tokens: int = 512
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = min(max(2, len(text) // 8), max_tokens)
        targets = torch.tensor(
            [(ord(character) % 5) for character in text[:count]],
            dtype=torch.long,
        )
        if len(targets) < count:
            targets = torch.cat(
                [targets, torch.zeros(count - len(targets), dtype=torch.long)]
            )
        rows = torch.arange(count, dtype=torch.float32).unsqueeze(1)
        columns = torch.arange(5, dtype=torch.float32).unsqueeze(0)
        logits = -torch.abs(columns - ((rows + len(text)) % 5))
        return logits, targets

    def sequence_loglikelihood(self, text: str, *, max_tokens: int = 512) -> float:
        parts = text.split("\n\n---\n\n")
        return -float(
            sum(
                left > right
                for index, left in enumerate(parts)
                for right in parts[index + 1 :]
            )
        )


def _config(
    *,
    model_id: str = "fake",
    limit: int = 10,
    control: bool = True,
    **overrides: object,
) -> RunConfig:
    kwargs: dict[str, object] = {
        "model_id": model_id,
        "benchmark_source": "sample",
        "control_source": "sample" if control else None,
        "limit": limit,
        "shard_count": 2,
        "permutation_count": 2,
    }
    kwargs.update(overrides)
    return RunConfig(**kwargs)  # type: ignore[arg-type]


def test_tpr_at_fpr_known() -> None:
    separated = tpr_at_fpr([10.0, 11.0], list(range(10)), rates=(0.05, 0.10))
    assert separated["tpr_at_fpr_0.05"] == pytest.approx(1.0)
    assert separated["tpr_at_fpr_0.1"] == pytest.approx(1.0)
    overlapping = tpr_at_fpr([0.0, 5.0], [1.0, 2.0, 3.0], rates=(0.5,))
    assert overlapping["tpr_at_fpr_0.5"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        tpr_at_fpr([1.0], [0.0], rates=(0.0,))


def test_bonferroni_fusion_not_vetoed_by_null_component() -> None:
    from contamcheckr.stats import bonferroni_combination

    assert bonferroni_combination([1.0, 3e-05]) == pytest.approx(6e-05)
    assert bonferroni_combination([0.05]) == pytest.approx(0.05)


def test_evaluate_scores_perfect_separation() -> None:
    evaluation = evaluate_scores(
        [10.0, 11.0, 12.0],
        [float(value) for value in range(10)],
        ["a", "b", "c"],
    )
    assert evaluation["auc"] == pytest.approx(1.0)
    assert evaluation["tpr_at_fpr"]["tpr_at_fpr_0.1"] == pytest.approx(1.0)
    assert evaluation["screen"]["flagged_ids"] == ["a", "b", "c"]


def test_ngram_overlap() -> None:
    text = " ".join(f"w{index}" for index in range(16))
    reference = {text}
    assert score_corpus([text], list(reference)) == [1.0]
    assert score_corpus(
        [" ".join(["a"] * 20)], list(reference)
    ) == [0.0]
    assert overlap_score("short text", set()) == 0.0
    # Leave-one-out: a lone text must not match itself.
    assert score_corpus([text], [text], leave_one_out=True) == [0.0]


def test_report_ngram_and_control_vectors_with_fake_model() -> None:
    benchmark = load_benchmark("sample", limit=10)
    control = load_benchmark("sample", limit=10, sample_control=True)
    report = build_report(FakeModel(), benchmark, control, _config())
    assert report["score_baselines"]["ngram_auc"] is not None
    assert len(report["control_per_example"]) == 10
    assert report["per_example"][0]["zlib"] is not None


def test_effective_sample_size_formula() -> None:
    # n / (1 + (n - 1) * rho) with n=10, rho=0.5
    assert effective_sample_size(10, [0.5] * 10) == pytest.approx(10 / 5.5)
    assert effective_sample_size([0.5] * 10) == pytest.approx(10 / 5.5)
    assert effective_sample_size([0.0, 0.0]) is None
    assert effective_sample_size(1, [0.5]) is None


def test_auc_posterior_mean_shrinks_small_samples() -> None:
    assert auc_posterior_mean([0.8, 0.9], [0.1, 0.2]) == pytest.approx(0.9)
    assert auc_posterior_mean([], [0.1]) is None
    assert auc_posterior_mean([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.5)


def test_domain_fingerprint_catches_code_bias_not_matched_text() -> None:
    from contamcheckr.calibration import domain_fingerprint

    leak_texts = [
        f"Archive card LQ-{number} records the amber dial faces north."
        for number in range(100, 140)
    ]
    clean_texts = [
        f"Archive card CQ-{number} records the amber dial faces north."
        for number in range(600, 640)
    ]
    biased = domain_fingerprint(leak_texts, clean_texts)
    assert biased is not None
    assert biased["digit_mann_whitney_p_value"] < 0.05
    assert abs(biased["digit_cohen_d"]) > 5.0
    sample_bench = load_benchmark("sample", limit=10).examples
    sample_ctrl = load_benchmark("sample", limit=10, sample_control=True).examples
    matched = domain_fingerprint(
        [item["text"] for item in sample_bench],
        [item["text"] for item in sample_ctrl],
    )
    assert matched is not None
    assert matched["digit_mann_whitney_p_value"] is None


def test_report_domain_fingerprint_fires_on_confounded_cards() -> None:
    leak = load_benchmark("experiments/data/leak_set.jsonl", limit=10)
    clean = load_benchmark("experiments/data/clean_set.jsonl", limit=10)
    report = build_report(FakeModel(), leak, clean, _config())
    assert report["domain_fingerprint"]["digit_mann_whitney_p_value"] < 0.05
    assert any("Withheld: numeric identifier" in caveat for caveat in report["caveats"])
    assert report["index_suppressed_reason"] is not None
    assert report["index_minkpp"] is None
    assert "insufficient evidence" in report["band"] or report["contamination_index"] is None
    sample = load_benchmark("sample", limit=10)
    sample_ctrl = load_benchmark("sample", limit=10, sample_control=True)
    calm = build_report(FakeModel(), sample, sample_ctrl, _config())
    assert not any("numeric identifier" in caveat.lower() for caveat in calm["caveats"])


def test_domain_shift_check() -> None:
    from contamcheckr.calibration import domain_shift_check

    distinct = domain_shift_check(
        [20.0, 21.0, 19.0, 20.0, 20.0, 21.0, 19.0, 20.0],
        [3.0, 4.0, 2.0, 3.0, 3.0, 4.0, 2.0, 3.0],
    )
    assert distinct is not None
    assert distinct["mann_whitney_p_value"] < 0.05
    assert distinct["cohen_d"] > 2.0
    matched = domain_shift_check([10.0, 11.0, 12.0, 13.0], [10.0, 11.0, 12.0, 13.0])
    assert matched is not None
    assert matched["mann_whitney_p_value"] == pytest.approx(1.0)
    assert domain_shift_check([5.0], [5.0, 6.0]) is None


def test_control_null_check() -> None:
    homogeneous = control_null_check(
        [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
    )
    assert homogeneous is not None
    assert homogeneous["auc"] == pytest.approx(0.5)
    heterogeneous = control_null_check(
        [1.0, 10.0, 2.0, 11.0, 3.0, 12.0, 4.0, 13.0]
    )
    assert heterogeneous is not None
    assert heterogeneous["auc"] == pytest.approx(0.0)
    assert control_null_check([1.0, 2.0]) is None


def test_minkpp_diagnostics_flags_separation() -> None:
    diagnostics = minkpp_diagnostics(
        [0.8, 0.9, 0.7, 0.85], [0.1, 0.2, 0.3, 0.15]
    )
    assert diagnostics is not None
    assert diagnostics["auc"] == pytest.approx(1.0)
    assert diagnostics["auc_ci"][0] <= 1.0 <= diagnostics["auc_ci"][1]
    assert diagnostics["mann_whitney_p_value"] < 0.05
    assert diagnostics["cohen_d"] > 0.0
    assert minkpp_diagnostics([0.5], [0.1, 0.2]) is None


def test_conformal_screen_flags_clear_outliers() -> None:
    control = [float(value) for value in range(10)]
    screen = conformal_membership_screen(
        [10.0, 11.0], control, ["a", "b"], fdr_level=0.10
    )
    assert screen is not None
    assert screen["flagged_ids"] == ["a", "b"]
    assert screen["n_flagged"] == 2
    below = conformal_membership_screen(
        [0.0, 1.0], [10.0, 11.0, 12.0], ["c", "d"], fdr_level=0.10
    )
    assert below is not None
    assert below["flagged_ids"] == []


def test_guided_mismatched_control_and_stability_with_fake_model() -> None:
    examples = load_benchmark("sample", limit=4).examples
    result = run_guided_overlap(
        FakeModel(),
        examples,
        dataset_name="demo",
        split_name="test",
        stability=True,
    )
    assert result["mismatched_rouge"] is not None
    # The fake returns one constant completion, so every arm ties at ROUGE 0
    # against these suffixes and both gaps are exactly zero.
    assert result["guided_mismatched_gap"] == pytest.approx(0.0)
    assert result["guided_general_gap"] == pytest.approx(0.0)
    assert result["split_stability"] == pytest.approx(
        {"0.25": 0.0, "0.75": 0.0}
    )
    for row in result["per_example"]:  # type: ignore[union-attr]
        assert "mismatched_rouge" in row


def test_report_control_null_and_score_baselines() -> None:
    benchmark = load_benchmark("sample", limit=10)
    control = load_benchmark("sample", limit=10, sample_control=True)
    report = build_report(
        FakeModel(), benchmark, control, _config(guided_stability=True)
    )
    assert report["control_null"] is not None
    assert 0.0 <= report["control_null"]["auc"] <= 1.0
    assert report["score_baselines"]["zlib_auc"] is not None
    assert report["score_baselines"]["entropy_auc"] is not None
    assert report["score_baselines"]["mean_rank_auc"] is not None
    assert report["minkpp_summary"]["auc_smoothed"] is not None
    assert report["guided_split_stability"] == pytest.approx(
        {"0.25": 0.0, "0.75": 0.0}
    )
    assert report["guided_mismatched_gap"] == pytest.approx(0.0)


def test_min_calibration_gate_withholds_tiny_minkpp() -> None:
    benchmark = load_benchmark("sample", limit=2)
    control = load_benchmark("sample", limit=2, sample_control=True)
    report = build_report(FakeModel(), benchmark, control, _config(limit=2))
    assert report["index_minkpp"] is None
    assert report["minkpp_auc"] is None
    assert "Fewer than" in report["minkpp_summary"]["uncalibrated_reason"]
    assert report["index_basis"] == "none"
    assert report["band"] == "insufficient evidence"
    assert report["combined_p_value"] is None
    assert report["combined_p_value_basis"] == "none"


def test_combined_p_value_fuses_active_methods() -> None:
    benchmark = load_benchmark("sample", limit=10)
    control = load_benchmark("sample", limit=10, sample_control=True)
    report = build_report(FakeModel(), benchmark, control, _config())
    assert report["method_p_values"]["minkpp"] is not None
    assert report["combined_p_value_basis"] == "minkpp"
    assert report["combined_p_value"] == pytest.approx(
        min(1.0, 3 * report["method_p_values"]["minkpp"])
    )
    assert 0.0 <= report["combined_p_value"] <= 1.0


def test_additive_report_surfaces_with_fake_model() -> None:
    benchmark = load_benchmark("sample", limit=10)
    control = load_benchmark("sample", limit=10, sample_control=True)
    report = build_report(FakeModel(), benchmark, control, _config())
    assert report["manifest"]["model_fingerprint"] == "fake"
    assert report["manifest"]["cache"]["enabled"] is False
    assert report["manifest"]["scoring_engine"]["fingerprinted"] is False
    assert report["minkpp_summary"]["auc_ci"] is not None
    assert report["minkpp_summary"]["mann_whitney_p_value"] is not None
    screen = report["membership_screen"]
    assert screen is not None
    assert screen["n_scored"] == 10
    assert len(screen["p_values"]) == 10
    assert report["planning"]["minimum_detectable_effect"] is not None
    assert report["planning"]["benchmark_scored"] == 10
    for row in report["per_example"]:
        assert "loss" in row and "mink_score" in row and "truncated" in row
    assert report["order_summary"]["effect_size"] is not None
    summary = format_summary(report)
    assert "Membership screen:" in summary
    assert "Min-K%++ AUC 95% interval:" in summary


def test_cli_run_flags_parse_and_version() -> None:
    args = _parser().parse_args(
        [
            "run",
            "--model", "m",
            "--benchmark", "b",
            "--max-tokens", "7",
            "--shard-count", "12",
            "--no-cache",
            "--conformal-fdr", "0.05",
        ]
    )
    assert args.max_tokens == 7
    assert args.shard_count == 12
    assert args.no_cache is True
    assert args.conformal_fdr == pytest.approx(0.05)
    with pytest.raises(SystemExit):
        _parser().parse_args(["--version"])


@pytest.fixture(scope="module")
def tiny_model():
    if not TINY_MODEL.exists():
        pytest.skip("Local tiny model fixture is missing.")
    return load_model(str(TINY_MODEL), device="cpu", local_files_only=True)


def test_sequence_loglikelihood_runs_on_mps_when_available() -> None:
    if not TINY_MODEL.exists() or not torch.backends.mps.is_available():
        pytest.skip("MPS is not available for this regression check.")
    model = load_model(str(TINY_MODEL), device="mps", local_files_only=True)
    values = model.sequence_loglikelihood_batch(
        ["alpha beta gamma", "one two three four five"],
        max_tokens=32,
        batch_size=2,
    )
    assert all(np.isfinite(value) for value in values)


def test_batched_minkpp_matches_single_scoring(tiny_model) -> None:
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Archive card LQ-100 records that the amber dial faces north.",
        "A triangle has three sides and three interior angles.",
    ]
    batched = score_text_batch(
        tiny_model, texts, k=20.0, max_tokens=64, batch_size=3
    )
    singles = [
        score_text(tiny_model, text, k=20.0, max_tokens=64) for text in texts
    ]
    for left, right in zip(batched, singles, strict=True):
        assert left.minkpp == pytest.approx(right.minkpp, abs=1e-5)
        assert left.loss == pytest.approx(right.loss, abs=1e-5)


def test_guided_batched_generation_matches_single(tiny_model) -> None:
    examples = load_benchmark("sample", limit=4).examples
    single = run_guided_overlap(
        tiny_model,
        examples,
        dataset_name="demo",
        split_name="test",
        generation_batch_size=1,
    )
    batched = run_guided_overlap(
        tiny_model,
        examples,
        dataset_name="demo",
        split_name="test",
        generation_batch_size=4,
    )
    assert batched["guided_rouge"] == pytest.approx(single["guided_rouge"], abs=1e-9)
    assert batched["general_rouge"] == pytest.approx(
        single["general_rouge"], abs=1e-9
    )
    assert len(batched["per_example"]) == len(single["per_example"])  # type: ignore[arg-type]
    assert [item["id"] for item in batched["per_example"]] == [  # type: ignore[arg-type]
        item["id"] for item in single["per_example"]  # type: ignore[arg-type]
    ]


def test_perturb_transforms_deterministic() -> None:
    from contamcheckr.methods.perturb import TRANSFORMS, mutate_number_shift

    assert mutate_number_shift("LQ-100 and 12 items") == "LQ-107 and 19 items"
    for fn in TRANSFORMS.values():
        assert fn("Alpha beta. Gamma delta.") == fn("Alpha beta. Gamma delta.")


def test_refutation_panel_present_with_fake_model() -> None:
    benchmark = load_benchmark("sample", limit=10)
    control = load_benchmark("sample", limit=10, sample_control=True)
    report = build_report(FakeModel(), benchmark, control, _config())
    panel = report["refutation_panel"]
    assert isinstance(panel, list)
    assert len(panel) == 6
    for item in panel:
        assert set(item) == {"explanation", "evidence", "verdict"}
        assert item["verdict"] in {"plausible", "not supported", "unmeasured"}
    assert report["score_baselines"]["perturb_auc"] is not None


def test_refutation_panel_flags_code_bias_on_cards() -> None:
    leak = load_benchmark("experiments/data/leak_set.jsonl", limit=10)
    clean = load_benchmark("experiments/data/clean_set.jsonl", limit=10)
    report = build_report(FakeModel(), leak, clean, _config())
    code_entry = next(
        item
        for item in report["refutation_panel"]
        if "Symbol/code" in item["explanation"]
    )
    assert code_entry["verdict"] == "plausible"


def test_canonicalize_flag_recorded_and_deterministic() -> None:
    benchmark = load_benchmark("sample", limit=6)
    control = load_benchmark("sample", limit=6, sample_control=True)
    plain = build_report(FakeModel(), benchmark, control, _config())
    assert plain["manifest"]["settings"]["canonicalize_scores"] is False
    canon = build_report(
        FakeModel(), benchmark, control, _config(canonicalize_scores=True)
    )
    assert canon["manifest"]["settings"]["canonicalize_scores"] is True
    repeat = build_report(
        FakeModel(), benchmark, control, _config(canonicalize_scores=True)
    )
    assert canon["minkpp_auc"] == repeat["minkpp_auc"]
    assert len(canon["per_example"]) == len(plain["per_example"])


def test_windowed_scoring_mode_changes_long_text_scores(tiny_model) -> None:
    long_benchmark = load_benchmark("sample", limit=2)
    for example in long_benchmark.examples:
        example["text"] = "Archive card LQ-100 records that the amber dial faces north. " * 12
    control = load_benchmark("sample", limit=2, sample_control=True)
    truncated = build_report(
        tiny_model,
        long_benchmark,
        control,
        _config(model_id="tiny", limit=2, max_tokens=64, scoring_mode="truncated"),
    )
    windowed = build_report(
        tiny_model,
        long_benchmark,
        control,
        _config(model_id="tiny", limit=2, max_tokens=64, scoring_mode="windowed", window_tokens=64),
    )
    assert truncated["manifest"]["settings"]["scoring_mode"] == "truncated"
    assert windowed["manifest"]["settings"]["scoring_mode"] == "windowed"
    assert not any(
        row["truncated"] for row in windowed["per_example"] if row["truncated"] is not None
    )
    # Windowed mode scores every token: minkpp differs from truncated scoring,
    # which only sees the first 64 tokens (verified at the unit level too).
    assert truncated["per_example"][0]["minkpp_score"] != pytest.approx(
        windowed["per_example"][0]["minkpp_score"], abs=1e-9
    )


def test_cache_roundtrip_preserves_report_scores(
    tiny_model, tmp_path: Path
) -> None:
    cache_file = tmp_path / "scores.sqlite3"
    benchmark = load_benchmark("sample", limit=10)
    control = load_benchmark("sample", limit=10, sample_control=True)
    fields: dict[str, object] = {
        "cache_path": str(cache_file),
        "shard_count": 2,
        "permutation_count": 2,
        "seed": 11,
    }
    first = build_report(
        tiny_model,
        benchmark,
        control,
        _config(model_id="tiny", limit=10, **fields),
    )
    second = build_report(
        tiny_model,
        benchmark,
        control,
        _config(model_id="tiny", limit=10, **fields),
    )
    assert first["manifest"]["cache"]["enabled"] is True
    assert first["manifest"]["cache"]["stats"]["token_stats"] > 0
    assert first["contamination_index"] == second["contamination_index"]
    assert first["minkpp_auc"] == pytest.approx(second["minkpp_auc"])
    assert first["confidence_interval"] == second["confidence_interval"]
    assert np.allclose(
        [row["minkpp_score"] for row in first["per_example"]],
        [row["minkpp_score"] for row in second["per_example"]],
    )
