import csv
import importlib.metadata
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from .cache import NullCache, cache_key, open_cache
from .calibration import (
    auc_posterior_mean,
    bootstrap_combined_ci,
    calibrate_minkpp,
    combine_indices,
    conformal_membership_screen,
    control_null_check,
    domain_fingerprint,
    domain_shift_check,
    minkpp_diagnostics,
)
from .config import RunConfig
from .hashing import config_fingerprint, hash_examples, hash_ordered_examples, hash_text, source_snapshot
from .methods.canonical_order import canonical_order_test
from .methods.guided import run_guided_overlap
from .methods.minkpp import MinKScores, score_text, score_token_stats_text
from .methods.canonicalize import canonicalize
from .methods.perturb import perturb_delta_mean
from .methods.token_stats import TokenStats
from .methods.windowed import windowed_token_stats, windowed_token_stats_batch
from .model import LocalCausalLM
from .stats import minimum_detectable_effect, paired_permutation_test
from .validity import combine_diagnostic_pvalues, likelihood_comparison_validity, validate_example_ids


def _versions() -> dict[str, str]:
    names = ("contamcheckr", "torch", "transformers", "datasets", "numpy")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def _distribution(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "standard_deviation": float(np.std(array)),
    }


def _scoring_capabilities(model: Any) -> dict[str, Any]:
    """Surface the batched engine capabilities of a model, if present."""

    def _has(name: str) -> bool:
        return callable(getattr(model, name, None))

    return {
        "batched_token_stats": _has("token_stats_batch"),
        "batched_sequence_ll": _has("sequence_loglikelihood_batch"),
        "batched_generation": _has("generate_batch"),
        "chat_template": _has("format_prompt"),
        "fingerprinted": bool(getattr(model, "fingerprint", None)),
    }


def _scored_input_hashes(model: Any, texts: list[str], config: RunConfig) -> list[str] | None:
    """Mirror local scorer tokenization to detect duplicate truncated inputs.

    Custom adapters need their own explicit input-identity contract; do
    not assume their tokenizer proxy matches the scoring backend.
    """
    if not isinstance(model, LocalCausalLM):
        return None
    try:
        if config.scoring_mode == "windowed":
            inputs = [model.tokenizer.encode(text, add_special_tokens=False) for text in texts]
        else:
            inputs = [model.tokenizer(text, truncation=True, max_length=config.max_tokens)["input_ids"] for text in texts]
        return [hash_text(json.dumps(ids, separators=(",", ":"))) for ids in inputs]
    except (ValueError, TypeError, AttributeError, KeyError):
        return None


def _open_cache(config: RunConfig) -> tuple[Any, str | None]:
    """Open the content-addressed score cache, or a no-op fallback.

    Returns ``(cache, error)``.  A missing or unwritable cache directory
    never fails an audit; the error is surfaced as a caveat instead.
    """
    if not config.cache_enabled:
        return NullCache(), None
    try:
        return open_cache(enabled=True, path=config.cache_path), None
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        return NullCache(), str(error)


def _score_examples(
    model: Any,
    examples: list[dict[str, str]],
    *,
    config: RunConfig,
    cache: Any | None,
    cache_model_fingerprint: str,
    cache_tokenizer_fingerprint: str,
    id_prefix: str = "",
) -> tuple[dict[str, MinKScores], list[str]]:
    """Score examples with the batched engine and cache, isolating failures.

    Returns ``(scores_by_id, failures)``.  One unscorable text never poisons
    its batch mates: the failed chunk is rescored one text at a time.
    """
    scores: dict[str, MinKScores] = {}
    failures: list[str] = []
    if config.canonicalize_scores:
        examples = [
            {"id": item["id"], "text": canonicalize(item["text"])} for item in examples
        ]
    use_windowed = config.scoring_mode == "windowed" and all(
        hasattr(model, attr) for attr in ("tokenizer", "model", "device")
    )
    if config.scoring_mode == "windowed" and not use_windowed:
        return {}, [f"{id_prefix}{item['id']}: windowed scoring requires a local model, tokenizer, and device" for item in examples]
    batch_capable = use_windowed or callable(getattr(model, "token_stats_batch", None))

    def make_cache_key(text: str) -> str:
        return cache_key(
            cache_model_fingerprint,
            cache_tokenizer_fingerprint,
            hash_text(text),
            config.max_tokens,
            config.scoring_mode,
            config.window_tokens,
            config.window_stride_tokens,
        )

    pending: list[dict[str, str]] = []
    for example in examples:
        example_id = example["id"]
        if cache is not None and batch_capable and cache_model_fingerprint:
            cache_key_value = make_cache_key(example["text"])
            try:
                payload = cache.get_token_stats(cache_key_value)
            except Exception:
                payload = None
            if payload is not None:
                try:
                    stats = TokenStats.from_dict(payload)
                    scores[example_id] = score_token_stats_text(
                        stats, example["text"], k=config.k
                    )
                    continue
                except Exception:
                    scores.pop(example_id, None)
        pending.append(example)

    batch_size = max(1, config.scoring_batch_size)

    def score_single(item: dict[str, str]) -> None:
        try:
            if use_windowed:
                stats = windowed_token_stats(
                    model, item["text"], window_tokens=config.window_tokens,
                    stride_tokens=config.window_stride_tokens,
                )
                scores[item["id"]] = score_token_stats_text(stats, item["text"], k=config.k)
            else:
                scores[item["id"]] = score_text(
                    model, item["text"], k=config.k, max_tokens=config.max_tokens,
                )
        except (ValueError, TypeError, RuntimeError) as error:
            failures.append(f"{id_prefix}{item['id']}: {error}")

    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        chunk_texts = [item["text"] for item in chunk]
        if not batch_capable:
            for item in chunk:
                score_single(item)
            continue
        try:
            if use_windowed:
                stats_list = windowed_token_stats_batch(
                    model,
                    chunk_texts,
                    window_tokens=config.window_tokens,
                    stride_tokens=config.window_stride_tokens,
                    batch_size=batch_size,
                )
            else:
                stats_list = model.token_stats_batch(
                    chunk_texts,
                    max_tokens=config.max_tokens,
                    batch_size=batch_size,
                )
        except (ValueError, TypeError, RuntimeError):
            for item in chunk:
                score_single(item)
            continue
        if len(stats_list) != len(chunk):
            for item in chunk:
                score_single(item)
            continue
        for item, stats in zip(chunk, stats_list, strict=True):
            try:
                scores[item["id"]] = score_token_stats_text(
                    stats, item["text"], k=config.k
                )
            except (ValueError, TypeError, RuntimeError) as error:
                failures.append(f"{id_prefix}{item['id']}: {error}")
                continue
            if cache is not None and cache_model_fingerprint:
                try:
                    cache.put_token_stats(make_cache_key(item["text"]), stats.to_dict())
                except Exception:
                    continue
    return scores, failures


def _make_order_scorer(
    model: Any,
    *,
    config: RunConfig,
    cache: Any | None,
    cache_model_fingerprint: str,
    cache_tokenizer_fingerprint: str,
) -> Any:
    """Return a batch-capable sequence scorer with cache support."""
    batch_ll = getattr(model, "sequence_loglikelihood_batch", None)

    def score_many(texts: list[str]) -> list[float]:
        if (
            cache is not None
            and cache_model_fingerprint
            and callable(getattr(model, "sequence_loglikelihood", None))
        ):
            results: list[float | None] = [None] * len(texts)
            miss_indexes: list[int] = []
            miss_keys: list[str | None] = []
            for index, text in enumerate(texts):
                key = cache_key(
                    cache_model_fingerprint,
                    cache_tokenizer_fingerprint,
                    "sequence_ll",
                    hash_text(text),
                    config.max_tokens,
                )
                try:
                    cached = cache.get_sequence_ll(key)
                except Exception:
                    cached = None
                if cached is not None:
                    results[index] = float(cached)
                else:
                    miss_indexes.append(index)
                    miss_keys.append(key)
            if miss_indexes:
                miss_texts = [texts[index] for index in miss_indexes]
                if callable(batch_ll):
                    miss_values = list(
                        batch_ll(
                            miss_texts,
                            max_tokens=config.max_tokens,
                            batch_size=config.scoring_batch_size,
                        )
                    )
                else:
                    miss_values = [
                        float(model.sequence_loglikelihood(text, max_tokens=config.max_tokens))
                        for text in miss_texts
                    ]
                for index, key, value in zip(
                    miss_indexes, miss_keys, miss_values, strict=True
                ):
                    results[index] = float(value)
                    if key is not None:
                        try:
                            cache.put_sequence_ll(key, float(value))
                        except Exception:
                            continue
            return [float(value) for value in results if value is not None]

        if callable(batch_ll):
            return [
                float(value)
                for value in batch_ll(
                    list(texts),
                    max_tokens=config.max_tokens,
                    batch_size=config.scoring_batch_size,
                )
            ]
        return [
            float(model.sequence_loglikelihood(text, max_tokens=config.max_tokens))
            for text in texts
        ]

    return score_many


def _build_refutation_panel(
    *,
    domain_shift: dict[str, Any] | None,
    fingerprint: dict[str, Any] | None,
    control_null: dict[str, Any] | None,
    guided_result: dict[str, Any] | None,
    repro_threshold: float,
    minkpp_auc: float | None,
    loss_auc: float | None,
    combined_p_value: float | None,
) -> list[dict[str, str]]:
    """ Ranked alternative explanations for the evidence.

    Each entry pairs an explanation of the signal with its measured
    magnitude and a verdict: "plausible" (the numbers support the
    alternative), "not supported" (measured and weak), or "unmeasured".
    The panel never feeds the index; it exists so the strongest cheap
    alternative is read first.
    """
    panel: list[dict[str, str]] = []

    def entry(explanation: str, evidence: str, verdict: str) -> None:
        panel.append(
            {
                "explanation": explanation,
                "evidence": evidence,
                "verdict": verdict,
            }
        )

    if domain_shift is not None:
        ratio = float(domain_shift["benchmark_median_tokens"]) / max(
            float(domain_shift["control_median_tokens"]), 1.0
        )
        p_value = float(domain_shift["mann_whitney_p_value"])
        verdict = (
            "plausible"
            if p_value < 0.05 and (ratio > 1.5 or ratio < 1.0 / 1.5)
            else "not supported"
        )
        entry(
            "Domain mismatch (token-length distribution)",
            f"median ratio {ratio:.2f}, MW p={p_value:.3g}",
            verdict,
        )
    else:
        entry("Domain mismatch (token-length distribution)", "not scored", "unmeasured")

    digit_p = (fingerprint or {}).get("digit_mann_whitney_p_value")
    digit_effect = (fingerprint or {}).get("digit_rank_effect")
    if digit_p is not None and digit_effect is not None:
        verdict = (
            "plausible"
            if float(digit_p) < 0.05 and abs(float(digit_effect)) > 0.5
            else "not supported"
        )
        entry(
            "Symbol/code bias (numeric identifier populations)",
            f"per-text digit MW p={float(digit_p):.3g}, rank effect={float(digit_effect):.2f}",
            verdict,
        )
    else:
        entry("Symbol/code bias (numeric identifier populations)", "not scored", "unmeasured")

    if control_null is not None:
        null_ci = control_null.get("auc_ci")
        null_p = float(control_null.get("mann_whitney_p_value", 1.0))
        excludes = bool(null_ci) and (
            float(null_ci[0]) > 0.5 or float(null_ci[1]) < 0.5
        )
        verdict = "plausible" if excludes and null_p < 0.05 else "not supported"
        entry(
            "Domain-heterogeneous control (split-half null)",
            f"split-half AUC={float(control_null.get('auc', 0.5)):.3f}, MW p={null_p:.3g}",
            verdict,
        )
    else:
        entry("Domain-heterogeneous control (split-half null)", "not scored", "unmeasured")

    if guided_result is not None:
        mismatch_gap = guided_result.get("guided_mismatched_gap")
        guided_high = float(guided_result.get("guided_rouge") or 0.0) >= repro_threshold
        if mismatch_gap is not None:
            verdict = (
                "plausible"
                if guided_high and abs(float(mismatch_gap)) < 0.05
                else "not supported"
            )
            entry(
                "Format/prior bias (guided with wrong prefix)",
                f"mismatched gap={float(mismatch_gap):.3f}",
                verdict,
            )
        else:
            entry("Format/prior bias (guided with wrong prefix)", "not scored", "unmeasured")
    else:
        entry("Format/prior bias (guided with wrong prefix)", "not scored", "unmeasured")

    if minkpp_auc is not None and loss_auc is not None:
        gap = float(minkpp_auc) - float(loss_auc)
        verdict = "plausible" if gap < 0.05 else "unmeasured"
        entry(
            "No demonstrated advantage over mean log-probability",
            f"Min-K%++ minus likelihood AUC={gap:.3f}; no paired superiority test",
            verdict,
        )
    else:
        entry("No demonstrated advantage over mean log-probability", "not scored", "unmeasured")
    entry(
        "Training exposure and benchmark-performance inflation",
        "These diagnostics do not identify training inclusion, rule out unmeasured confounds, or measure task-performance inflation.",
        "unmeasured",
    )
    return panel


def build_report(
    model: Any,
    benchmark: Any,
    control: Any | None,
    config: RunConfig,
) -> dict[str, Any]:
    validate_example_ids(benchmark.examples, "Benchmark")
    if control is not None:
        validate_example_ids(control.examples, "Control")
    capabilities = model.capabilities.as_dict()
    scoring_engine = _scoring_capabilities(model)
    caveats = [
        "Scores are evidence signals, not proof that a training example was present.",
        "Prompting, tokenization, model capability, and control-set quality can affect the result.",
        "The index weights and evidence bands are heuristics, not calibrated probabilities or validated decision thresholds.",
    ]
    generation_metadata = getattr(model, "generation_metadata", None)
    if generation_metadata:
        caveats.extend(str(note) for note in generation_metadata.get("caveats", []))
    skipped_methods: list[dict[str, str]] = []

    model_fingerprint = getattr(model, "fingerprint", None) or config.model_id
    tokenizer_fingerprint = getattr(model, "tokenizer_fingerprint", None) or ""
    use_cache = bool(
        config.cache_enabled and getattr(model, "fingerprint", None)
    )
    cache: Any | None = None
    cache_error: str | None = None
    cache_stats: dict[str, int] | None = None
    if use_cache:
        opened_cache, cache_error = _open_cache(config)
        cache = None if cache_error else opened_cache
        if cache_error:
            caveats.append(
                "The score cache could not be opened "
                f"({cache_error}); continuing without caching."
            )

    guided_result: dict[str, Any] | None = None
    if capabilities["supports_generation"]:
        try:
            guided_result = run_guided_overlap(
                model,
                benchmark.examples,
                dataset_name=benchmark.dataset_name,
                split_name=benchmark.split,
                split_ratio=config.split_ratio,
                repro_threshold=config.repro_threshold,
                seed=config.seed,
                chat_template=config.chat_template,
                generation_batch_size=config.generation_batch_size,
                cache=cache,
                cache_model_fingerprint=model_fingerprint if cache else "",
                cache_tokenizer_fingerprint=tokenizer_fingerprint,
                stability=config.guided_stability,
            )
        except (ValueError, RuntimeError, TypeError) as error:
            skipped_methods.append({"method": "guided", "reason": str(error)})
            guided_result = None
        if guided_result is not None:
            if guided_result["guided_rouge"] >= config.repro_threshold and abs(
                guided_result["guided_general_gap"]
            ) < 0.05:
                caveats.append(
                    "Guided overlap was high but the guided-over-general gap was small, "
                    "so the overlap is weak contamination evidence."
                )
            mismatched_gap = guided_result.get("guided_mismatched_gap")
            if (
                guided_result["guided_rouge"] >= config.repro_threshold
                and mismatched_gap is not None
                and abs(float(mismatched_gap)) < 0.05
            ):
                caveats.append(
                    "The mismatched-prefix control reproduced about as well as the "
                    "guided prompt, so guided overlap looks format-driven rather than "
                    "memory of the target example."
                )
    else:
        skipped_methods.append(
            {"method": "guided", "reason": "The model does not support generation."}
        )

    benchmark_minkpp: list[float] = []
    control_minkpp: list[float] = []
    benchmark_loss: list[float] = []
    control_loss: list[float] = []
    benchmark_mink: list[float] = []
    control_mink: list[float] = []
    benchmark_zlib: list[float] = []
    control_zlib: list[float] = []
    benchmark_entropy: list[float] = []
    control_entropy: list[float] = []
    benchmark_rank: list[float] = []
    control_rank: list[float] = []
    benchmark_top1: list[float] = []
    control_top1: list[float] = []
    benchmark_lengths: list[float] = []
    control_lengths: list[float] = []
    mink_by_id: dict[str, MinKScores] = {}
    control_scores: dict[str, MinKScores] = {}
    mink_failures: list[str] = []
    truncated_count = 0
    if capabilities["supports_full_distribution"]:
        benchmark_scores, benchmark_failures = _score_examples(
            model,
            benchmark.examples,
            config=config,
            cache=cache,
            cache_model_fingerprint=model_fingerprint,
            cache_tokenizer_fingerprint=tokenizer_fingerprint,
        )
        mink_by_id = benchmark_scores
        mink_failures.extend(benchmark_failures)
        for example in benchmark.examples:
            score = benchmark_scores.get(example["id"])
            if score is None:
                continue
            benchmark_minkpp.append(score.minkpp)
            benchmark_mink.append(score.mink)
            benchmark_loss.append(score.loss)
            benchmark_zlib.append(score.zlib)
            benchmark_entropy.append(score.entropy)
            benchmark_rank.append(score.mean_rank)
            benchmark_top1.append(score.top1_prob)
            benchmark_lengths.append(float(score.token_count))
            truncated_count += int(bool(score.truncated))
        if control is not None:
            control_scores, control_failures = _score_examples(
                model,
                control.examples,
                config=config,
                cache=cache,
                cache_model_fingerprint=model_fingerprint,
                cache_tokenizer_fingerprint=tokenizer_fingerprint,
                id_prefix="control ",
            )
            mink_failures.extend(control_failures)
            for example in control.examples:
                score = control_scores.get(example["id"])
                if score is None:
                    continue
                control_minkpp.append(score.minkpp)
                control_mink.append(score.mink)
                control_loss.append(score.loss)
                control_zlib.append(score.zlib)
                control_entropy.append(score.entropy)
                control_rank.append(score.mean_rank)
                control_top1.append(score.top1_prob)
                control_lengths.append(float(score.token_count))
                truncated_count += int(bool(score.truncated))
        if not benchmark_minkpp:
            skipped_methods.append(
                {
                    "method": "minkpp",
                    "reason": "No benchmark example produced a valid Min-K%++ score.",
                }
            )
    else:
        skipped_methods.append(
            {
                "method": "minkpp",
                "reason": "The model does not expose full next-token distributions.",
            }
        )
    if truncated_count:
        caveats.append(
            f"{truncated_count} scored text(s) reached max_tokens={config.max_tokens}; "
            "scores cover only a prefix of those texts."
        )

    # Confounder gate (early): strong numeric-identifier bias makes the
    # calibrated AUC uninterpretable, so the likelihood index is withheld
    # entirely rather than caveated-and-kept.
    fingerprint = None
    if control is not None and benchmark_minkpp and control_minkpp:
        fingerprint = domain_fingerprint(
            [
                example["text"]
                for example in benchmark.examples
                if example["id"] in mink_by_id
            ],
            [
                example["text"]
                for example in control.examples
                if example["id"] in control_scores
            ],
        )
    index_suppressed_reason: str | None = None

    uncalibrated_reason: str | None = None
    if benchmark_minkpp and control is not None and control_minkpp:
        if (
            len(benchmark_minkpp) < config.min_calibration_examples
            or len(control_minkpp) < config.min_calibration_examples
        ):
            uncalibrated_reason = (
                f"Fewer than {config.min_calibration_examples} scored examples per "
                "side, so Min-K%++ was not calibrated."
            )
            caveats.append(uncalibrated_reason)
            index_minkpp, minkpp_auc = None, None
        else:
            index_minkpp, minkpp_auc = calibrate_minkpp(
                benchmark_minkpp, control_minkpp
            )
    elif benchmark_minkpp:
        index_minkpp, minkpp_auc = calibrate_minkpp(benchmark_minkpp, None)
    else:
        index_minkpp, minkpp_auc = None, None
    loss_index, loss_auc = calibrate_minkpp(
        benchmark_loss, control_loss
    ) if (
        benchmark_loss
        and control is not None
        and control_loss
    ) else (None, None)
    if benchmark_minkpp and index_minkpp is None and uncalibrated_reason is None:
        uncalibrated_reason = (
            "A matched control set is required for a calibrated Min-K%++ index."
        )
        caveats.append(
            "Min-K%++ is shown as a raw distribution because no usable matched control "
            "set was supplied."
        )
    elif benchmark_minkpp and index_minkpp is None and uncalibrated_reason:
        caveats.append(
            "Min-K%++ is shown as a raw distribution because calibration was "
            "withheld (see the calibration note in minkpp_summary)."
        )

    order_result = None
    if capabilities["supports_logprobs"]:
        try:
            order_result = canonical_order_test(
                benchmark.examples,
                _make_order_scorer(
                    model,
                    config=config,
                    cache=cache,
                    cache_model_fingerprint=model_fingerprint,
                    cache_tokenizer_fingerprint=tokenizer_fingerprint,
                ),
                shard_count=config.shard_count,
                permutation_count=config.permutation_count,
                seed=config.seed,
                token_counter=getattr(model, "count_tokens", None),
                max_tokens=config.max_tokens,
            )
        except ValueError as error:
            skipped_methods.append({"method": "canonical_order", "reason": str(error)})
    else:
        skipped_methods.append(
            {
                "method": "canonical_order",
                "reason": "The model does not expose sequence likelihood.",
            }
        )

    index_guided = (
        float(guided_result["index_guided"]) if guided_result is not None else None
    )
    if guided_result is not None:
        near_absent = (
            guided_result["guided_near_rate"] == 0.0
            and guided_result["general_near_rate"] == 0.0
        )
        if near_absent and abs(float(guided_result["guided_general_gap"])) < 0.05:
            index_guided = None
            caveats.append(
                "Guided overlap produced no near matches and a negligible gap, "
                "so the model is not reproducing the reference suffix. The gap "
                "was left out of the combined index."
            )
        mismatch_gap = guided_result.get("guided_mismatched_gap")
        if (
            float(guided_result.get("guided_rouge") or 0.0) >= config.repro_threshold
            and mismatch_gap is not None
            and abs(float(mismatch_gap)) < 0.05
        ):
            index_guided = None
            caveats.append(
                "Guided evidence was withheld because the wrong-prefix control "
                "reproduces the target about as well. A dataset-name prompt effect "
                "alone does not identify example memorization."
            )
    index_order = order_result.index_order if order_result is not None else None
    if order_result is not None:
        caveats.extend(str(warning) for warning in order_result.warnings)
        if order_result.aggregate_reason:
            caveats.append(order_result.aggregate_reason)
    control_null = control_null_check(control_minkpp) if control is not None else None
    domain_shift = domain_shift_check(benchmark_lengths, control_lengths)
    effective_text = canonicalize if config.canonicalize_scores else lambda text: text
    comparison_validity = likelihood_comparison_validity(
        [effective_text(item["text"]) for item in benchmark.examples],
        [effective_text(item["text"]) for item in control.examples] if control is not None else None,
        benchmark_scored=len(benchmark_minkpp),
        control_scored=len(control_minkpp),
        min_examples=config.min_calibration_examples,
        domain_shift=domain_shift,
        fingerprint=fingerprint,
        control_null=control_null,
        benchmark_scoring_ids=_scored_input_hashes(model, [effective_text(item["text"]) for item in benchmark.examples if item["id"] in mink_by_id], config),
        control_scoring_ids=_scored_input_hashes(model, [effective_text(item["text"]) for item in control.examples if item["id"] in control_scores], config) if control is not None else None,
    )
    if not comparison_validity["eligible"]:
        index_suppressed_reason = "Withheld: " + " ".join(
            item["message"] for item in comparison_validity["reasons"]
        )
        index_minkpp = None
        loss_index = None
        caveats.append(index_suppressed_reason)
    else:
        caveats.append(comparison_validity["interpretation"])
    if comparison_validity["scored_input_identity_check"] == "unavailable" and benchmark_minkpp:
        caveats.append("Effective scoring-token duplication was not checked for this adapter; full-text duplicate checks cannot rule out identical truncated inputs.")

    combined = combine_indices(index_guided, index_minkpp, index_order)

    per_example: list[dict[str, Any]] = []
    guided_by_id = {
        item["id"]: item
        for item in guided_result["per_example"]
    } if guided_result else {}
    for example in benchmark.examples:
        guided_item = guided_by_id.get(example["id"], {})
        mink_score = mink_by_id.get(example["id"])
        per_example.append(
            {
                "id": example["id"],
                "guided_rouge": guided_item.get("guided_rouge"),
                "general_rouge": guided_item.get("general_rouge"),
                "guided_exact": guided_item.get("guided_exact"),
                "general_exact": guided_item.get("general_exact"),
                "guided_near": guided_item.get("guided_near"),
                "general_near": guided_item.get("general_near"),
                "mismatched_rouge": guided_item.get("mismatched_rouge"),
                "minkpp_score": mink_score.minkpp if mink_score else None,
                "mink_score": mink_score.mink if mink_score else None,
                "loss": mink_score.loss if mink_score else None,
                "zlib": mink_score.zlib if mink_score else None,
                "entropy": mink_score.entropy if mink_score else None,
                "mean_rank": mink_score.mean_rank if mink_score else None,
                "top1_prob": mink_score.top1_prob if mink_score else None,
                "truncated": bool(mink_score.truncated) if mink_score else None,
            }
        )

    guided_gaps = None
    if guided_result:
        guided_gaps = [
            float(item["guided_rouge"]) - float(item["general_rouge"])
            for item in guided_result["per_example"]
        ]
    order_advantages = (
        [float(item["advantage"]) for item in order_result.shards]
        if order_result is not None and order_result.index_order is not None
        else None
    )
    confidence_interval = None
    if len(combined.active_weights) <= 1:
        confidence_interval = bootstrap_combined_ci(
            guided_gaps=guided_gaps if index_guided is not None else None,
            minkpp_benchmark=benchmark_minkpp if index_minkpp is not None else None,
            minkpp_control=control_minkpp if index_minkpp is not None else None,
            order_shard_advantages=order_advantages,
            samples=config.bootstrap_samples,
            seed=config.seed,
        )
    else:
        caveats.append(
            "The combined interval is withheld: methods share examples and need an "
            "aligned joint resampling design. Independent method bootstraps would "
            "discard cross-method dependence."
        )
    if confidence_interval is None:
        caveats.append(
            "A supported interval could not be estimated for this method combination and sample."
        )
    elif confidence_interval[0] == confidence_interval[1]:
        caveats.append("The bootstrap interval is degenerate on this sample; it does not establish perfect population discrimination.")
    if mink_failures:
        caveats.append(
            f"Min-K%++ skipped {len(mink_failures)} text entries that could not be scored."
        )

    diagnostics = minkpp_diagnostics(benchmark_minkpp, control_minkpp)
    raw_method_p_values: dict[str, float | None] = {"guided": None, "minkpp": None, "order": None}
    if guided_gaps:
        raw_method_p_values["guided"] = float(
            paired_permutation_test(
                guided_gaps, permutations=2000, seed=config.seed, alternative="greater"
            )["p_value"]
        )
    if diagnostics is not None:
        raw_method_p_values["minkpp"] = float(diagnostics["mann_whitney_p_value"])
    if order_result is not None:
        raw_method_p_values["order"] = float(order_result.p_value)
    method_p_values = {
        "guided": raw_method_p_values["guided"] if index_guided is not None else None,
        "minkpp": raw_method_p_values["minkpp"] if index_minkpp is not None else None,
        "order": raw_method_p_values["order"] if index_order is not None else None,
    }
    combined_p_details = combine_diagnostic_pvalues(method_p_values)
    combined_p_details["component_null_assumptions"] = {
        "guided": "Sign flips require within-pair exchangeability or symmetric gap distributions; fixed prompt treatments and a zero mean alone do not establish this.",
        "minkpp": "Group labels must be exchangeable under equal score distributions; a difference need not be caused by training exposure.",
        "order": "Benchmark order must be exchangeable under the null; the sharded t-test is asymptotic and depends on shard assumptions.",
    }
    combined_p_value = combined_p_details["p_value"]
    active_p_names = sorted(
        name for name, value in method_p_values.items() if value is not None
    )
    combined_p_value_basis = (
        "none" if not active_p_names else "_plus_".join(active_p_names)
    )
    if domain_shift is not None:
        median_ratio = float(domain_shift["benchmark_median_tokens"]) / max(
            float(domain_shift["control_median_tokens"]), 1.0
        )
        if (
            float(domain_shift["mann_whitney_p_value"]) < 0.05
            and (median_ratio > 1.5 or median_ratio < 1.0 / 1.5)
        ):
            caveats.append(
                "Benchmark and control differ sharply in token length "
                f"(medians {domain_shift['benchmark_median_tokens']:.0f} vs "
                f"{domain_shift['control_median_tokens']:.0f} tokens), so their "
                "AUC may partly reflect domain mismatch rather than memorization."
            )
    if control_null is not None:
        null_ci = control_null["auc_ci"]
        null_p = float(control_null["mann_whitney_p_value"])
        if null_ci is not None and (float(null_ci[0]) > 0.5 or float(null_ci[1]) < 0.5) and null_p < 0.05:
            caveats.append(
                "The control split-half AUC deviates from 0.5, so the control "
                "looks domain-heterogeneous. Benchmark-vs-control AUC may then "
                "partly reflect domain mismatch, not memorization."
            )
    ngram_auc = None
    if control is not None and benchmark_minkpp and control_minkpp:
        from .methods.ngram import score_corpus

        benchmark_texts = [
            example["text"]
            for example in benchmark.examples
            if example["id"] in mink_by_id
        ]
        control_texts_all = [example["text"] for example in control.examples]
        benchmark_ngrams = score_corpus(benchmark_texts, control_texts_all)
        control_texts_scored = [
            example["text"]
            for example in control.examples
            if example["id"] in control_scores
        ]
        control_ngrams = score_corpus(
            control_texts_scored, control_texts_scored, leave_one_out=True
        )
        ngram_auc = calibrate_minkpp(benchmark_ngrams, control_ngrams)[1]

    perturb_auc = None
    if control is not None and benchmark_minkpp and control_minkpp:
        bench_texts = [
            example["text"]
            for example in benchmark.examples
            if example["id"] in mink_by_id
        ]
        ctrl_texts = [
            example["text"]
            for example in control.examples
            if example["id"] in control_scores
        ]
        try:
            bench_delta = perturb_delta_mean(
                model, bench_texts, max_tokens=config.max_tokens
            )
            ctrl_delta = perturb_delta_mean(
                model, ctrl_texts, max_tokens=config.max_tokens
            )
            perturb_auc = calibrate_minkpp(bench_delta, ctrl_delta)[1]
        except (ValueError, TypeError, RuntimeError):
            perturb_auc = None
    score_baselines = {
        "ngram_auc": ngram_auc,
        "perturb_auc": perturb_auc,
        "zlib_auc": (
            calibrate_minkpp(
                benchmark_zlib, control_zlib if control is not None and control_zlib else None
            )[1]
            if benchmark_zlib
            else None
        ),
        "entropy_auc": (
            calibrate_minkpp(
                benchmark_entropy, control_entropy if control is not None and control_entropy else None
            )[1]
            if benchmark_entropy
            else None
        ),
        "mean_rank_auc": (
            calibrate_minkpp(
                benchmark_rank, control_rank if control is not None and control_rank else None
            )[1]
            if benchmark_rank
            else None
        ),
        "top1_prob_auc": (
            calibrate_minkpp(
                benchmark_top1, control_top1 if control is not None and control_top1 else None
            )[1]
            if benchmark_top1
            else None
        ),
        "note": (
            "Additional token-statistic scores (including the perturbation-"
            "sensitivity score, which measures brittleness under surface "
            "mutation) calibrated like Min-K%++. Reported for comparison "
            "only; none enter the combined index."
        ),
    }
    membership_screen = None
    if index_minkpp is not None and comparison_validity["eligible"]:
        aligned_ids = [
            example["id"]
            for example in benchmark.examples
            if example["id"] in mink_by_id
        ]
        membership_screen = conformal_membership_screen(
            benchmark_minkpp,
            control_minkpp,
            aligned_ids,
            fdr_level=config.conformal_fdr,
        )
    membership_screen_status = {
        "status": "conditional" if membership_screen is not None else "withheld",
        "reason": (
            "Conditional score-outlier screen; exchangeability, nonmembership and dependence assumptions are unverified."
            if membership_screen is not None else
            index_suppressed_reason or uncalibrated_reason or "Usable calibrated likelihood scores are unavailable."
        ),
    }
    planning = {
        "minimum_detectable_effect": minimum_detectable_effect(
            len(benchmark_minkpp), len(control_minkpp)
        ),
        "benchmark_scored": len(benchmark_minkpp),
        "control_scored": len(control_minkpp),
        "note": (
            "Standardized mean difference detectable with a two-sample z-test "
            "at 80% power. A planning aid, not a detection guarantee."
        ),
    }

    if cache is not None:
        try:
            cache_stats = cache.stats()
        except Exception:
            cache_stats = None

    refutation_panel = _build_refutation_panel(
        domain_shift=domain_shift,
        fingerprint=fingerprint,
        control_null=control_null,
        guided_result=guided_result,
        repro_threshold=config.repro_threshold,
        minkpp_auc=minkpp_auc,
        loss_auc=loss_auc,
        combined_p_value=combined_p_value,
    )

    guided_defaults = {
        "guided_rouge": None,
        "general_rouge": None,
        "guided_general_gap": None,
        "guided_gap_ci": None,
        "guided_exact_rate": None,
        "general_exact_rate": None,
        "exact_gap": None,
        "guided_near_rate": None,
        "general_near_rate": None,
        "near_gap": None,
        "mismatched_rouge": None,
        "guided_mismatched_gap": None,
        "mismatched_near_rate": None,
        "mismatched_exact_rate": None,
        "reproduction_rate": None,
    }
    if guided_result:
        guided_defaults.update(
            {key: guided_result[key] for key in guided_defaults}
        )
    manifest = {
        "schema_version": "contamcheckr-report-v2",
        "comparison_policy_version": comparison_validity["policy_version"],
        "contamcheckr_version": _versions()["contamcheckr"],
        "source_snapshot": source_snapshot(),
        "model_fingerprint": model_fingerprint,
        "model_provenance": getattr(model, "provenance", None),
        "tokenizer_fingerprint": tokenizer_fingerprint or None,
        "benchmark_hash": hash_examples(benchmark.examples),
        "benchmark_ordered_hash": hash_ordered_examples(benchmark.examples),
        "control_hash": (
            hash_examples(control.examples) if control is not None else None
        ),
        "control_ordered_hash": hash_ordered_examples(control.examples) if control is not None else None,
        "generation_metadata": generation_metadata,
        "config_fingerprint": config_fingerprint(config.__dict__),
        "scoring_engine": scoring_engine,
        "cache": {
            "enabled": cache is not None,
            "path": str(getattr(cache, "path", None)) if cache is not None else None,
            "stats": cache_stats,
        },
        "settings": {
            "k": config.k,
            "max_tokens": config.max_tokens,
            "shard_count": config.shard_count,
            "permutation_count": config.permutation_count,
            "split_ratio": config.split_ratio,
            "repro_threshold": config.repro_threshold,
            "chat_template": config.chat_template,
            "guided_stability": config.guided_stability,
            "scoring_mode": config.scoring_mode,
            "canonicalize_scores": config.canonicalize_scores,
            "window_tokens": config.window_tokens,
            "window_stride_tokens": config.window_stride_tokens,
            "generation_batch_size": config.generation_batch_size,
            "scoring_batch_size": config.scoring_batch_size,
            "bootstrap_samples": config.bootstrap_samples,
            "min_calibration_examples": config.min_calibration_examples,
            "conformal_fdr": config.conformal_fdr,
        },
    }
    return {
        "model_id": config.model_id,
        "benchmark_source": benchmark.source,
        "control_source": control.source if control is not None else None,
        "split": benchmark.split,
        "dataset_name": benchmark.dataset_name,
        "example_count": len(benchmark.examples),
        "control_count": len(control.examples) if control is not None else 0,
        "seed": config.seed,
        "library_versions": _versions(),
        "capabilities": capabilities,
        "index_guided": index_guided,
        **guided_defaults,
        "index_minkpp": index_minkpp,
        "minkpp_auc": minkpp_auc,
        "minkpp_summary": {
            "benchmark_minkpp": _distribution(benchmark_minkpp),
            "control_minkpp": _distribution(control_minkpp),
            "benchmark_mink": _distribution(benchmark_mink),
            "control_mink": _distribution(control_mink),
            "uncalibrated_reason": uncalibrated_reason,
            "auc_ci": diagnostics["auc_ci"] if diagnostics else None,
            "auc_standard_error": (
                diagnostics["auc_standard_error"] if diagnostics else None
            ),
            "mann_whitney_p_value": (
                diagnostics["mann_whitney_p_value"] if diagnostics else None
            ),
            "mann_whitney_method": diagnostics["mann_whitney_method"] if diagnostics else None,
            "cohen_d": diagnostics["cohen_d"] if diagnostics else None,
            "auc_smoothed": auc_posterior_mean(benchmark_minkpp, control_minkpp)
            if benchmark_minkpp and control_minkpp
            else None,
            "auc_smoothed_interpretation": "Legacy add-half heuristic; pairwise wins are dependent, so this is not a Bayesian posterior.",
        },
        "index_order": index_order,
        "order_p_value": order_result.p_value if order_result else None,
        "method_p_values": method_p_values,
        "raw_method_p_values": raw_method_p_values,
        "combined_p_value": combined_p_value,
        "combined_p_value_details": combined_p_details,
        "combined_p_value_basis": combined_p_value_basis,
        "likelihood_baseline": {
            "loss_auc": loss_auc,
            "loss_index": loss_index,
            "eligible_for_inference": comparison_validity["eligible"],
            "note": (
                "Mean token log-probability scored like Min-K%++ against the "
                "same control. Reported for comparison and excluded from the "
                "combined index."
            ),
        },
        "order_summary": (
            {
                "canonical_advantage": order_result.canonical_advantage,
                "shards": order_result.shards,
                "permutation_count": config.permutation_count,
                "effect_size": order_result.effect_size,
                "effect_size_ci": (
                    list(order_result.effect_size_ci)
                    if order_result.effect_size_ci is not None
                    else None
                ),
                "shard_win_rate": order_result.shard_win_rate,
                "p_value_rank": order_result.p_value_rank,
            }
            if order_result
            else None
        ),
        "membership_screen": membership_screen,
        "membership_screen_status": membership_screen_status,
        "comparison_validity": comparison_validity,
        "control_null": control_null,
        "domain_shift": domain_shift,
        "domain_fingerprint": fingerprint,
        "refutation_panel": refutation_panel,
        "score_baselines": score_baselines,
        "guided_split_stability": guided_result.get("split_stability") if guided_result else None,
        "planning": planning,
        "manifest": manifest,
        "index_suppressed_reason": index_suppressed_reason,
        "contamination_index": combined.contamination_index,
        "index_interpretation": "Heuristic comparative index; not a contamination probability or validated decision rule.",
        "confidence_interval": confidence_interval,
        "confidence_interval_note": "Bootstrap interval conditional on selected methods; not uncertainty about a contamination probability.",
        "index_basis": combined.index_basis,
        "band": combined.band,
        "caveats": caveats,
        "skipped_methods": skipped_methods,
        "guided_skipped_examples": (
            guided_result.get("skipped_examples", []) if guided_result else []
        ),
        "minkpp_skipped": list(mink_failures),
        "per_example": per_example,
        "control_per_example": (
            [
                {
                    "id": example["id"],
                    "minkpp": control_scores[example["id"]].minkpp,
                    "mink": control_scores[example["id"]].mink,
                    "loss": control_scores[example["id"]].loss,
                    "zlib": control_scores[example["id"]].zlib,
                    "entropy": control_scores[example["id"]].entropy,
                    "mean_rank": control_scores[example["id"]].mean_rank,
                    "top1_prob": control_scores[example["id"]].top1_prob,
                    "truncated": bool(control_scores[example["id"]].truncated),
                }
                for example in control.examples
                if example["id"] in control_scores
            ]
            if control is not None
            else []
        ),
    }


def write_json(report: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(report: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "guided_rouge",
        "general_rouge",
        "guided_exact",
        "general_exact",
        "guided_near",
        "general_near",
        "mismatched_rouge",
        "minkpp_score",
        "mink_score",
        "loss",
        "truncated",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in fields}
            for row in report["per_example"]
        )


def format_summary(report: dict[str, Any]) -> str:
    interval = report["confidence_interval"]
    interval_text = (
        f"{interval[0]:.3f} to {interval[1]:.3f}"
        if interval is not None
        else "not estimated"
    )
    lines = [
        f"Model: {report['model_id']}",
        f"Benchmark: {report['benchmark_source']} ({report['example_count']} examples)",
        "Contamination index: "
        + (
            f"{report['contamination_index']:.3f}"
            if report["contamination_index"] is not None
            else "not estimated"
        ),
        f"Band: {report['band']}",
        f"Basis: {report['index_basis']} (heuristic index)",
        f"Confidence interval: {interval_text}",
    ]
    validity = report.get("comparison_validity")
    if validity:
        lines.insert(2, f"Likelihood comparison: {validity['status']} — {validity['interpretation']}")
        lines.extend(f"Withheld reason: {item['message']}" for item in validity["reasons"])
    if report["guided_general_gap"] is not None:
        lines.append(
            f"Guided-over-general ROUGE-L gap: {report['guided_general_gap']:.3f}"
        )
    if report["minkpp_auc"] is not None:
        lines.append(f"Min-K%++ AUC: {report['minkpp_auc']:.3f}")
        auc_ci = (report.get("minkpp_summary") or {}).get("auc_ci")
        if auc_ci is not None:
            lines.append(f"Min-K%++ AUC 95% interval: {auc_ci[0]:.3f} to {auc_ci[1]:.3f}")
    screen = report.get("membership_screen")
    if screen:
        lines.append(
            f"Membership screen: {screen['n_flagged']} of {screen['n_scored']} "
            f"score outliers at requested FDR {screen['fdr_level']} (conditional)"
        )
    elif report.get("membership_screen_status"):
        lines.append(f"Membership screen: withheld — {report['membership_screen_status']['reason']}")
    if report.get("combined_p_value") is not None:
        lines.append(
            f"Combined diagnostic p-value (fixed family of 3, observed {report['combined_p_value_basis']}): "
            f"{report['combined_p_value']:.3g}"
        )
    if report["order_p_value"] is not None:
        lines.append(f"Canonical-order p-value: {report['order_p_value']:.3f}")
    if report["skipped_methods"]:
        lines.append(
            "Skipped: "
            + "; ".join(
                f"{item['method']} ({item['reason']})"
                for item in report["skipped_methods"]
            )
        )
    lines.append("Caveats:")
    lines.extend(f"- {caveat}" for caveat in report["caveats"])
    return "\n".join(lines)
