import csv
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import bootstrap_combined_ci, calibrate_minkpp, combine_indices
from .config import RunConfig
from .methods.canonical_order import canonical_order_test
from .methods.guided import run_guided_overlap
from .methods.minkpp import score_text


def _versions() -> dict[str, str]:
    names = ("contamcheck", "torch", "transformers", "datasets", "numpy")
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


def build_report(
    model: Any,
    benchmark: Any,
    control: Any | None,
    config: RunConfig,
) -> dict[str, Any]:
    capabilities = model.capabilities.as_dict()
    caveats = [
        "Scores are evidence signals, not proof that a training example was present.",
        "Prompting, tokenization, model capability, and control-set quality can affect the result.",
    ]
    skipped_methods: list[dict[str, str]] = []

    guided_result: dict[str, Any] | None = None
    if capabilities["supports_generation"]:
        guided_result = run_guided_overlap(
            model,
            benchmark.examples,
            dataset_name=benchmark.dataset_name,
            split_name=benchmark.split,
            split_ratio=config.split_ratio,
            repro_threshold=config.repro_threshold,
            seed=config.seed,
        )
        if guided_result["guided_rouge"] >= config.repro_threshold and abs(
            guided_result["guided_general_gap"]
        ) < 0.05:
            caveats.append(
                "Guided overlap was high but the guided-over-general gap was small, "
                "so the overlap is weak contamination evidence."
            )
    else:
        skipped_methods.append(
            {"method": "guided", "reason": "The model does not support generation."}
        )

    benchmark_minkpp: list[float] = []
    control_minkpp: list[float] = []
    benchmark_mink: list[float] = []
    control_mink: list[float] = []
    mink_by_id: dict[str, float] = {}
    mink_failures: list[str] = []
    if capabilities["supports_full_distribution"]:
        for example in benchmark.examples:
            try:
                score = score_text(
                    model,
                    example["text"],
                    k=config.k,
                    max_tokens=config.max_tokens,
                )
            except ValueError as error:
                mink_failures.append(f"{example['id']}: {error}")
                continue
            benchmark_minkpp.append(score.minkpp)
            benchmark_mink.append(score.mink)
            mink_by_id[example["id"]] = score.minkpp
        if control is not None:
            for example in control.examples:
                try:
                    score = score_text(
                        model,
                        example["text"],
                        k=config.k,
                        max_tokens=config.max_tokens,
                    )
                except ValueError as error:
                    mink_failures.append(f"control {example['id']}: {error}")
                    continue
                control_minkpp.append(score.minkpp)
                control_mink.append(score.mink)
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
    index_minkpp, minkpp_auc = calibrate_minkpp(
        benchmark_minkpp,
        control_minkpp if control is not None and control_minkpp else None,
    )
    if benchmark_minkpp and index_minkpp is None:
        caveats.append(
            "Min-K%++ is shown as a raw distribution because no usable matched control "
            "set was supplied."
        )

    order_result = None
    if capabilities["supports_logprobs"]:
        try:
            order_result = canonical_order_test(
                benchmark.examples,
                model.sequence_loglikelihood,
                shard_count=config.shard_count,
                permutation_count=config.permutation_count,
                seed=config.seed,
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
    index_order = order_result.index_order if order_result is not None else None
    combined = combine_indices(index_guided, index_minkpp, index_order)

    per_example: list[dict[str, Any]] = []
    guided_by_id = {
        item["id"]: item
        for item in guided_result["per_example"]
    } if guided_result else {}
    for example in benchmark.examples:
        guided_item = guided_by_id.get(example["id"], {})
        per_example.append(
            {
                "id": example["id"],
                "guided_rouge": guided_item.get("guided_rouge"),
                "general_rouge": guided_item.get("general_rouge"),
                "guided_exact": guided_item.get("guided_exact"),
                "general_exact": guided_item.get("general_exact"),
                "guided_near": guided_item.get("guided_near"),
                "general_near": guided_item.get("general_near"),
                "minkpp_score": mink_by_id.get(example["id"]),
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
        if order_result
        else None
    )
    confidence_interval = bootstrap_combined_ci(
        guided_gaps=guided_gaps,
        minkpp_benchmark=benchmark_minkpp if index_minkpp is not None else None,
        minkpp_control=control_minkpp if index_minkpp is not None else None,
        order_shard_advantages=order_advantages,
        order_null_advantages=(
            order_result.permutation_advantages if order_result else None
        ),
        seed=config.seed,
    )
    if confidence_interval is None:
        caveats.append(
            "A confidence interval could not be estimated from the available sample sizes."
        )
    if mink_failures:
        caveats.append(
            f"Min-K%++ skipped {len(mink_failures)} text entries that could not be scored."
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
        "reproduction_rate": None,
    }
    if guided_result:
        guided_defaults.update(
            {key: guided_result[key] for key in guided_defaults}
        )
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
            "uncalibrated_reason": (
                None
                if index_minkpp is not None
                else "A matched control set is required for a calibrated Min-K%++ index."
            ),
        },
        "index_order": index_order,
        "order_p_value": order_result.p_value if order_result else None,
        "order_summary": (
            {
                "canonical_advantage": order_result.canonical_advantage,
                "shards": order_result.shards,
                "permutation_count": config.permutation_count,
            }
            if order_result
            else None
        ),
        "contamination_index": combined.contamination_index,
        "confidence_interval": confidence_interval,
        "index_basis": combined.index_basis,
        "band": combined.band,
        "caveats": caveats,
        "skipped_methods": skipped_methods,
        "per_example": per_example,
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
        "minkpp_score",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["per_example"])


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
        f"Contamination index: {report['contamination_index']:.3f}",
        f"Band: {report['band']}",
        f"Basis: {report['index_basis']}",
        f"Confidence interval: {interval_text}",
    ]
    if report["guided_general_gap"] is not None:
        lines.append(
            f"Guided-over-general ROUGE-L gap: {report['guided_general_gap']:.3f}"
        )
    if report["minkpp_auc"] is not None:
        lines.append(f"Min-K%++ AUC: {report['minkpp_auc']:.3f}")
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
