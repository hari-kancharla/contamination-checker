"""Paper evaluation harness: per-example detectors -> publication tables.

Given benchmark/control per-example scores for every detector, this
module reports discrimination (AUC + DeLong interval), operating-point
power (TPR at fixed FPR), null calibration (control split-half), and
conformal-screen behaviour.  All statistics come from
:mod:`contamcheckr.stats` / :mod:`contamcheckr.calibration` so harness
numbers and report numbers share one implementation.
"""

from __future__ import annotations

from typing import Sequence

from contamcheckr.calibration import (
    auc_posterior_mean,
    calibrate_minkpp,
    conformal_membership_screen,
    control_null_check,
    minkpp_diagnostics,
)
from contamcheckr.stats import tpr_at_fpr


def evaluate_scores(
    benchmark_scores: Sequence[float],
    control_scores: Sequence[float],
    example_ids: Sequence[str] | None = None,
    *,
    fdr_level: float = 0.10,
    fpr_rates: Sequence[float] = (0.01, 0.05, 0.10),
) -> dict[str, object]:
    """Full evaluation battery for one detector on benchmark vs control."""
    benchmark = [float(value) for value in benchmark_scores]
    control = [float(value) for value in control_scores]
    index, auc_value = calibrate_minkpp(benchmark, control)
    diagnostics = minkpp_diagnostics(benchmark, control)
    evaluation: dict[str, object] = {
        "index": index,
        "auc": auc_value,
        "auc_smoothed": auc_posterior_mean(benchmark, control),
        "auc_ci": diagnostics["auc_ci"] if diagnostics else None,
        "mann_whitney_p_value": (
            diagnostics["mann_whitney_p_value"] if diagnostics else None
        ),
        "cohen_d": diagnostics["cohen_d"] if diagnostics else None,
        "tpr_at_fpr": tpr_at_fpr(benchmark, control, fpr_rates)
        if benchmark and control
        else None,
    }
    ids = list(example_ids) if example_ids is not None else [
        f"ex-{index}" for index in range(len(benchmark))
    ]
    screen = conformal_membership_screen(
        benchmark, control, ids, fdr_level=fdr_level
    )
    evaluation["screen"] = screen
    evaluation["control_null"] = control_null_check(control)
    return evaluation


def tpr_table(
    runs: list[dict[str, object]],
    detector: str,
    rates: Sequence[float] = (0.05, 0.10),
) -> list[dict[str, object]]:
    """Mean TPR at FPR across seeds for generated table rows."""
    rows: list[dict[str, object]] = []
    for rate in rates:
        key = f"tpr_at_fpr_{float(rate):g}"
        values = [
            float(run[detector]["tpr_at_fpr"][key])  # type: ignore[index]
            for run in runs
            if run.get(detector, {}).get("tpr_at_fpr")
        ]
        if values:
            rows.append({"fpr": rate, "mean": sum(values) / len(values), "values": values})
    return rows
