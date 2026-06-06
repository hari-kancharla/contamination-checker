from dataclasses import dataclass

import numpy as np

from .metrics import auc, clamp


@dataclass(frozen=True)
class CombinedIndex:
    contamination_index: float
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
        raise ValueError("No method produced evidence that can be aggregated.")
    total_weight = sum(active.values())
    normalized = {
        name: weight / total_weight for name, weight in active.items()
    }
    score = sum(
        normalized[name] * float(values[name])
        for name in normalized
    )
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
    order_null_advantages: list[float] | None,
    samples: int = 2000,
    seed: int = 0,
) -> tuple[float, float] | None:
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
        if order_shard_advantages is not None and order_null_advantages is not None:
            observed = float(
                np.mean(
                    rng.choice(
                        order_shard_advantages,
                        len(order_shard_advantages),
                        replace=True,
                    )
                )
            )
            p_value = (
                1 + sum(value >= observed for value in order_null_advantages)
            ) / (len(order_null_advantages) + 1)
            order_index = clamp(1.0 - p_value)
        estimates.append(
            combine_indices(guided_index, minkpp_index, order_index).contamination_index
        )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)

