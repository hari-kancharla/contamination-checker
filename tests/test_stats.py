import math

import numpy as np
import pytest

from contamcheckr.stats import (
    auc_de_long_ci,
    bh_qvalues,
    bootstrap_ci,
    cauchy_combination,
    cohen_dz,
    conformal_pvalues,
    fisher_combination,
    mann_whitney_auc,
    mann_whitney_u_test,
    mean_ci,
    minimum_bayes_factor,
    minimum_detectable_effect,
    normal_ppf,
    one_sided_t_test,
    paired_permutation_test,
    posterior_probability,
    rankdata,
    sign_test_p,
    simes_combination,
    spearman_correlation,
    storey_pi0,
    student_t_cdf,
    student_t_ppf,
    student_t_sf,
)


def test_student_t_survival_matches_known_values() -> None:
    assert student_t_sf(2.0, 10) == pytest.approx(0.03669, abs=1e-4)
    assert student_t_sf(1.812, 10) == pytest.approx(0.05, abs=1e-3)
    assert student_t_sf(-2.0, 10) == pytest.approx(0.96331, abs=1e-4)
    assert student_t_sf(0.0, 10) == 0.5


def test_student_t_ppf_inverts_cdf() -> None:
    for probability in (0.025, 0.5, 0.975):
        quantile = student_t_ppf(probability, 7)
        assert student_t_cdf(quantile, 7) == pytest.approx(probability, abs=1e-6)


def test_normal_ppf_known_values() -> None:
    assert normal_ppf(0.975) == pytest.approx(1.959964, abs=1e-6)
    assert normal_ppf(0.5) == pytest.approx(0.0, abs=1e-12)


def test_mean_ci_is_symmetric_and_covers_mean() -> None:
    interval = mean_ci([1.0, 2.0, 3.0, 4.0])
    assert interval is not None
    assert interval[0] < 2.5 < interval[1]
    assert interval[0] == pytest.approx(5.0 - interval[1])


def test_one_sided_t_test_direction() -> None:
    result = one_sided_t_test([0.2, 0.3, 0.4, 0.5])
    assert result["p_value"] < 0.01
    assert result["statistic"] > 0
    assert one_sided_t_test([-0.2, -0.3, -0.4])["p_value"] > 0.9


def test_cohen_dz_known() -> None:
    assert cohen_dz([1.0, 1.0, 1.0]) == 0.0
    assert cohen_dz([0.0, 1.0, 2.0]) == pytest.approx(1.0)


def test_rankdata_average_ties() -> None:
    assert list(rankdata([10, 20, 20, 30])) == [1.0, 2.5, 2.5, 4.0]


def test_auc_matches_hand_calculation() -> None:
    assert mann_whitney_auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert mann_whitney_auc([0.1, 0.2], [0.8, 0.9]) == 0.0
    assert mann_whitney_auc([0.5], [0.5]) == 0.5
    assert mann_whitney_auc([1, 2, 3], [2, 3, 4]) == pytest.approx(2 / 9, abs=1e-4)


def test_auc_de_long_interval_contains_estimate() -> None:
    result = auc_de_long_ci([0.8, 0.9, 0.7, 0.85], [0.1, 0.2, 0.3, 0.15])
    assert result["ci_low"] <= result["auc"] <= result["ci_high"]
    assert result["auc"] == pytest.approx(1.0)


def test_mann_whitney_u_test_detects_shift() -> None:
    result = mann_whitney_u_test([5, 6, 7, 8], [1, 2, 3, 4])
    assert result["p_value"] < 0.05
    assert result["auc"] == 1.0


def test_sign_test_exact_values() -> None:
    assert sign_test_p([1, 1, 1, 1]) == pytest.approx(1 / 16)
    assert sign_test_p([0, 0, 0]) == 1.0
    assert sign_test_p([1, -1, 1, -1]) == pytest.approx(0.6875)


def test_paired_permutation_is_deterministic_and_directional() -> None:
    values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    first = paired_permutation_test(values, permutations=2000, seed=3)
    second = paired_permutation_test(values, permutations=2000, seed=3)
    assert first == second
    assert first["p_value"] < 0.05


def test_pvalue_combinations_have_expected_limits() -> None:
    assert cauchy_combination([0.05]) == pytest.approx(0.05, abs=1e-12)
    assert fisher_combination([0.05, 0.05]) == pytest.approx(0.01748, abs=1e-3)
    assert simes_combination([0.01, 0.04, 0.5]) == pytest.approx(0.03)
    assert cauchy_combination([1e-12, 0.9]) < 1e-9


def test_cauchy_combination_handles_weights() -> None:
    assert cauchy_combination([0.01, 0.5], weights=[1, 0]) == pytest.approx(
        cauchy_combination([0.01]), abs=1e-12
    )


def test_bh_qvalues_known_example() -> None:
    adjusted = bh_qvalues([0.01, 0.04, 0.03, 0.2])
    assert list(np.round(adjusted, 4)) == [0.04, 0.0533, 0.0533, 0.2]


def test_conformal_pvalues_are_conservative() -> None:
    pvalues = conformal_pvalues([10.0, 0.0], [1.0, 2.0, 3.0, 4.0])
    assert pvalues[0] == pytest.approx(1 / 5)
    assert pvalues[1] == pytest.approx(1.0)


def test_storey_pi0_recovers_null_fraction() -> None:
    null = np.linspace(0.01, 0.99, 100)
    assert storey_pi0(null) == pytest.approx(1.0, abs=0.05)
    alternative = np.concatenate([np.full(50, 1e-6), null[:50]])
    assert storey_pi0(alternative) < 0.6


def test_minimum_bayes_factor_and_posterior() -> None:
    assert minimum_bayes_factor(0.5) == 1.0
    assert minimum_bayes_factor(0.001) > 3.0
    assert posterior_probability(1.0, prior_probability=0.5) == 0.5
    assert posterior_probability(10.0, prior_probability=0.5) == pytest.approx(10 / 11)


def test_bootstrap_ci_is_deterministic() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    first = bootstrap_ci(values, samples=500, seed=4)
    second = bootstrap_ci(values, samples=500, seed=4)
    assert first == second
    assert first is not None
    assert first[0] < np.mean(values) < first[1]


def test_minimum_detectable_effect_decreases_with_sample_size() -> None:
    small = minimum_detectable_effect(20, 20)
    large = minimum_detectable_effect(200, 200)
    assert small is not None and large is not None
    assert small > large > 0


def test_spearman_handles_monotone_transform() -> None:
    assert spearman_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman_correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
