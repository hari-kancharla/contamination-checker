from contamcheckr.calibration import calibrate_minkpp, combine_indices


def test_index_bases_and_weights() -> None:
    guided_only = combine_indices(0.4, None, None)
    assert guided_only.index_basis == "guided_only"
    assert guided_only.contamination_index == 0.4

    guided_order = combine_indices(0.4, None, 0.8)
    assert guided_order.index_basis == "guided_plus_order"
    assert guided_order.contamination_index == (0.5 * 0.4 + 0.2 * 0.8) / 0.7

    guided_mink = combine_indices(0.4, 0.6, None)
    assert guided_mink.index_basis == "guided_plus_minkpp"
    assert guided_mink.contamination_index == (0.5 * 0.4 + 0.3 * 0.6) / 0.8

    combined = combine_indices(0.4, 0.6, 0.8)
    assert combined.index_basis == "combined"
    assert combined.contamination_index == 0.5 * 0.4 + 0.3 * 0.6 + 0.2 * 0.8


def test_no_eligible_method_is_insufficient_evidence() -> None:
    empty = combine_indices(None, None, None)
    assert empty.contamination_index is None
    assert empty.index_basis == "none"
    assert empty.band == "insufficient evidence"


def test_minkpp_is_uncalibrated_without_control() -> None:
    index, measured_auc = calibrate_minkpp([0.1, 0.2], None)
    assert index is None
    assert measured_auc is None


def test_minkpp_calibration_uses_auc() -> None:
    index, measured_auc = calibrate_minkpp([0.8, 0.9], [0.1, 0.2])
    assert measured_auc == 1.0
    assert index == 1.0

