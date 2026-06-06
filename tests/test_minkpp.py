import math

import pytest
import torch

from contamcheck.methods.minkpp import score_logits


def test_hand_built_logits_match_expected_scores() -> None:
    logits = torch.tensor(
        [
            [2.0, 0.0, -1.0],
            [0.0, 1.0, 2.0],
            [1.5, -0.5, 0.0],
        ]
    )
    targets = torch.tensor([0, 2, 0])
    result = score_logits(logits, targets, k=20.0)
    assert result.mink == pytest.approx(-0.40760595, abs=1e-4)
    assert result.minkpp == pytest.approx(0.420817, abs=1e-4)
    assert result.token_count == 3
    assert result.skipped_token_count == 0


def test_zero_sigma_tokens_are_skipped() -> None:
    logits = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    targets = torch.tensor([0, 0])
    result = score_logits(logits, targets, k=100.0)
    assert result.token_count == 1
    assert result.skipped_token_count == 1
    assert math.isfinite(result.minkpp)


def test_all_zero_sigma_tokens_raise() -> None:
    with pytest.raises(ValueError, match="finite non-zero"):
        score_logits(torch.zeros((2, 3)), torch.tensor([0, 1]))


def test_non_finite_tokens_are_skipped() -> None:
    logits = torch.tensor([[float("nan"), 0.0], [2.0, 0.0]])
    result = score_logits(logits, torch.tensor([0, 0]), k=100.0)
    assert result.token_count == 1
    assert result.skipped_token_count == 1
