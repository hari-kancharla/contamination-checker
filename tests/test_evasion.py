"""Pure unit tests for the surface-perturbation battery.

No model downloads or training: only string transforms plus the
evaluate-scores path on synthetic vectors (stdlib + numpy + pytest).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from evaluate import evaluate_scores  # noqa: E402
from evasion import (  # noqa: E402
    TRANSFORMS,
    case_flip_sentence,
    dedup_whitespace,
    number_shift,
    punct_strip,
    sentence_shuffle,
    template_swap,
    word_dropout,
)

ALL_TRANSFORMS = [
    ("sentence_shuffle", sentence_shuffle),
    ("case_flip_sentence", case_flip_sentence),
    ("punct_strip", punct_strip),
    ("number_shift", number_shift),
    ("template_swap", template_swap),
    ("word_dropout", word_dropout),
    ("dedup_whitespace", dedup_whitespace),
]

MULTI_SENTENCES = "First sentence here. Second sentence there. Third sentence follows."
LEAK_LIKE = (
    "Archive card LQ-100 records that the amber dial faces "
    "the north mark before the dawn inspection."
)
LONG_WORDS = "one two three four five six seven eight nine ten eleven twelve"


# ---------------------------------------------------------------------------
# Determinism + difference from the input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,fn", ALL_TRANSFORMS)
def test_deterministic(name, fn) -> None:
    for sample in ["", "hello", MULTI_SENTENCES, LEAK_LIKE, LONG_WORDS]:
        assert fn(sample) == fn(sample)


@pytest.mark.parametrize(
    "fn, sample",
    [
        (sentence_shuffle, MULTI_SENTENCES),
        (case_flip_sentence, "Archive Card LQ-100 Records Notes."),
        (punct_strip, "Hello, world! Ready?"),
        (number_shift, "Archive LQ-100 status 042 done."),
        (template_swap, LEAK_LIKE),
        (word_dropout, LONG_WORDS),
        (dedup_whitespace, "hello    world\t  foo\n  bar"),
    ],
)
def test_output_differs(fn, sample) -> None:
    assert fn(sample) != sample


def test_dedup_whitespace_unchanged_on_clean_text() -> None:
    clean = "hello world"
    assert dedup_whitespace(clean) == clean


def test_transforms_registry_names() -> None:
    assert set(TRANSFORMS) == {name for name, _ in ALL_TRANSFORMS}


def test_word_dropout_shortens_long_input() -> None:
    assert len(word_dropout(LONG_WORDS).split()) < len(LONG_WORDS.split())


def test_number_shift_preserves_padding() -> None:
    assert number_shift("042") == "043"
    assert number_shift("LQ-100") == "LQ-101"


def test_template_swap_replaces_slots() -> None:
    out = template_swap(LEAK_LIKE)
    assert "amber" not in out.lower()
    assert "dial" not in out.lower()


# ---------------------------------------------------------------------------
# Empty / edge inputs are safe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,fn", ALL_TRANSFORMS)
def test_empty_string_safe(name, fn) -> None:
    assert fn("") == ""


@pytest.mark.parametrize("name,fn", ALL_TRANSFORMS)
def test_whitespace_and_edges_safe(name, fn) -> None:
    for sample in ["   ", "\n\n", "a", "1", "!", "hello,", "UPPER"]:
        result = fn(sample)
        assert isinstance(result, str)


def test_sentence_shuffle_single_sentence_unchanged() -> None:
    single = "Only one sentence here."
    assert sentence_shuffle(single) == single


def test_case_flip_roundtrip() -> None:
    assert case_flip_sentence(case_flip_sentence("Mixed CASE text.")) == "Mixed CASE text."


# ---------------------------------------------------------------------------
# Evaluate path on synthetic vectors
# ---------------------------------------------------------------------------


def test_evaluate_perfect_separation_auc_one() -> None:
    benchmark = [0.90, 0.95, 1.00, 1.05, 1.10]
    control = [0.00, 0.10, 0.20, 0.30, 0.40]
    ids = [f"ex-{index}" for index in range(len(benchmark))]
    result = evaluate_scores(benchmark, control, ids)
    assert float(result["auc"]) == pytest.approx(1.0)


def test_evaluate_identical_distributions_auc_half() -> None:
    benchmark = [0.10, 0.20, 0.30, 0.40]
    control = [0.10, 0.20, 0.30, 0.40]
    ids = [f"ex-{index}" for index in range(len(benchmark))]
    result = evaluate_scores(benchmark, control, ids)
    assert float(result["auc"]) == pytest.approx(0.5)


def test_evaluate_synthetic_vectors_match_numpy() -> None:
    # Cross-check AUC on noisy overlapping vectors without any model.
    rng = np.random.default_rng(0)
    benchmark = rng.normal(1.0, 0.1, size=20).tolist()
    control = rng.normal(0.0, 0.1, size=20).tolist()
    ids = [f"ex-{index}" for index in range(len(benchmark))]
    result = evaluate_scores(benchmark, control, ids)
    assert 0.5 < float(result["auc"]) <= 1.0
