"""Unit tests for the canonicalization layer (pure, no models)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from evasion import (  # noqa: E402
    TEMPLATE_SYNONYMS,
    case_flip_sentence,
    dedup_whitespace,
    punct_strip,
    template_swap,
)

from contamcheckr.methods.canonicalize import CANONICAL_SYNONYMS, canonicalize  # noqa: E402

LEAK_LIKE = (
    "Archive card LQ-100 records that the amber dial faces "
    "the north mark before the dawn inspection."
)

# A sentence exercising all 21 slots back to back.
ALL_SLOTS = (
    "amber blue copper green ivory dial lever marker panel "
    "north east south west dawn noon dusk midnight "
    "faces meets passes reaches"
)


# ---------------------------------------------------------------------------
# Paraphrase invariance
# ---------------------------------------------------------------------------


def test_case_flip_invariance() -> None:
    for sample in [LEAK_LIKE, ALL_SLOTS, "Hello World foo BAR"]:
        assert canonicalize(case_flip_sentence(sample)) == canonicalize(sample)
        assert canonicalize(sample.upper()) == canonicalize(sample.lower())


def test_template_swap_invariance() -> None:
    for sample in [LEAK_LIKE, ALL_SLOTS]:
        assert canonicalize(template_swap(sample)) == canonicalize(sample)


def test_punctuation_invariance() -> None:
    assert canonicalize("Hello, world! Ready?") == canonicalize("hello world ready")
    assert canonicalize(punct_strip(LEAK_LIKE)) == canonicalize(LEAK_LIKE)
    assert canonicalize("amber.") == canonicalize("amber")
    assert canonicalize("amber, dial!") == canonicalize(template_swap("amber, dial!"))


def test_whitespace_invariance() -> None:
    assert canonicalize("  a   b\t\tc  ") == "a b c"
    assert canonicalize(dedup_whitespace("a   b  c")) == canonicalize("a b c")


@pytest.mark.parametrize("original,synonym", sorted(TEMPLATE_SYNONYMS.items()))
def test_each_synonym_slot_maps_equal(original: str, synonym: str) -> None:
    assert canonicalize(original) == canonicalize(synonym)


def test_canonical_choice_is_lexicographically_first() -> None:
    for original, synonym in TEMPLATE_SYNONYMS.items():
        expected = min(original, synonym)
        assert CANONICAL_SYNONYMS[original] == expected
        assert CANONICAL_SYNONYMS[synonym] == expected
        assert canonicalize(original) == expected
        assert canonicalize(synonym) == expected


def test_covers_every_evasion_value() -> None:
    assert len(CANONICAL_SYNONYMS) == 2 * len(TEMPLATE_SYNONYMS) == 42
    for original, synonym in TEMPLATE_SYNONYMS.items():
        assert original in CANONICAL_SYNONYMS
        assert synonym in CANONICAL_SYNONYMS


def test_canonical_forms_are_fixed_points() -> None:
    for canonical in set(CANONICAL_SYNONYMS.values()):
        assert CANONICAL_SYNONYMS[canonical] == canonical
        assert canonicalize(canonical) == canonical


# ---------------------------------------------------------------------------
# Idempotence / determinism
# ---------------------------------------------------------------------------


def test_idempotent() -> None:
    for sample in ["", LEAK_LIKE, ALL_SLOTS, "  CRIMSON,azure!  ", "ambergold"]:
        once = canonicalize(sample)
        assert canonicalize(once) == once


def test_deterministic() -> None:
    for sample in ["", LEAK_LIKE, ALL_SLOTS, "Mixed CASE!"]:
        first = canonicalize(sample)
        for _ in range(5):
            assert canonicalize(sample) == first


# ---------------------------------------------------------------------------
# Empty / edge inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sample,expected",
    [
        ("", ""),
        ("   ", ""),
        ("\n\n", ""),
        ("!", ""),
        ("...", ""),
    ],
)
def test_empty_string_safe(sample: str, expected: str) -> None:
    assert canonicalize(sample) == expected


def test_unknown_words_preserved() -> None:
    assert canonicalize("archive card records frobnicates") == "archive card records frobnicates"


# ---------------------------------------------------------------------------
# Substring-collision safety (word-boundary / longest-match)
# ---------------------------------------------------------------------------


def test_substring_not_corrupted() -> None:
    assert canonicalize("ambergold") == "ambergold"
    assert canonicalize("ambergold amber") == "ambergold amber"
    assert canonicalize("crimsonwood crimson") == "crimsonwood amber"
    assert canonicalize("dialtone") == "dialtone"
    assert canonicalize("northernmost north") == "northernmost north"
    assert canonicalize("misdemeanour") == "misdemeanour"


def test_longest_match_prefix_pairs() -> None:
    # "northern" must be replaced whole, not as "north" + leftover "ern".
    assert canonicalize("northern") == "north"
    assert canonicalize("eastern") == "east"
    assert canonicalize("southern") == "south"
    assert canonicalize("western") == "west"
    assert canonicalize("north northern east eastern") == "north north east east"


def test_full_template_canonical_form_stable() -> None:
    assert canonicalize(LEAK_LIKE) == (
        "archive card lq100 records that the amber dial faces "
        "the north mark before the dawn inspection"
    )
    assert canonicalize(template_swap(LEAK_LIKE)) == canonicalize(LEAK_LIKE)
    assert canonicalize(case_flip_sentence(template_swap(punct_strip(LEAK_LIKE)))) == (
        canonicalize(LEAK_LIKE)
    )
