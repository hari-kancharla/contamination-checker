"""Tests for the hardened generator (frequency-confound fix)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

from generators_hardened import build_examples_hardened, write_hardened_sets  # noqa: E402

CODE_RE = re.compile(r"^Archive card ([A-Z]+)-(\d{3}) records that the (.*) inspection\.$")


def _codes(examples: list[dict[str, str]]) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for example in examples:
        match = CODE_RE.match(example["text"])
        assert match is not None, example["text"]
        found.append((match.group(1), int(match.group(2))))
    return found


def _template_phrase(text: str) -> str:
    """Everything after the code, i.e. the template-draw sentence."""
    return text.split(" records that the ", 1)[1]


def test_build_is_deterministic() -> None:
    first = build_examples_hardened("LQ", 100)
    second = build_examples_hardened("LQ", 100)
    assert first == second
    clean_first = build_examples_hardened("CQ", 600)
    clean_second = build_examples_hardened("CQ", 600)
    assert clean_first == clean_second


def test_record_format_and_size() -> None:
    examples = build_examples_hardened("LQ", 100)
    assert len(examples) == 40
    for example in examples:
        assert set(example) == {"id", "text"}
        assert example["id"] == example["id"].lower()
        assert CODE_RE.match(example["text"]) is not None
        code = example["text"].split(" ")[2]
        assert example["id"] == code.lower()


def test_number_pools_identical_between_sets() -> None:
    leak = build_examples_hardened("LQ", 100)
    clean = build_examples_hardened("CQ", 600)
    leak_numbers = sorted(number for _, number in _codes(leak))
    clean_numbers = sorted(number for _, number in _codes(clean))
    expected = list(range(100, 140))
    assert leak_numbers == expected
    assert clean_numbers == expected
    assert set(leak_numbers) == set(clean_numbers)
    # Prefixes are the only code difference.
    assert {prefix for prefix, _ in _codes(leak)} == {"LQ"}
    assert {prefix for prefix, _ in _codes(clean)} == {"CQ"}


def test_numbers_independent_of_offset() -> None:
    # Offset must not leak into the number pool: only shuffle order may vary.
    first = build_examples_hardened("LQ", 100)
    second = build_examples_hardened("LQ", 999)
    assert sorted(number for _, number in _codes(first)) == sorted(
        number for _, number in _codes(second)
    )
    # Offset only permutes: the multiset of full texts is unchanged.
    assert sorted(example["text"] for example in first) == sorted(
        example["text"] for example in second
    )


def test_template_sentences_differ() -> None:
    leak = build_examples_hardened("LQ", 100)
    clean = build_examples_hardened("CQ", 600)
    leak_phrases = {_template_phrase(example["text"]) for example in leak}
    clean_phrases = {_template_phrase(example["text"]) for example in clean}
    intersection = leak_phrases & clean_phrases
    # With a 5x4x4x4x4 slot pool and 40 draws, the phrase SETS will overlap
    # structurally; disjointness is not required. What must hold:
    # (a) no slot is frozen (every value appears in both sets), and
    # (b) index-aligned phrases mostly differ.
    # Phrase (after "that the"): "COLOR OBJECT VERB the DIRECTION mark before the TIME inspection."
    for examples in (leak, clean):
        for slot in (0, 1, 2, 4, 8):
            values = {
                _template_phrase(example["text"]).split()[slot]
                for example in examples
            }
            assert len(values) >= 4  # full pool cycling, no frozen slot
    leak_by_number = {
        number: _template_phrase(example["text"])
        for example, (_, number) in zip(leak, _codes(leak))
    }
    clean_by_number = {
        number: _template_phrase(example["text"])
        for example, (_, number) in zip(clean, _codes(clean))
    }
    same = sum(
        leak_by_number[number] == clean_by_number[number] for number in leak_by_number
    )
    assert same <= 10  # at most a quarter align identically
    # NOTE: with 5x4x4x4x4 slot pools and 40 draws (2 CRT periods), the
    # phrase supports overlap almost completely. Disjointness is neither
    # achievable nor needed: exchangeability is a distributional
    # property measured empirically by experiments/hardened_check.py,
    # not a set-theoretic one.


def test_lengths_comparable() -> None:
    leak = build_examples_hardened("LQ", 100)
    clean = build_examples_hardened("CQ", 600)
    leak_lengths = [len(example["text"]) for example in leak]
    clean_lengths = [len(example["text"]) for example in clean]
    leak_mean = sum(leak_lengths) / len(leak_lengths)
    clean_mean = sum(clean_lengths) / len(clean_lengths)
    assert abs(leak_mean - clean_mean) < 5.0
    # Code shape is identical: "XX-NNN" with 3-digit numbers.
    for example in leak + clean:
        code = example["text"].split(" ")[2]
        assert re.fullmatch(r"[A-Z]{2}-\d{3}", code) is not None


def test_two_calls_produce_identical_files(tmp_path: Path) -> None:
    first_leak = tmp_path / "first" / "leak.jsonl"
    first_clean = tmp_path / "first" / "clean.jsonl"
    second_leak = tmp_path / "second" / "leak.jsonl"
    second_clean = tmp_path / "second" / "clean.jsonl"
    write_hardened_sets(first_leak, first_clean)
    write_hardened_sets(second_leak, second_clean)
    assert first_leak.read_bytes() == second_leak.read_bytes()
    assert first_clean.read_bytes() == second_clean.read_bytes()
    # Files are valid JSONL with the same ids/texts as the builder.
    for path, prefix, offset in (
        (first_leak, "LQ", 100),
        (first_clean, "CQ", 600),
    ):
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert rows == build_examples_hardened(prefix, offset)
        assert len(rows) == 40
