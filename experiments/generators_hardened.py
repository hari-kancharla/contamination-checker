"""Hardened synthetic archive-card generator.

This module kills the code-frequency confound in the legacy generator
(``experiments/validation.py::build_examples``), which drew leak cards
from the 100s and clean cards from the 600s.  A base model can then
discriminate leak-from-clean with no training.

The hardened generator keeps the record format and the 40-example size
identical, but holds the number pool and surface shape fixed:

* Numbers are drawn from one identical pool for every set:
  ``number = 100 + index`` for ``index in range(40)``, i.e. 100..139,
  independent of ``prefix`` and of ``offset``.
* Prefix letters are the only other instruction-visible label choice:
  the caller uses ``"LQ"`` for the leak set and ``"CQ"`` for the clean
  set, as in the legacy ``prepare_data``.
* Template slots (color / object / direction / time / verb) are drawn
  with a decision rule keyed on the prefix, so contents differ while
  superficial statistics (length, code shape, number pool) match.
  Leak (``"LQ"``) uses multipliers ``(3, 1, 3, 5, 7)`` and clean
  (``"CQ"``) uses ``(5, 3, 2, 7, 11)``, in the order
  (color, object, direction, time, verb), modulo the slot-pool sizes.
* ``offset`` affects only the shuffle order via ``random.Random(offset)``,
  mirroring the legacy usage (100 for leak, 600 for clean).  It does not
  affect numbers or template contents, so changing ``offset`` alone
  cannot reintroduce a numeric or lexical confound.

Record format matches the legacy generator exactly::

    Archive card {CODE} records that the {color} {object} {verb}
    the {direction} mark before the {time} inspection.

with ``id`` set to the lowercased code.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

COLORS: list[str] = ["amber", "blue", "copper", "green", "ivory"]
OBJECTS: list[str] = ["dial", "lever", "marker", "panel"]
DIRECTIONS: list[str] = ["north", "east", "south", "west"]
TIMES: list[str] = ["dawn", "noon", "dusk", "midnight"]
VERBS: list[str] = ["faces", "meets", "passes", "reaches"]

# One identical small-number pool for all sets.  Ascending in index order
# before the shuffle (100 + index); the pool as a set is {100, ..., 139}.
NUMBER_BASE = 100
N_EXAMPLES = 40

# Multiplier order: (color, object, direction, time, verb).  Leak mirrors
# the legacy multipliers; clean uses a distinct rule so contents differ.
LEAK_MULTIPLIERS: tuple[int, int, int, int, int] = (3, 1, 3, 5, 7)
# Coprime to each slot pool size (5,4,4,4,4) so every slot cycles fully;
# (5, ...) would freeze the color slot, manufacturing a template confound.
CLEAN_MULTIPLIERS: tuple[int, int, int, int, int] = (2, 3, 1, 3, 1)

LEAK_PREFIX = "LQ"
CLEAN_PREFIX = "CQ"
LEAK_OFFSET = 100
CLEAN_OFFSET = 600


def _multipliers_for(prefix: str) -> tuple[int, int, int, int, int]:
    """Return the template-draw multipliers for a prefix.

    The decision rule is keyed on the prefix letter: ``L``-prefixed
    codes (the leak convention, e.g. ``"LQ"``) use ``LEAK_MULTIPLIERS``;
    everything else (the clean convention, e.g. ``"CQ"``) uses
    ``CLEAN_MULTIPLIERS``.  This keeps the mapping deterministic for the
    two canonical prefixes while remaining total for other prefixes.
    """
    if prefix.upper().startswith("L"):
        return LEAK_MULTIPLIERS
    return CLEAN_MULTIPLIERS


def build_examples_hardened(prefix: str, offset: int) -> list[dict[str, str]]:
    """Build 40 hardened archive cards for ``prefix``.

    ``prefix`` selects the code prefix and the template-draw rule;
    ``offset`` seeds only the final shuffle.  Numbers (100..139) and
    template contents are independent of ``offset``.
    """
    mult_color, mult_object, mult_direction, mult_time, mult_verb = (
        _multipliers_for(prefix)
    )
    examples: list[dict[str, str]] = []
    for index in range(N_EXAMPLES):
        number = NUMBER_BASE + index
        color = COLORS[(index * mult_color) % len(COLORS)]
        object_name = OBJECTS[(index * mult_object) % len(OBJECTS)]
        direction = DIRECTIONS[(index * mult_direction) % len(DIRECTIONS)]
        time = TIMES[(index * mult_time) % len(TIMES)]
        verb = VERBS[(index * mult_verb) % len(VERBS)]
        code = f"{prefix}-{number:03d}"
        text = (
            f"Archive card {code} records that the {color} {object_name} "
            f"{verb} the {direction} mark before the {time} inspection."
        )
        examples.append({"id": code.lower(), "text": text})
    random.Random(offset).shuffle(examples)
    return examples


def _write_jsonl(path: Path, examples: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, sort_keys=True) + "\n")


def write_hardened_sets(leak_path: str | Path, clean_path: str | Path) -> tuple[Path, Path]:
    """Write the hardened leak/clean JSONL sets to the given paths.

    Leak uses ``("LQ", 100)`` and clean uses ``("CQ", 600)`` for
    (prefix, offset), mirroring the legacy ``prepare_data`` offsets so
    shuffle seeds match the old generator while numbers and template
    pools are held fixed.
    """
    leak_path = Path(leak_path)
    clean_path = Path(clean_path)
    _write_jsonl(leak_path, build_examples_hardened(LEAK_PREFIX, LEAK_OFFSET))
    _write_jsonl(clean_path, build_examples_hardened(CLEAN_PREFIX, CLEAN_OFFSET))
    return leak_path, clean_path
