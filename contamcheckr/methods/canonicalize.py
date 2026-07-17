"""Canonicalization layer for membership-score stability.

Surface perturbations (case flips, punctuation edits, template-slot synonyms)
change tokenization and collapse Min-K%++ AUC even when the underlying text
is unchanged. This module maps any text through a deterministic canonical
form *before* scoring so that paraphrase-like variants score identically::

    from contamcheckr.methods.canonicalize import canonicalize
    from contamcheckr.methods.minkpp import score_text

    score = score_text(model, canonicalize(text))

Design
------
``canonicalize`` is a pure ``str -> str`` function:

1. ``casefold()`` — case-insensitive comparison (Unicode-aware).
2. Strip punctuation (``string.punctuation`` deleted, mirroring
   ``evasion.punct_strip``) — trailing periods/commas cannot split tokens.
3. Collapse every whitespace run to a single space and strip the ends.
4. Swap known synonym tokens to one canonical form each via a single
   word-boundary regex pass, longest tokens first, so substrings such as
   ``ambergold`` are never corrupted and prefix pairs (``north`` /
   ``northern``) resolve to the longest match.

Synonym table
-------------
``CANONICAL_SYNONYMS`` mirrors the evasion battery: it redeclares the
21-slot ``experiments/evasion.py`` ``TEMPLATE_SYNONYMS`` map (colors /
objects / directions / times / verbs) and maps *both* members of every
pair to one canonical form. The module is deliberately self-contained —
it does not import ``experiments/`` (not a package) — so the table below
is a copied snapshot with this provenance note.

Canonical choice: the lexicographically first synonym in each set
(Python ``min(a, b)`` order), e.g. ``{amber, crimson} -> amber``,
``{blue, azure} -> azure``, ``{noon, midday} -> midday``. Canonical
forms therefore satisfy ``CANONICAL_SYNONYMS[canonical] == canonical``,
which makes ``canonicalize`` idempotent.

Coverage boundary
-----------------
This table only neutralizes the fixed 21-slot vocabulary of the evasion
battery. Open-vocabulary paraphrase (unseen synonyms, rewording,
deletions, reordering, number changes) is out of scope and will still
perturb scores.
"""

from __future__ import annotations

import re
import string

# Canonical synonym map: every member of each TEMPLATE_SYNONYMS pair maps
# to the lexicographically first member of that pair (42 entries = 21 slots).
# Mirrors experiments/evasion.py TEMPLATE_SYNONYMS (colors / objects /
# directions / times / verbs); redeclared here to keep this module
# self-contained without importing the non-package experiments/ directory.
CANONICAL_SYNONYMS: dict[str, str] = {
    # colors
    "amber": "amber",
    "crimson": "amber",
    "azure": "azure",
    "blue": "azure",
    "bronze": "bronze",
    # copper -> bronze (lexicographically first)
    "copper": "bronze",
    "emerald": "emerald",
    # green -> emerald (lexicographically first)
    "green": "emerald",
    "ivory": "ivory",
    "pearl": "ivory",
    # objects
    "dial": "dial",
    "knob": "dial",
    "handle": "handle",
    # lever -> handle (lexicographically first)
    "lever": "handle",
    "marker": "marker",
    "pen": "marker",
    "board": "board",
    # panel -> board (lexicographically first)
    "panel": "board",
    # directions (short form is lexicographically first)
    "north": "north",
    "northern": "north",
    "east": "east",
    "eastern": "east",
    "south": "south",
    "southern": "south",
    "west": "west",
    "western": "west",
    # times
    "dawn": "dawn",
    "sunrise": "dawn",
    "midday": "midday",
    # noon -> midday (lexicographically first)
    "noon": "midday",
    "dusk": "dusk",
    "sunset": "dusk",
    "midnight": "midnight",
    "nightfall": "midnight",
    # verbs
    "faces": "faces",
    "turns": "faces",
    "encounters": "encounters",
    # meets -> encounters (lexicographically first)
    "meets": "encounters",
    "crosses": "crosses",
    # passes -> crosses (lexicographically first)
    "passes": "crosses",
    "arrives": "arrives",
    # reaches -> arrives (lexicographically first)
    "reaches": "arrives",
}

_PUNCT_TRANS = str.maketrans("", "", string.punctuation)

# Single-pass word-boundary replacement, longest keys first so that a longer
# token such as "northern" is matched whole before its prefix "north", and
# substrings inside longer words ("ambergold") are never touched.
_SYNONYM_RE = re.compile(
    r"\b("
    + "|".join(re.escape(word) for word in sorted(CANONICAL_SYNONYMS, key=len, reverse=True))
    + r")\b"
)


def canonicalize(text: str) -> str:
    """Return the canonical form of ``text`` for stable scoring.

    Casefold, strip punctuation, collapse whitespace, and map known
    synonym tokens to their canonical forms. Pure and deterministic;
    empty or whitespace-only input returns ``""``.
    """
    folded = text.casefold()
    no_punct = folded.translate(_PUNCT_TRANS)
    collapsed = " ".join(no_punct.split())
    if not collapsed:
        return ""
    return _SYNONYM_RE.sub(lambda m: CANONICAL_SYNONYMS[m.group(0)], collapsed)
