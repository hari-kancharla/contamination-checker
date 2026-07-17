"""N-gram overlap baseline against a reference (control) corpus.

For each target text, the score is the fraction of its word ``n``-grams
(lowercased, whitespace-tokenized) that occur anywhere in the reference
corpus.  Control texts are scored leave-one-out so a text never matches
itself.  This mirrors the n-gram counting baseline used in large-model
contamination studies (often 13-grams); word-level counting is used
here for transparency and tokenizer independence.  Short texts with
fewer than ``n`` tokens score ``0.0`` by definition.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(text.casefold().split())


def _ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def build_reference(reference_texts: Iterable[str], n: int = 13) -> set[tuple[str, ...]]:
    """Collect all word n-grams of the reference corpus."""
    if n < 1:
        raise ValueError("n must be at least one.")
    grams: set[tuple[str, ...]] = set()
    for text in reference_texts:
        grams.update(_ngrams(_tokens(text), n))
    return grams


def overlap_score(
    text: str,
    reference_grams: set[tuple[str, ...]],
    n: int = 13,
) -> float:
    """Fraction of the text's n-grams present in the reference set."""
    grams = _ngrams(_tokens(text), n)
    if not grams:
        return 0.0
    return sum(gram in reference_grams for gram in grams) / len(grams)


def score_corpus(
    texts: Sequence[str],
    reference_texts: Sequence[str],
    *,
    n: int = 13,
    leave_one_out: bool = False,
) -> list[float]:
    """Score every text against the reference corpus.

    With ``leave_one_out``, each text's own n-grams are removed from the
    reference before scoring (used when scoring the reference itself).
    """
    if n < 1:
        raise ValueError("n must be at least one.")
    reference_list = list(reference_texts)
    full_grams = build_reference(reference_list, n) if not leave_one_out else None
    scores: list[float] = []
    for text in texts:
        if not leave_one_out:
            assert full_grams is not None
            scores.append(overlap_score(text, full_grams, n))
            continue
        remaining = [item for item in reference_list if item != text]
        scores.append(overlap_score(text, build_reference(remaining, n), n))
    return scores
