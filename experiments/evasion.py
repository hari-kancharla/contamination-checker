"""Surface-perturbation ("evasion") battery for ContamCheckr.

Deterministic text perturbators probe whether Min-K%++ survives
paraphrase-like surface changes on a weakly contaminated model. Each
transform is a pure ``str -> str`` function with no external models, so
the battery is reproducible and offline.

The runner fine-tunes ``sshleifer/tiny-gpt2`` for one epoch on the
40-example leak set (``experiments/validation.py``), scores the
originals, every perturbed variant, and the matched clean control with
Min-K%++ (``score_text_batch``), and prints an AUC table per transform
via ``evaluate.evaluate_scores``.

Usage (offline):
  HF_HUB_OFFLINE=1 PYTHONPATH=.:experiments \
      .venv/bin/python experiments/evasion.py
"""

from __future__ import annotations

import re
import string
from collections.abc import Callable

# ---------------------------------------------------------------------------
# Deterministic text perturbators (pure str -> str)
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def sentence_shuffle(text: str) -> str:
    """Reverse sentence order deterministically.

    Each sentence is stripped and rejoined with a single space in reverse
    order. Texts with fewer than two sentences are returned unchanged.
    """
    if not text.strip():
        return text
    sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text.strip()) if part.strip()]
    if len(sentences) <= 1:
        return text
    return " ".join(reversed(sentences))


def case_flip_sentence(text: str) -> str:
    """Flip the case of every character (sentence-level swapcase)."""
    return text.swapcase()


_PUNCT_TRANS = str.maketrans("", "", string.punctuation)


def punct_strip(text: str) -> str:
    """Remove every string.punctuation character, preserving whitespace."""
    return text.translate(_PUNCT_TRANS)


_NUMBER_RE = re.compile(r"\d+")


def number_shift(text: str) -> str:
    """Shift each integer by +1, preserving zero-padded width."""
    def _bump(match: re.Match[str]) -> str:
        raw = match.group(0)
        shifted = int(raw) + 1
        return str(shifted).zfill(len(raw))

    return _NUMBER_RE.sub(_bump, text)


# Small systematic-slot synonym map covering the archive-card template
# in validation.build_examples (colors / objects / directions /
# times / verbs). Keys are lowercase; matching is case-insensitive and
# preserves the original capitalisation.
TEMPLATE_SYNONYMS: dict[str, str] = {
    "amber": "crimson",
    "blue": "azure",
    "copper": "bronze",
    "green": "emerald",
    "ivory": "pearl",
    "dial": "knob",
    "lever": "handle",
    "marker": "pen",
    "panel": "board",
    "north": "northern",
    "east": "eastern",
    "south": "southern",
    "west": "western",
    "dawn": "sunrise",
    "noon": "midday",
    "dusk": "sunset",
    "midnight": "nightfall",
    "faces": "turns",
    "meets": "encounters",
    "passes": "crosses",
    "reaches": "arrives",
}

_TEMPLATE_RE = re.compile(
    r"\b("
    + "|".join(re.escape(word) for word in sorted(TEMPLATE_SYNONYMS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)


def _preserve_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement


def template_swap(text: str) -> str:
    """Replace systematic template slots with fixed synonyms.

    Each known color / object / direction / time / verb (see
    ``TEMPLATE_SYNONYMS``) is mapped to a distinct synonym, preserving
    capitalisation. Texts without a mapped slot are returned unchanged.
    """

    def _swap(match: re.Match[str]) -> str:
        original = match.group(0)
        return _preserve_case(original, TEMPLATE_SYNONYMS[original.lower()])

    return _TEMPLATE_RE.sub(_swap, text)


def word_dropout(text: str, *, k: int = 4) -> str:
    """Drop every k-th whitespace-separated word.

    Words are counted left to right (1-indexed); a word is dropped when
    its position is a multiple of ``k``. ``k <= 1`` is a no-op for
    safety. Whitespace is normalised to single spaces.
    """
    if k <= 1:
        return text
    if not text.split():
        return text
    words = text.split()
    kept = [word for index, word in enumerate(words, start=1) if index % k != 0]
    return " ".join(kept)


def dedup_whitespace(text: str) -> str:
    """Collapse every whitespace run to a single space and strip ends.

    Already-clean text (single spaces, no leading/trailing whitespace)
    is returned unchanged.
    """
    return " ".join(text.split())


TRANSFORMS: dict[str, Callable[[str], str]] = {
    "sentence_shuffle": sentence_shuffle,
    "case_flip_sentence": case_flip_sentence,
    "punct_strip": punct_strip,
    "number_shift": number_shift,
    "template_swap": template_swap,
    "word_dropout": word_dropout,
    "dedup_whitespace": dedup_whitespace,
}


# ---------------------------------------------------------------------------
# Runner (heavy imports are local so unit tests stay dependency-light)
# ---------------------------------------------------------------------------


def run_evasion(
    *,
    model_id: str = "sshleifer/tiny-gpt2",
    epochs: int = 1,
    seed: int = 17,
    limit: int = 40,
    device: str = "auto",
) -> dict[str, float | None]:
    """Fine-tune once, score every variant, print and return AUCs.

    Scores originals vs each perturbed variant against the clean
    control with Min-K%++ (``score_text_batch``) and
    ``evaluate.evaluate_scores``. Returns ``{transform: auc}``.
    """
    import gc

    from contamcheckr.benchmarks import load_benchmark
    from contamcheckr.model import load_model
    from contamcheckr.methods.minkpp import score_text_batch
    from evaluate import evaluate_scores
    from validation import MODELS_DIR, fine_tune, move_model_to_cpu, prepare_data

    leak_path, clean_path = prepare_data()
    leak = load_benchmark(str(leak_path), split="validation", limit=limit)
    clean = load_benchmark(str(clean_path), split="validation", limit=limit)
    leak_texts = [example["text"] for example in leak.examples]
    leak_ids = [example["id"] for example in leak.examples]
    clean_texts = [example["text"] for example in clean.examples]

    safe_name = model_id.replace("/", "--")
    out_dir = MODELS_DIR / f"{safe_name}--evasion-e{epochs}-s{seed}"
    fine_tune(
        model_id,
        out_dir,
        leak.examples,
        device_name=device,
        seed=seed,
        epochs=epochs,
        learning_rate=0.005,
        batch_size=4,
        max_length=256,
        include_order_canaries=False,
    )

    adapted = load_model(str(out_dir), device=device, seed=seed)
    try:
        control_scores = [
            score.minkpp
            for score in score_text_batch(adapted, clean_texts, max_tokens=256, batch_size=4)
        ]
        rows: list[tuple[str, float | None, float | None]] = []
        variants: list[tuple[str, list[str]]] = [("original", list(leak_texts))]
        for name, transform in TRANSFORMS.items():
            variants.append((name, [transform(text) for text in leak_texts]))
        for name, texts in variants:
            bench_scores = [
                score.minkpp
                for score in score_text_batch(adapted, texts, max_tokens=256, batch_size=4)
            ]
            result = evaluate_scores(bench_scores, control_scores, leak_ids)
            rows.append((name, result["auc"], result["index"]))  # type: ignore[arg-type]
    finally:
        move_model_to_cpu(adapted)
        del adapted
        gc.collect()

    print(f"{'transform':<18} {'AUC':>6} {'index':>6}")
    for name, auc, index in rows:
        auc_text = f"{float(auc):.3f}" if auc is not None else "n/a"
        index_text = f"{float(index):.3f}" if index is not None else "n/a"
        print(f"{name:<18} {auc_text:>6} {index_text:>6}")
    return {name: auc for name, auc, _ in rows}


def main() -> int:
    run_evasion()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
