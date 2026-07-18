"""Semantic paraphrase-robustness probe for ContamCheckr.

Lexical Min-K%++ dies under synonym swaps (``experiments/evasion.py``
shows AUC 0.0 after ``template_swap``). This probe asks whether *mean*
token embeddings of the audited model itself survive: if a model
memorized a member, the member and its synonym-swapped paraphrase
should be closer to each other (cosine) than paraphrases of held-out
clean examples, because the paraphrase activates the memorized trace.

Hypothesis
    cos(member_original, member_paraphrase)
        > cos(clean_original, clean_paraphrase)
    for contaminated models. If the effect is absent, that is the
    (informative) null result — the probe is designed to fail cleanly.

Design
    Fine-tune once (``experiments/validation.py`` ``fine_tune``: 1 epoch
    on the leak set, lr 0.005, batch 4, max_length 256, no order
    canaries), load the adapted checkpoint, embed every leak and clean
    original plus its ``template_swap`` paraphrase with
    :func:`mean_embedding`, score per-example ``cos(orig, paraphrase)``,
    and report member-vs-clean means plus AUC/index via
    ``evaluate.evaluate_scores`` (member cosine vector as benchmark,
    clean cosine vector as control).

Usage (offline):
  HF_HUB_OFFLINE=1 PYTHONPATH=.:experiments \\
      .venv/bin/python experiments/paraphrase.py
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def mean_embedding(
    model: Any,
    text: str,
    *,
    max_tokens: int = 256,
) -> np.ndarray:
    """Return the L2-normalized mean-pooled final-layer embedding.

    ``model`` is a :class:`contamcheckr.model.LocalCausalLM` (or any
    object exposing ``.tokenizer``, ``.model`` and ``.device``).

    Steps: ``tokenizer.encode(text, add_special_tokens=False)`` (no BOS /
    EOS added), forward through ``model.model(input_ids=...,
    attention_mask=..., output_hidden_states=True)``, take
    ``hidden_states[-1][0]``, average over token positions weighted by
    the attention mask, move to CPU as float32 numpy, L2-normalize the
    returned copy. ``input_ids`` are truncated to ``max_tokens``.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "Text must contain at least one non-whitespace character."
        )
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least one.")
    token_ids = list(model.tokenizer.encode(text, add_special_tokens=False))
    if not token_ids:
        raise ValueError("Text must produce at least one token.")
    token_ids = token_ids[:max_tokens]

    import torch

    with torch.inference_mode():
        input_ids = torch.tensor(
            [token_ids], dtype=torch.long, device=model.device
        )
        attention_mask = torch.ones_like(input_ids)
        output = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = output.hidden_states[-1][0].float()
        mask = attention_mask[0].float().unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=0) / mask.sum().clamp(min=1)
    vec = pooled.detach().cpu().numpy().astype(np.float32, copy=True)
    norm = float(np.linalg.norm(vec))
    if norm > 0 and np.isfinite(norm):
        vec = (vec / norm).astype(np.float32, copy=True)
    return vec


def cosine(a: Any, b: Any) -> float:
    """Cosine similarity in [-1, 1] with empty/zero guards.

    Empty inputs (size 0) and zero vectors return 0.0 instead of
    raising or producing NaN. A dimension mismatch raises ValueError.
    Non-finite quotients also return 0.0.
    """
    va = np.asarray(a, dtype=np.float64).ravel()
    vb = np.asarray(b, dtype=np.float64).ravel()
    if va.size == 0 or vb.size == 0:
        return 0.0
    if va.shape != vb.shape:
        raise ValueError("Vectors must have the same dimension.")
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0 or not np.isfinite(denom):
        return 0.0
    value = float(np.dot(va, vb) / denom)
    if not np.isfinite(value):
        return 0.0
    return max(-1.0, min(1.0, value))


def format_results_table(result: dict[str, Any]) -> str:
    """Format the probe result dict as a plain-text table (pure helper)."""

    def _fmt(value: Any) -> str:
        return f"{float(value):.3f}" if value is not None else "n/a"

    lines = [
        "group     mean_cos",
        f"member    {_fmt(result.get('member_mean_cosine'))}",
        f"clean     {_fmt(result.get('clean_mean_cosine'))}",
        f"gap       {_fmt(result.get('gap'))}",
        f"AUC       {_fmt(result.get('auc'))}",
        f"index     {_fmt(result.get('index'))}",
    ]
    return "\n".join(lines)


def run_paraphrase_probe(
    *,
    model_id: str = "experiments/models/sshleifer--tiny-gpt2",
    epochs: int = 1,
    seed: int = 17,
) -> dict[str, Any]:
    """Fine-tune on the leak set and score paraphrase cosines.

    ``model_id`` is passed to ``validation.fine_tune`` as the starting
    checkpoint and to ``load_model(str(out_dir), ...)`` for the adapted
    copy via ``out_dir = MODELS_DIR / f"{model_id.replace('/', '--')}--paraphrase-e{epochs}-s{seed}"``.
    The default is a local ``experiments/models`` checkpoint (offline
    safe); pass ``"sshleifer/tiny-gpt2"`` to start from the HF hub base
    (needs ``HF_HUB_OFFLINE=1`` cache or network). Returns a dict with
    member/clean mean cosines, gap, ``auc``/``index`` and raw vectors.
    """
    import gc

    from contamcheckr.benchmarks import load_benchmark
    from contamcheckr.model import load_model

    try:
        from evaluate import evaluate_scores
        from evasion import template_swap
        from validation import MODELS_DIR, fine_tune, move_model_to_cpu, prepare_data
    except ImportError:  # pragma: no cover - PYTHONPATH=.:experiments preferred
        from experiments.evaluate import evaluate_scores
        from experiments.evasion import template_swap
        from experiments.validation import MODELS_DIR, fine_tune, move_model_to_cpu, prepare_data

    leak_path, clean_path = prepare_data()
    leak = load_benchmark(str(leak_path), split="validation", limit=40)
    clean = load_benchmark(str(clean_path), split="validation", limit=40)
    leak_texts = [example["text"] for example in leak.examples]
    leak_ids = [example["id"] for example in leak.examples]
    clean_texts = [example["text"] for example in clean.examples]
    clean_ids = [example["id"] for example in clean.examples]

    safe_name = model_id.replace("/", "--")
    out_dir = MODELS_DIR / f"{safe_name}--paraphrase-e{epochs}-s{seed}"
    training = fine_tune(
        model_id,
        out_dir,
        leak.examples,
        device_name="auto",
        seed=seed,
        epochs=epochs,
        learning_rate=0.005,
        batch_size=4,
        max_length=256,
        include_order_canaries=False,
    )

    adapted = load_model(str(out_dir), device="auto", seed=seed)
    try:
        member_cos: list[float] = [
            cosine(mean_embedding(adapted, text), mean_embedding(adapted, template_swap(text)))
            for text in leak_texts
        ]
        clean_cos: list[float] = [
            cosine(mean_embedding(adapted, text), mean_embedding(adapted, template_swap(text)))
            for text in clean_texts
        ]
        evaluation = evaluate_scores(member_cos, clean_cos, leak_ids)
    finally:
        move_model_to_cpu(adapted)
        del adapted
        gc.collect()

    member_mean = float(np.mean(member_cos)) if member_cos else None
    clean_mean = float(np.mean(clean_cos)) if clean_cos else None
    gap = (
        member_mean - clean_mean
        if member_mean is not None and clean_mean is not None
        else None
    )
    result: dict[str, Any] = {
        "model_id": model_id,
        "adapted_model": str(out_dir),
        "epochs": epochs,
        "seed": seed,
        "n_members": len(member_cos),
        "n_clean": len(clean_cos),
        "member_mean_cosine": member_mean,
        "clean_mean_cosine": clean_mean,
        "gap": gap,
        "auc": evaluation.get("auc"),
        "index": evaluation.get("index"),
        "member_cosines": member_cos,
        "clean_cosines": clean_cos,
        "member_ids": leak_ids,
        "clean_ids": clean_ids,
        "training": training,
    }
    print(format_results_table(result))
    return result


def parser() -> "Any":
    from argparse import ArgumentParser

    argument_parser = ArgumentParser(
        description="Mean-embedding paraphrase-robustness probe."
    )
    argument_parser.add_argument("--model", default="experiments/models/sshleifer--tiny-gpt2")
    argument_parser.add_argument("--epochs", type=int, default=1)
    argument_parser.add_argument("--seed", type=int, default=17)
    return argument_parser


def main() -> int:
    args = parser().parse_args()
    run_paraphrase_probe(model_id=args.model, epochs=args.epochs, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
