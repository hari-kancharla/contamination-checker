"""Context-windowed token statistics for long texts.

All scoring functions in :mod:`contamcheckr.methods.token_stats` consume a
single forward pass, so long texts are either truncated or require a
context window larger than the model supports.  This module slices the
encoded token ids into windows and scores each window with its own
forward pass, then aggregates per-token statistics across windows.

Window-boundary semantics
-------------------------

Let ``N`` be the number of token ids produced by encoding ``text`` once
(with ``add_special_tokens=False``) and ``W = window_tokens``,
``S = stride_tokens`` (defaulting to ``W``).

* Window ``i`` covers the token ids ``[i*S : i*S + W]``.  Its forward pass
  makes each token in the window a *context* token, but only tokens
  that are **not** the window's first token are *scored* as targets:
  a forward pass over ``[t_0, t_1, ..., t_{L-1}]`` yields logits
  predicting ``t_1, ..., t_{L-1}`` from a correct left-shift, and the
  logits row predicting ``t_0`` (which would need history from before
  the window) does not exist.  The first token of each window therefore
  contributes **nothing** as a target in that window; its context is the
  preceding tokens, which are only present when windows overlap.

* With the default ``stride_tokens=None`` (i.e. ``S = W``, contiguous
  non-overlapping windows), the token at each window start
  (positions ``0, W, 2W, ...``) is dropped: every token except window
  starts is scored exactly once.  The returned ``token_count`` is
  therefore ``N - ceil(N / W)`` for long texts.  In particular the
  global first token is never scored, exactly as in a single-pass
  forward pass.

* With overlapping windows (``S < W``), a token may be a non-first token
  in several windows.  Each token is scored **at most once**: the
  earliest window that covers it as a non-first token wins, so every
  token gets the longest available left context inside the window cap
  (up to ``W - 1`` previous tokens).  Rows already scored by an earlier
  window are discarded ("first-seen wins", no duplicates).  With this
  rule every token except the global first one is scored exactly once,
  whether windows overlap or not.

The returned :class:`TokenStats` aggregates per-token statistics across
windows in token order and always reports ``truncated=False``: no token
is silently dropped merely because the full text exceeds one window's
context (window starts are the only tokens omitted, as documented
above).

Short texts whose full encoded length fits in one window are processed
by a single forward pass over the full token ids, so their statistics
are exactly those of one forward pass.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch

from .token_stats import TokenStats, token_stats_from_logits

ProgressCallback = Callable[[str, int, int], None]

# Sentinel label for progress callbacks, mirroring
# LocalCausalLM.token_stats_batch.
_PROGRESS_LABEL = "Scoring tokens"


def windowed_token_stats(
    model,
    text: str,
    *,
    window_tokens: int = 128,
    stride_tokens: int | None = None,
    progress: ProgressCallback | None = None,
) -> TokenStats:
    """Score ``text`` with context-windowed forward passes.

    Args:
        model: A :class:`contamcheckr.model.LocalCausalLM` exposing
            ``.tokenizer``, ``.model`` (HF causal LM), and ``.device``.
        text: The text to score.  Must produce at least two tokens when
            encoded (without special tokens), consistent with
            ``LocalCausalLM.token_stats``.
        window_tokens: Maximum number of token ids per forward pass.
            Must be at least two.
        stride_tokens: Distance between the starts of consecutive
            windows.  ``None`` selects contiguous non-overlapping windows
            with ``stride_tokens == window_tokens``.  Otherwise must
            satisfy ``1 <= stride_tokens <= window_tokens``.
        progress: Optional callback invoked as
            ``progress(label, done, total)`` after each window, where
            ``total`` is the number of windows.

    Returns:
        Aggregated :class:`TokenStats` (``truncated=False``) whose
        per-token statistics are in token order with no duplicates.

    Raises:
        ValueError: If the text encodes to fewer than two tokens, or if
            ``window_tokens`` / ``stride_tokens`` are invalid.
    """
    if window_tokens < 2:
        raise ValueError("window_tokens must be at least two.")
    stride = window_tokens if stride_tokens is None else stride_tokens
    if stride < 1 or stride > window_tokens:
        raise ValueError(
            "stride_tokens must be between 1 and window_tokens inclusive."
        )
    if not text.strip():
        raise ValueError(
            "Text must produce at least two tokens for scoring."
        )

    # Encode once without special tokens.
    encoded = model.tokenizer.encode(text, add_special_tokens=False)
    token_ids: list[int] = list(encoded)
    total_tokens = len(token_ids)
    if total_tokens < 2:
        raise ValueError(
            "Text must produce at least two tokens for scoring."
        )

    device = model.device
    starts = list(range(0, total_tokens, stride))
    total_windows = len(starts)

    target_logprobs: list[float] = []
    means: list[float] = []
    stds: list[float] = []
    entropies: list[float] = []
    top1_probs: list[float] = []
    ranks: list[int] = []
    seen: set[int] = set()

    for window_index, start in enumerate(starts):
        end = min(start + window_tokens, total_tokens)
        window_ids = torch.tensor(token_ids[start:end], dtype=torch.long)
        if window_ids.shape[0] < 2:
            # A single-token window cannot score any target (its only
            # token is a window start).  Skip the forward pass.
            if progress is not None:
                progress(
                    _PROGRESS_LABEL, window_index + 1, total_windows
                )
            continue
        input_ids = window_ids.unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            logits = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
        # Move the shifted logits rows and target ids to CPU float32
        # before computing statistics, as required.
        logits_cpu = logits[0, :-1, :].float().cpu()
        targets_cpu = input_ids[0, 1:].cpu()
        del logits
        window_stats = token_stats_from_logits(logits_cpu, targets_cpu)

        # Keep only targets (window positions start+1 .. end-1) that no
        # earlier window has scored yet: first-seen wins, no duplicates.
        for j in range(len(window_stats.target_logprobs)):
            position = start + 1 + j
            if position in seen:
                continue
            seen.add(position)
            target_logprobs.append(window_stats.target_logprobs[j])
            means.append(window_stats.means[j])
            stds.append(window_stats.stds[j])
            entropies.append(window_stats.entropies[j])
            top1_probs.append(window_stats.top1_probs[j])
            ranks.append(window_stats.ranks[j])

        if progress is not None:
            progress(_PROGRESS_LABEL, window_index + 1, total_windows)

    if not target_logprobs:
        # Only reachable if the text produced exactly 2 tokens and the
        # windowing logic somehow scored nothing, which the checks above
        # prevent; guard for safety to match token_stats semantics.
        raise ValueError(
            "Text must produce at least two tokens for scoring."
        )

    return TokenStats(
        target_logprobs=tuple(target_logprobs),
        means=tuple(means),
        stds=tuple(stds),
        entropies=tuple(entropies),
        top1_probs=tuple(top1_probs),
        ranks=tuple(ranks),
        token_count=len(target_logprobs),
        truncated=False,
    )


def windowed_token_stats_batch(
    model,
    texts: Sequence[str],
    *,
    window_tokens: int = 128,
    stride_tokens: int | None = None,
    batch_size: int = 4,
) -> list[TokenStats]:
    """Score multiple texts with :func:`windowed_token_stats`.

    This is a plain loop wrapper: each text is scored with
    ``window_tokens``-wide windows (contiguous, non-overlapping).
    ``batch_size`` controls the chunking of the loop only; every text
    still gets its own windowed forward passes.

    Args:
        model: A :class:`contamcheckr.model.LocalCausalLM`.
        texts: The texts to score.
        window_tokens: Maximum number of token ids per forward pass.
        batch_size: Number of texts processed per outer chunk.  Must be
            at least one.

    Returns:
        One :class:`TokenStats` per input text, in order.  An empty
        input yields an empty list.
    """
    if batch_size < 1:
        raise ValueError("Batch size must be at least one.")
    results: list[TokenStats] = []
    items = list(texts)
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        for text in batch:
            results.append(
                windowed_token_stats(model, text, window_tokens=window_tokens, stride_tokens=stride_tokens)
            )
    return results
