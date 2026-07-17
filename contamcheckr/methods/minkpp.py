"""Min-K%++ scoring on top of the shared token-statistics pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .token_stats import TokenStats, score_token_stats, token_stats_from_logits


@dataclass(frozen=True)
class MinKScores:
    mink: float
    minkpp: float
    token_count: int
    skipped_token_count: int
    loss: float
    truncated: bool = False
    zlib: float = 0.0
    entropy: float = 0.0
    mean_rank: float = 0.0
    top1_prob: float = 0.0


def score_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    k: float = 20.0,
) -> MinKScores:
    """Compatibility wrapper around the shared token-statistics pipeline."""
    stats = token_stats_from_logits(logits, targets)
    scores = score_token_stats(stats, "", k=k)
    return _to_min_k_scores(scores, stats)


def _to_min_k_scores(scores: Any, stats: TokenStats) -> MinKScores:
    return MinKScores(
        mink=scores.mink,
        minkpp=scores.minkpp,
        token_count=scores.token_count,
        skipped_token_count=scores.skipped_token_count,
        loss=scores.loss,
        truncated=stats.truncated,
        zlib=scores.zlib,
        entropy=scores.entropy,
        mean_rank=scores.mean_rank,
        top1_prob=scores.top1_prob,
    )


def score_token_stats_text(
    stats: TokenStats,
    text: str,
    *,
    k: float = 20.0,
) -> MinKScores:
    """Turn cached/computed token statistics into Min-K scores for a text."""
    scores = score_token_stats(stats, text, k=k)
    return _to_min_k_scores(scores, stats)


def score_text(
    model: Any,
    text: str,
    *,
    k: float = 20.0,
    max_tokens: int = 512,
) -> MinKScores:
    token_stats = getattr(model, "token_stats", None)
    if callable(token_stats):
        stats = token_stats(text, max_tokens=max_tokens)
        return score_token_stats_text(stats, text, k=k)
    logits, targets = model.token_logprob_table(text, max_tokens=max_tokens)
    stats = token_stats_from_logits(logits, targets)
    return score_token_stats_text(stats, text, k=k)


def score_text_batch(
    model: Any,
    texts: list[str],
    *,
    k: float = 20.0,
    max_tokens: int = 512,
    batch_size: int = 4,
) -> list[MinKScores]:
    """Score a list of texts, batching the forward pass when possible."""
    if not texts:
        return []
    batch_stats = getattr(model, "token_stats_batch", None)
    if callable(batch_stats):
        stats_list = batch_stats(
            texts, max_tokens=max_tokens, batch_size=batch_size
        )
        return [
            score_token_stats_text(stats, text, k=k)
            for stats, text in zip(stats_list, texts, strict=True)
        ]
    return [
        score_text(model, text, k=k, max_tokens=max_tokens) for text in texts
    ]
