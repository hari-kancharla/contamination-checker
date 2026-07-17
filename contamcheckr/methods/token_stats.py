"""Token-level statistics shared by Min-K, Min-K%++, LOSS, and Zlib.

All methods in this module consume a single forward pass.  The model wrapper
produces :class:`TokenStats`; the scoring functions turn those statistics into
membership-style scores without keeping the full vocabulary logits in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
import zlib

import numpy as np
import torch

from ..metrics import lowest_percent


@dataclass(frozen=True)
class TokenStats:
    """Per-token statistics from one causal-LM forward pass.

    ``means`` and ``stds`` are the probability-weighted mean and standard
    deviation of the full next-token log-probability distribution.  They are
    the normalization terms used by Min-K%++.
    """

    target_logprobs: tuple[float, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    entropies: tuple[float, ...]
    top1_probs: tuple[float, ...]
    ranks: tuple[int, ...]
    token_count: int
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_logprobs": list(self.target_logprobs),
            "means": list(self.means),
            "stds": list(self.stds),
            "entropies": list(self.entropies),
            "top1_probs": list(self.top1_probs),
            "ranks": list(self.ranks),
            "token_count": self.token_count,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TokenStats":
        return cls(
            target_logprobs=tuple(float(value) for value in payload["target_logprobs"]),
            means=tuple(float(value) for value in payload["means"]),
            stds=tuple(float(value) for value in payload["stds"]),
            entropies=tuple(float(value) for value in payload["entropies"]),
            top1_probs=tuple(float(value) for value in payload["top1_probs"]),
            ranks=tuple(int(value) for value in payload["ranks"]),
            token_count=int(payload["token_count"]),
            truncated=bool(payload.get("truncated", False)),
        )


@dataclass(frozen=True)
class TokenScores:
    """Aggregate scores derived from :class:`TokenStats`."""

    mink: float
    minkpp: float
    loss: float
    zlib: float
    entropy: float
    top1_prob: float
    mean_rank: float
    token_count: int
    skipped_token_count: int
    zlib_ratio: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mink": self.mink,
            "minkpp": self.minkpp,
            "loss": self.loss,
            "zlib": self.zlib,
            "entropy": self.entropy,
            "top1_prob": self.top1_prob,
            "mean_rank": self.mean_rank,
            "token_count": self.token_count,
            "skipped_token_count": self.skipped_token_count,
            "zlib_ratio": self.zlib_ratio,
        }


def token_stats_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    token_chunk_size: int = 64,
    truncated: bool = False,
) -> TokenStats:
    """Compute per-token statistics from ``[tokens, vocabulary]`` logits.

    The vocabulary softmax is evaluated in token chunks so that a long sequence
    does not require several full ``[tokens, vocabulary]`` copies in memory.
    """
    if logits.ndim != 2:
        raise ValueError("Logits must have shape [tokens, vocabulary].")
    if targets.ndim != 1 or len(targets) != len(logits):
        raise ValueError("Target ids must contain one id per logits row.")
    if len(targets) == 0:
        raise ValueError("Token statistics require at least one target token.")
    if token_chunk_size < 1:
        raise ValueError("Token chunk size must be at least one.")

    target_logprobs: list[float] = []
    means: list[float] = []
    stds: list[float] = []
    entropies: list[float] = []
    top1_probs: list[float] = []
    ranks: list[int] = []

    with torch.inference_mode():
        for start in range(0, len(targets), token_chunk_size):
            block_logits = logits[start : start + token_chunk_size].float()
            block_targets = targets[start : start + token_chunk_size].to(
                block_logits.device
            )
            log_probs = torch.log_softmax(block_logits, dim=-1)
            probabilities = log_probs.exp()
            target_logprob = log_probs.gather(
                1, block_targets.unsqueeze(1)
            ).squeeze(1)
            mean = torch.sum(probabilities * log_probs, dim=-1)
            variance = torch.sum(
                probabilities * (log_probs - mean.unsqueeze(1)).square(),
                dim=-1,
            )
            standard_deviation = torch.sqrt(torch.clamp(variance, min=0.0))
            top1 = torch.max(probabilities, dim=-1).values
            rank = torch.sum(
                log_probs > target_logprob.unsqueeze(1), dim=-1
            ) + 1

            target_logprobs.extend(target_logprob.detach().cpu().tolist())
            means.extend(mean.detach().cpu().tolist())
            stds.extend(standard_deviation.detach().cpu().tolist())
            entropies.extend((-mean).detach().cpu().tolist())
            top1_probs.extend(top1.detach().cpu().tolist())
            ranks.extend(int(value) for value in rank.detach().cpu().tolist())

    return TokenStats(
        target_logprobs=tuple(target_logprobs),
        means=tuple(means),
        stds=tuple(stds),
        entropies=tuple(entropies),
        top1_probs=tuple(top1_probs),
        ranks=tuple(ranks),
        token_count=len(target_logprobs),
        truncated=truncated,
    )


def zlib_ratio(text: str) -> float:
    """Compressed bytes per raw byte, clipped away from zero."""
    raw = text.encode("utf-8")
    if not raw:
        return 1.0
    return max(len(zlib.compress(raw, level=9)) / len(raw), 1.0e-9)


def score_token_stats(
    stats: TokenStats,
    text: str,
    *,
    k: float = 20.0,
) -> TokenScores:
    """Turn per-token statistics into aggregate membership-style scores."""
    if not 0.0 < k <= 100.0:
        raise ValueError("k must be greater than 0 and at most 100.")
    if stats.token_count == 0:
        raise ValueError("Token statistics are empty.")

    raw_scores: list[float] = []
    normalized_scores: list[float] = []
    entropies: list[float] = []
    top1_probs: list[float] = []
    ranks: list[float] = []
    skipped = 0
    for index in range(stats.token_count):
        target_logprob = stats.target_logprobs[index]
        mean = stats.means[index]
        sigma = stats.stds[index]
        entropy = stats.entropies[index]
        top1 = stats.top1_probs[index]
        rank = stats.ranks[index]
        if not math.isfinite(target_logprob) or not math.isfinite(mean):
            skipped += 1
            continue
        raw_scores.append(target_logprob)
        entropies.append(entropy)
        top1_probs.append(top1)
        ranks.append(float(rank))
        if sigma == 0.0 or not math.isfinite(sigma):
            skipped += 1
            continue
        normalized = (target_logprob - mean) / sigma
        if math.isfinite(normalized):
            normalized_scores.append(normalized)
        else:
            skipped += 1

    if not raw_scores:
        raise ValueError("No tokens had a finite log-probability.")
    if not normalized_scores:
        raise ValueError("No tokens had a finite non-zero Min-K%++ variance.")

    compression = zlib_ratio(text)
    mean_logprob = float(np.mean(raw_scores))
    return TokenScores(
        mink=float(np.mean(lowest_percent(raw_scores, k))),
        minkpp=float(np.mean(lowest_percent(normalized_scores, k))),
        loss=mean_logprob,
        zlib=mean_logprob + math.log(compression),
        entropy=float(np.mean(entropies)),
        top1_prob=float(np.mean(top1_probs)),
        mean_rank=float(np.mean(ranks)),
        token_count=len(normalized_scores),
        skipped_token_count=skipped,
        zlib_ratio=compression,
    )
