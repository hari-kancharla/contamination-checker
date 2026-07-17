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
