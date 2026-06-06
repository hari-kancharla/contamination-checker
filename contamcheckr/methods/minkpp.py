from dataclasses import dataclass

import numpy as np
import torch

from ..metrics import lowest_percent


@dataclass(frozen=True)
class MinKScores:
    mink: float
    minkpp: float
    token_count: int
    skipped_token_count: int


def score_logits(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    k: float = 20.0,
) -> MinKScores:
    if logits.ndim != 2:
        raise ValueError("Logits must have shape [tokens, vocabulary].")
    if target_ids.ndim != 1 or len(target_ids) != len(logits):
        raise ValueError("Target ids must contain one id per logits row.")
    if len(target_ids) == 0:
        raise ValueError("Min-K scoring requires at least one target token.")
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = torch.softmax(logits.float(), dim=-1)
    raw_scores: list[float] = []
    normalized_scores: list[float] = []
    skipped = 0
    for row_index, (row, target_id) in enumerate(
        zip(log_probs, target_ids, strict=True)
    ):
        target = int(target_id.item())
        if target < 0 or target >= row.shape[0]:
            raise ValueError(f"Target token id {target} is outside the vocabulary.")
        probability_row = probs[row_index]
        target_logprob = float(row[target].item())
        mu = torch.sum(probability_row * row)
        variance = torch.sum(probability_row * (row - mu).square())
        sigma = torch.sqrt(torch.clamp(variance, min=0.0))
        mu_value = float(mu.item())
        sigma_value = float(sigma.item())
        if not np.isfinite(target_logprob) or not np.isfinite(mu_value):
            skipped += 1
            continue
        raw_scores.append(target_logprob)
        if sigma_value == 0.0 or not np.isfinite(sigma_value):
            skipped += 1
            continue
        normalized = (target_logprob - mu_value) / sigma_value
        if np.isfinite(normalized):
            normalized_scores.append(normalized)
        else:
            skipped += 1
    if not raw_scores or not normalized_scores:
        raise ValueError("No tokens had a finite non-zero Min-K%++ variance.")
    return MinKScores(
        mink=float(np.mean(lowest_percent(raw_scores, k))),
        minkpp=float(np.mean(lowest_percent(normalized_scores, k))),
        token_count=len(normalized_scores),
        skipped_token_count=skipped,
    )


def score_text(
    model: object,
    text: str,
    *,
    k: float = 20.0,
    max_tokens: int = 512,
) -> MinKScores:
    logits, target_ids = model.token_logprob_table(text, max_tokens=max_tokens)
    return score_logits(logits, target_ids, k=k)
