#!/usr/bin/env python3
"""Offline audit of the released OLMo-Detect pretraining texts and saved scores.

This program imports no project or upstream code, loads no model, and makes no
network requests. It reconstructs published-score AUCs, not fresh model scores.
Run with --help for inputs. Output directories must not already exist.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import platform
import re
import sys
import tarfile
from typing import Any


REVISION = "6eb88de34a09dea85c4b41deff4048e5a203e1b0"
ARCHIVE_SHA256 = "d7a157423bf537a343975db13bfddd9b271a7c24a9cb7ecc8f6b2342ca8d30ce"
PAPER_URL = "https://arxiv.org/html/2610.02986v1"
SIZES = ("1B", "7B", "13B", "32B")
METHODS = {
    "LOSS": ("loss_zlib_lowercase", "loss_score"),
    "MinK": ("minkprob", "min_k_prob_mean_logprob"),
    "MinK++": ("minkprob", "min_k_pp_score"),
}
DOMAINS = {"dclm": ("",), "pes2o": ("",), "openwebmath": ("",),
           "starcoder": ("java", "python")}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def auc_pairs(positive: list[float], negative: list[float]) -> float | None:
    """P(S_member > S_nonmember) + .5 P(tie), using binary searches."""
    if not positive or not negative:
        return None
    ordered = sorted(negative)
    twice_wins = sum(bisect_left(ordered, p) + bisect_right(ordered, p)
                     for p in positive)
    return twice_wins / (2 * len(positive) * len(negative))


def auc_ranks(positive: list[float], negative: list[float]) -> float | None:
    """Independent Mann-Whitney rank-sum calculation with midranks."""
    if not positive or not negative:
        return None
    combined = sorted([(s, 1) for s in positive] + [(s, 0) for s in negative])
    rank_sum = 0.0
    start = 0
    while start < len(combined):
        stop = start + 1
        while stop < len(combined) and combined[stop][0] == combined[start][0]:
            stop += 1
        midrank = (start + 1 + stop) / 2
        rank_sum += midrank * sum(label for _, label in combined[start:stop])
        start = stop
    n = len(positive)
    return (rank_sum - n * (n + 1) / 2) / (n * len(negative))


def finite_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def count_summary(counts: list[int]) -> dict:
    return {
        "n": len(counts), "min": min(counts) if counts else None,
        "max": max(counts) if counts else None,
        "above_1023": sum(n > 1023 for n in counts),
        "above_1024": sum(n > 1024 for n in counts),
        "equal_4095": counts.count(4095),
        "above_4095": sum(n > 4095 for n in counts),
    }
