"""Timing/throughput benchmark for ``build_report`` (offline-safe).

Loads the local tiny model at ``experiments/models/sshleifer--tiny-gpt2``
on CPU with ``local_files_only=True``, builds deterministic synthetic
benchmarks of 50/200/500 sentences of varying length, and runs
``build_report`` for each size with ``shard_count=2`` and
``permutation_count=2``.  Wall time is measured with
``time.perf_counter`` and printed to stdout.

The score cache is rooted in a ``tempfile`` directory so runs never
pollute the real cache.  If the local model is absent the script exits
gracefully (status 0) with a skip message -- it never touches the network.

Run from the repository root:

    PYTHONPATH=.:experiments .venv/bin/python experiments/stress.py

Set ``HF_HUB_OFFLINE=1`` for belt-and-braces offline use.
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
import time
from pathlib import Path

from contamcheckr.benchmarks import BenchmarkData
from contamcheckr.config import RunConfig
from contamcheckr.model import load_model
from contamcheckr.report import build_report

MODEL_PATH = (
    Path(__file__).resolve().parent / "models" / "sshleifer--tiny-gpt2"
)
MODEL_ID_LABEL = "sshleifer/tiny-gpt2 (local copy)"
SIZES = (50, 200, 500)

_VOCAB = (
    "river mountain garden window silver planet coffee marble velvet harbor "
    "candle puzzle orbit meadow timber walnut crystal emerald lantern "
    "compass orchestra telescope vineyard lighthouse backpack waterfall "
    "sunrise twilight midnight thunder breeze glacier volcano desert "
    "island forest valley castle bridge market museum theater bakery"
).split()

# Deterministic length mix: short/medium/long/targeted-long sentences.
_LENGTHS = (9, 14, 22, 30, 46)


def _sentences(count: int, *, rng: random.Random, offset: int = 0) -> list[str]:
    sentences: list[str] = []
    for index in range(count):
        word_count = _LENGTHS[(offset + index) % len(_LENGTHS)]
        words = [rng.choice(_VOCAB) for _ in range(word_count)]
        sentences.append(" ".join(words).capitalize() + ".")
    return sentences


def _benchmark(texts: list[str], *, prefix: str, source: str) -> BenchmarkData:
    return BenchmarkData(
        examples=[
            {"id": f"{prefix}-{index:04d}", "text": text}
            for index, text in enumerate(texts)
        ],
        source=source,
        dataset_name=source,
        split="test",
    )


def main() -> int:
    if not MODEL_PATH.exists():
        print(
            f"SKIP: local tiny model not found at {MODEL_PATH}; "
            "nothing to benchmark (offline-safe no-op).",
            flush=True,
        )
        return 0

    rng = random.Random(20261107)
    benchmark_texts = _sentences(max(SIZES), rng=rng, offset=0)
    control_texts = _sentences(max(SIZES), rng=rng, offset=1_000)

    try:
        model = load_model(
            str(MODEL_PATH),
            device="cpu",
            local_files_only=True,
            seed=0,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"SKIP: local tiny model could not be loaded: {error}", flush=True)
        return 0

    with tempfile.TemporaryDirectory(prefix="contamcheckr-stress-") as tmpdir:
        cache_path = os.path.join(tmpdir, "stress-cache.sqlite3")
        rows: list[tuple[int, int, float, str, object]] = []
        for size in SIZES:
            benchmark = _benchmark(
                benchmark_texts[:size], prefix="bench", source="stress-bench"
            )
            control = _benchmark(
                control_texts[:size], prefix="ctrl", source="stress-control"
            )
            config = RunConfig(
                model_id=MODEL_ID_LABEL,
                benchmark_source="stress-bench",
                control_source="stress-control",
                limit=size,
                shard_count=2,
                permutation_count=2,
                device="cpu",
                local_files_only=True,
                cache_path=cache_path,
                seed=0,
            )
            started = time.perf_counter()
            report = build_report(model, benchmark, control, config)
            elapsed = time.perf_counter() - started
            rows.append(
                (
                    size,
                    report["example_count"],
                    elapsed,
                    report["band"],
                    report["contamination_index"],
                )
            )

    print()
    print(f"{'limit':>6}  {'examples':>8}  {'wall_s':>9}  {'band':<22}  contamination_index")
    print(f"{'-' * 6}  {'-' * 8}  {'-' * 9}  {'-' * 22}  {'-' * 19}")
    for size, examples, elapsed, band, index in rows:
        index_text = f"{index:.4f}" if index is not None else "None"
        print(f"{size:>6}  {examples:>8}  {elapsed:>9.3f}  {band:<22}  {index_text}")
    print()
    print(f"model: {MODEL_PATH}")
    print(f"cache: temporary directory (discarded after run)")
    print(f"device: cpu, shard_count=2, permutation_count=2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
