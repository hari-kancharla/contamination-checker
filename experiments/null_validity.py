"""Null-validity study: does the audit reject itself on clean models?

Runs audits on uncontaminated base models across matched and
domain-mismatched controls, then checks (a) uniformity of the combined
and per-method p-values, (b) the fraction below 0.05, and (c) how often
the conformal screen false-flags.  Under no contamination these should
be near their nominal rates except in the intentionally mismatched
condition, where the domain check is expected to fire.

Usage:
  HF_HUB_OFFLINE=1 PYTHONPATH=. python experiments/null_validity.py
"""

from __future__ import annotations

import json
from pathlib import Path

from contamcheckr.benchmarks import load_benchmark
from contamcheckr.config import RunConfig
from contamcheckr.model import load_model
from contamcheckr.report import build_report
from suite import easy_control_examples
from validation import prepare_data

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "null-validity.jsonl"

MODELS = ["sshleifer/tiny-gpt2", "experiments/models/distilgpt2"]


def subset(benchmark, kind: str):
    return {
        "first": benchmark.examples[:15],
        "second": benchmark.examples[15:],
        "every-other": benchmark.examples[::2][:15],
    }[kind]


def main() -> int:
    leak_path, clean_path = prepare_data()
    cards = load_benchmark(str(leak_path), split="validation", limit=40)
    cards_clean = load_benchmark(str(clean_path), split="validation", limit=40)
    sample = load_benchmark("sample", limit=30)
    sample_control = load_benchmark("sample", limit=30, sample_control=True)
    easy = easy_control_examples()

    from contamcheckr.benchmarks import BenchmarkData

    easy_bench = BenchmarkData(examples=easy, source="easy", dataset_name="easy", split="validation")
    draws = []
    for model_id in MODELS:
        model = load_model(model_id, device="auto", seed=17)
        conditions = [
            ("cards-matched", cards, cards_clean),
            ("sample-matched", sample, sample_control),
            ("cards-mismatched-easy", cards, easy_bench),
        ]
        for label, benchmark, control in conditions:
            for piece in ("first", "second", "every-other"):
                bench_subset = BenchmarkData(
                    examples=subset(benchmark, piece),
                    source=benchmark.source,
                    dataset_name=benchmark.dataset_name,
                    split=benchmark.split,
                )
                report = build_report(
                    model,
                    bench_subset,
                    control,
                    RunConfig(
                        model_id=model_id,
                        benchmark_source=benchmark.source,
                        control_source=control.source,
                        split=benchmark.split,
                        limit=len(bench_subset.examples),
                        seed=17,
                        shard_count=5,
                        permutation_count=10,
                        max_tokens=256,
                    ),
                )
                draw = {
                    "model": model_id,
                    "condition": label,
                    "piece": piece,
                    "combined_p_value": report.get("combined_p_value"),
                    "method_p_values": report.get("method_p_values"),
                    "contamination_index": report.get("contamination_index"),
                    "n_flagged": (report.get("membership_screen") or {}).get("n_flagged"),
                    "domain_shift_p": (report.get("domain_shift") or {}).get("mann_whitney_p_value"),
                    "control_null_auc": (report.get("control_null") or {}).get("auc"),
                }
                draws.append(draw)
                print(draw, flush=True)
        model.model.to("cpu")
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        "\n".join(json.dumps(draw, sort_keys=True) for draw in draws) + "\n",
        encoding="utf-8",
    )
    nulls = [d for d in draws if "mismatched" not in d["condition"]]
    for label, group in (("matched", nulls), ("all", draws)):
        ps = [d["combined_p_value"] for d in group if d["combined_p_value"] is not None]
        below = sum(p < 0.05 for p in ps) / len(ps) if ps else None
        flagged = sum((d["n_flagged"] or 0) > 0 for d in group)
        print(f"{label}: n={len(ps)} p<0.05={below} screen-fired={flagged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
