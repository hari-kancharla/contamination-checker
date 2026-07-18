"""Multi-seed controlled-contamination suite for the paper experiments.

Each cell fine-tunes a copy of a base model on a leak set, then audits
three conditions (adapted-on-leak, adapted-on-clean, base-on-leak)
through the normal ``build_report`` pipeline.  Results append to a
JSONL file so long suites can resume and table generation never
recomputes.  Base audits are shared across cells with the same model
and dataset.

Usage:
  HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python experiments/suite.py
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from contamcheckr.benchmarks import BenchmarkData, load_benchmark
from contamcheckr.config import RunConfig
from contamcheckr.model import load_model
from contamcheckr.report import build_report
from validation import (
    MODELS_DIR,
    ROOT,
    fine_tune,
    move_model_to_cpu,
    prepare_data,
)

RESULTS = ROOT / "results" / "paper-suite.jsonl"


def easy_control_examples(prefix: str = "EZ", count: int = 40) -> list[dict[str, str]]:
    phrases = [
        "The lamp glows.",
        "Birds fly south.",
        "Tea cools slowly.",
        "Doors open wide.",
        "Rain taps glass.",
    ]
    return [
        {"id": f"{prefix.lower()}-{index:03d}", "text": phrases[index % len(phrases)]}
        for index in range(count)
    ]


def suite_config(model_id: str, *, seed: int) -> RunConfig:
    return RunConfig(
        model_id=model_id,
        benchmark_source="suite",
        split="validation",
        limit=24,
        device="auto",
        seed=seed,
        shard_count=5,
        permutation_count=10,
        max_tokens=256,
        min_calibration_examples=4,
    )


def compact(report: dict[str, object]) -> dict[str, object]:
    """Keep the score vectors the paper tables need, drop the bulk."""
    return {
        "contamination_index": report["contamination_index"],
        "band": report["band"],
        "index_basis": report["index_basis"],
        "index_minkpp": report["index_minkpp"],
        "minkpp_auc": report["minkpp_auc"],
        "minkpp_summary": report["minkpp_summary"],
        "likelihood_baseline": report["likelihood_baseline"],
        "score_baselines": report.get("score_baselines", {}),
        "control_null": report.get("control_null"),
        "domain_shift": report.get("domain_shift"),
        "membership_screen": report.get("membership_screen"),
        "order_p_value": report.get("order_p_value"),
        "method_p_values": report.get("method_p_values"),
        "combined_p_value": report.get("combined_p_value"),
        "combined_p_value_basis": report.get("combined_p_value_basis"),
        "guided_general_gap": report.get("guided_general_gap"),
        "guided_mismatched_gap": report.get("guided_mismatched_gap"),
        "per_example": report["per_example"],
        "control_per_example": report.get("control_per_example", []),
    }


def audit(
    model, benchmark: BenchmarkData, control: BenchmarkData, *, seed: int
) -> dict[str, object]:
    config = suite_config(str(getattr(model, "model_id", "model")), seed=seed)
    return compact(build_report(model, benchmark, control, config))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-seeds", default="17,23")
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    seeds = [int(value) for value in args.limit_seeds.split(",") if value]

    leak_path, clean_path = prepare_data()
    cards_leak = load_benchmark(str(leak_path), split="validation", limit=40)
    cards_clean = load_benchmark(str(clean_path), split="validation", limit=40)
    easy_control = BenchmarkData(
        examples=easy_control_examples(),
        source="easy-control",
        dataset_name="easy control",
        split="validation",
    )
    # NOTE: experiments/models/sshleifer--tiny-gpt2 is the historical
    # canary-trained adapted copy, NOT the base. Tiny must load from the
    # HF hub cache ("sshleifer/tiny-gpt2") for clean bases.
    tiny_id = "sshleifer/tiny-gpt2"
    _gpt_local = MODELS_DIR / "distilgpt2"
    distil_id = str(_gpt_local) if _gpt_local.exists() else "distilgpt2"
    cells = []
    for epoch in (1, 5, 20, 60):
        for seed in seeds:
            cells.append((tiny_id, epoch, seed))
    cells += [(distil_id, 5, 17), (distil_id, 20, 17)]

    seen: set[tuple[str, int, int]] = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            seen.add((row["model_id"], row["epochs"], row["seed"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    handle = args.output.open("a", encoding="utf-8")

    for model_id, epochs, seed in cells:
        key = (model_id, epochs, seed)
        if key in seen:
            print(f"skip {key}")
            continue
        print(f"\n=== cell {key} ===", flush=True)
        safe = model_id.replace("/", "--")
        out_dir = MODELS_DIR / f"suite-{safe}-e{epochs}-s{seed}"
        fine_tune(
            model_id,
            out_dir,
            cards_leak.examples,
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
            leak_audit = audit(adapted, cards_leak, cards_clean, seed=seed)
            clean_audit = audit(adapted, cards_clean, cards_leak, seed=seed)
        finally:
            move_model_to_cpu(adapted)
            del adapted
            gc.collect()
        row = {
            "model_id": model_id,
            "epochs": epochs,
            "seed": seed,
            "adapted_on_leak": leak_audit,
            "adapted_on_clean": clean_audit,
        }
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        print(
            f"leak index={leak_audit['index_basis']}: "
            f"{leak_audit['contamination_index']}, "
            f"clean index={clean_audit['contamination_index']}",
            flush=True,
        )

    # Mismatch condition: base model, cards benchmark, easy control.
    base = load_model(tiny_id, device="auto", seed=17)
    mismatch = audit(base, cards_leak, easy_control, seed=17)
    matched = audit(base, cards_leak, cards_clean, seed=17)
    row = {
        "model_id": "sshleifer/tiny-gpt2",
        "epochs": 0,
        "seed": 17,
        "mismatch_easy_control": mismatch,
        "matched_control": matched,
    }
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()
    move_model_to_cpu(base)
    handle.close()
    print("suite complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
