"""Larger-model replication: same contamination protocol on pythia-160m.

Fine-tunes EleutherAI/pythia-160m on the cards leak set, then audits
adapted-on-leak vs adapted-on-clean vs base-on-leak through the normal
build_report pipeline, printing AUCs / indices and the validity
diagnostics.  Freezes check: compare the Min-K%++/loss AUC trend with
the tiny-gpt2 and distilgpt2 cells in experiments/results/paper-suite.jsonl.

Usage (downloads required once):
  PYTHONPATH=. .venv/bin/python experiments/larger_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validation import (
    MODELS_DIR,
    fine_tune,
    move_model_to_cpu,
    prepare_data,
)
from contamcheckr.benchmarks import BenchmarkData, load_benchmark
from contamcheckr.config import RunConfig
from contamcheckr.model import load_model
from contamcheckr.report import build_report

MODEL_ID = "EleutherAI/pythia-160m"
OUTPUT = Path(__file__).resolve().parent / "results" / "larger-probe.jsonl"


def audit(model, benchmark: BenchmarkData, control: BenchmarkData) -> dict[str, object]:
    report = build_report(
        model,
        benchmark,
        control,
        RunConfig(
            model_id=str(getattr(model, "model_id", "pythia")),
            benchmark_source=benchmark.source,
            control_source=control.source,
            split=benchmark.split,
            limit=len(benchmark.examples),
            seed=17,
            shard_count=5,
            permutation_count=10,
            max_tokens=256,
        ),
    )
    return {
        "contamination_index": report["contamination_index"],
        "band": report["band"],
        "minkpp_auc": report["minkpp_auc"],
        "loss_auc": (report.get("likelihood_baseline") or {}).get("loss_auc"),
        "combined_p_value": report.get("combined_p_value"),
        "n_flagged": (report.get("membership_screen") or {}).get("n_flagged"),
        "domain_fingerprint_digit_p": (
            report.get("domain_fingerprint") or {}
        ).get("digit_mann_whitney_p_value"),
    }


def main() -> int:
    leak_path, clean_path = prepare_data()
    leak = load_benchmark(str(leak_path), split="validation", limit=40)
    clean = load_benchmark(str(clean_path), split="validation", limit=40)
    out_dir = MODELS_DIR / "larger-pythia-160m-e5"
    print("fine-tuning pythia-160m 5 epochs on the leak set...", flush=True)
    fine_tune(
        MODEL_ID,
        out_dir,
        leak.examples,
        device_name="auto",
        seed=17,
        epochs=5,
        learning_rate=0.0002,
        batch_size=2,
        max_length=256,
        include_order_canaries=False,
    )
    adapted = load_model(str(out_dir), device="auto", seed=17)
    leak_audit = audit(adapted, leak, clean)
    clean_audit = audit(adapted, clean, leak)
    move_model_to_cpu(adapted)
    del adapted
    base = load_model(MODEL_ID, device="auto", seed=17)
    base_audit = audit(base, leak, clean)
    move_model_to_cpu(base)
    row = {
        "model": MODEL_ID,
        "epochs": 5,
        "adapted_on_leak": leak_audit,
        "adapted_on_clean": clean_audit,
        "base_on_leak": base_audit,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
