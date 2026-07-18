"""REAL-benchmark membership experiment (GSM8K, pythia-160m).

Fine-tunes pythia-160m on 200 GSM8K train items, then scores those
200 (members) against 200 held-out train items (clean) with Min-K%++
and the loss baseline.  Also scores the base model as the null.
Everything else in the repo is synthetic; this is the grounded
membership-recovery check.  Results append to
experiments/results/gsm8k-probe.jsonl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validation import MODELS_DIR, fine_tune, move_model_to_cpu

RESULTS = Path(__file__).resolve().parent / "results" / "gsm8k-probe.jsonl"


def gsm8k_items(count: int = 200, offset: int = 0) -> list[dict[str, str]]:
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    items = []
    for row in dataset.select(range(offset, offset + count)):
        items.append(
            {
                "id": f"gsm-{offset}-{len(items):04d}",
                "text": f"Question: {row['question']}\n\nAnswer: {row['answer']}",
            }
        )
    return items


def scored(model, items, *, label: str):
    from contamcheckr.methods.minkpp import score_text_batch

    scores = score_text_batch(
        model, [item["text"] for item in items], max_tokens=512, batch_size=4
    )
    print(f"scored {label}: {len(scores)}", flush=True)
    return {
        "minkpp": [score.minkpp for score in scores],
        "loss": [score.loss for score in scores],
        "ids": [item["id"] for item in items],
    }


def main() -> int:
    from contamcheckr.model import load_model
    from evaluate import evaluate_scores

    leak = gsm8k_items(200, offset=0)
    clean = gsm8k_items(200, offset=200)
    model_id = "EleutherAI/pythia-160m"
    out_dir = MODELS_DIR / "gsm8k-pythia-e5"
    print("fine-tuning pythia-160m on 200 GSM8K items (5 epochs)...", flush=True)
    fine_tune(
        model_id,
        out_dir,
        leak,
        device_name="cpu",
        seed=17,
        epochs=5,
        learning_rate=0.00005,
        batch_size=2,
        max_length=512,
        include_order_canaries=False,
    )
    from contamcheckr.benchmarks import BenchmarkData

    def wrap(items) -> BenchmarkData:
        return BenchmarkData(
            examples=items, source="gsm8k", dataset_name="gsm8k", split="train"
        )

    from contamcheckr.report import build_report
    from contamcheckr.config import RunConfig

    def audit(model, bench, ctrl):
        report = build_report(
            model,
            wrap(bench),
            wrap(ctrl),
            RunConfig(
                model_id=model_id,
                benchmark_source="gsm8k",
                control_source="gsm8k",
                split="train",
                seed=17,
                shard_count=2,
                permutation_count=2,
            ),
        )
        return {
            "minkpp_auc": report["minkpp_auc"],
            "loss_auc": (report.get("likelihood_baseline") or {}).get("loss_auc"),
            "index_basis": report["index_basis"],
            "contamination_index": report["contamination_index"],
            "combined_p_value": report.get("combined_p_value"),
            "n_flagged": (report.get("membership_screen") or {}).get("n_flagged"),
        }

    row: dict[str, object] = {}
    adapted = load_model(str(out_dir), device="auto", seed=17)
    row["adapted"] = audit(adapted, leak, clean)
    move_model_to_cpu(adapted)
    del adapted
    base = load_model(model_id, device="auto", seed=17)
    row["base"] = audit(base, leak, clean)
    move_model_to_cpu(base)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
